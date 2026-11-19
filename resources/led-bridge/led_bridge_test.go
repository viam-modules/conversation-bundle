package ledbridge

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"go.bug.st/serial"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	generic "go.viam.com/rdk/services/generic"
)

// fakeSource is a status_source whose state the test sets.
type fakeSource struct {
	mu    sync.Mutex
	state string
}

func (f *fakeSource) Name() resource.Name { return generic.Named("source") }

func (f *fakeSource) DoCommand(context.Context, map[string]interface{}) (map[string]interface{}, error) {
	return nil, nil
}

func (f *fakeSource) Status(context.Context) (map[string]interface{}, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return map[string]interface{}{"state": f.state}, nil
}

func (f *fakeSource) Close(context.Context) error { return nil }

// fakePort is a serial.Port whose firmware side is driven by the test:
// reply() makes the firmware print a line, unplug() makes reads fail the way
// a disconnected adapter does.
type fakePort struct {
	serial.Port // unused methods panic

	fromFirmware *io.PipeReader
	firmware     *io.PipeWriter

	mu      sync.Mutex
	written []string
	rts     []bool
	dtr     []bool
	closed  bool
}

func newFakePort() *fakePort {
	r, w := io.Pipe()
	return &fakePort{fromFirmware: r, firmware: w}
}

func (p *fakePort) Read(b []byte) (int, error) { return p.fromFirmware.Read(b) }

func (p *fakePort) Write(b []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return 0, errors.New("port closed")
	}
	p.written = append(p.written, strings.TrimSpace(string(b)))
	return len(b), nil
}

func (p *fakePort) SetRTS(v bool) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.rts = append(p.rts, v)
	return nil
}

func (p *fakePort) SetDTR(v bool) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.dtr = append(p.dtr, v)
	return nil
}

func (p *fakePort) Close() error {
	p.mu.Lock()
	p.closed = true
	p.mu.Unlock()
	return p.fromFirmware.Close()
}

func (p *fakePort) reply(t *testing.T, line string) {
	t.Helper()
	if _, err := p.firmware.Write([]byte(line + "\n")); err != nil {
		t.Fatalf("firmware reply: %v", err)
	}
}

func (p *fakePort) unplug() { p.firmware.CloseWithError(errors.New("input/output error")) }

func (p *fakePort) writes() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.written...)
}

func (p *fakePort) rtsCalls() []bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]bool(nil), p.rts...)
}

// testBridge builds a bridge with short timings whose every open hands out
// a fresh fakePort, recorded in opened.
type testBridge struct {
	*bridge
	mu     sync.Mutex
	opened []*fakePort
}

func newTestBridge(t *testing.T, src *fakeSource) *testBridge {
	t.Helper()
	b := newBridge(generic.Named("led-bridge"), logging.NewTestLogger(t), src,
		"/dev/fake", defaultBaudRate, 5*time.Millisecond, defaultStateKey)
	b.reconnectBackoff = 20 * time.Millisecond
	b.resendInterval = 30 * time.Millisecond
	b.unresponsiveAfter = 100 * time.Millisecond
	b.resetCooldown = time.Hour
	tb := &testBridge{bridge: b}
	b.openPort = func(string, *serial.Mode) (serial.Port, error) {
		p := newFakePort()
		tb.mu.Lock()
		tb.opened = append(tb.opened, p)
		tb.mu.Unlock()
		return p, nil
	}
	t.Cleanup(func() {
		if err := b.Close(context.Background()); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return tb
}

func (tb *testBridge) ports() []*fakePort {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	return append([]*fakePort(nil), tb.opened...)
}

func (tb *testBridge) latestPort(t *testing.T) *fakePort {
	t.Helper()
	ports := tb.ports()
	if len(ports) == 0 {
		t.Fatal("no port opened")
	}
	return ports[len(ports)-1]
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

func countOf(lines []string, want string) int {
	n := 0
	for _, l := range lines {
		if l == want {
			n++
		}
	}
	return n
}

func TestResendsAnUnchangedState(t *testing.T) {
	tb := newTestBridge(t, &fakeSource{state: "idle"})
	tb.start()

	eventually(t, "idle to be written three times", func() bool {
		ports := tb.ports()
		return len(ports) > 0 && countOf(ports[0].writes(), `{"state":"idle"}`) >= 3
	})
}

func TestDropsAndReopensAnUnpluggedPort(t *testing.T) {
	tb := newTestBridge(t, &fakeSource{state: "responding"})
	tb.start()
	eventually(t, "the first write", func() bool {
		ports := tb.ports()
		return len(ports) > 0 && len(ports[0].writes()) > 0
	})

	tb.latestPort(t).unplug()
	eventually(t, "the port to be reopened", func() bool { return len(tb.ports()) >= 2 })
	eventually(t, "the state to be written to the new port", func() bool {
		return countOf(tb.ports()[1].writes(), `{"state":"responding"}`) > 0
	})
}

func TestStatusShowsADroppedPortAsDisconnected(t *testing.T) {
	tb := newTestBridge(t, &fakeSource{})
	tb.reconnectBackoff = time.Hour
	tb.connMu.Lock()
	if _, err := tb.openLocked(); err != nil {
		t.Fatalf("open: %v", err)
	}
	tb.connMu.Unlock()

	tb.latestPort(t).unplug()
	eventually(t, "Status to report disconnected", func() bool {
		status, err := tb.Status(context.Background())
		return err == nil && status["connected"] == false
	})
}

func TestResetsFirmwareThatStopsAnswering(t *testing.T) {
	tb := newTestBridge(t, &fakeSource{state: "responding"})
	tb.start()
	eventually(t, "the first write", func() bool {
		ports := tb.ports()
		return len(ports) > 0 && len(ports[0].writes()) > 0
	})
	port := tb.latestPort(t)
	port.reply(t, "-> mode: responding")

	eventually(t, "an RTS reset pulse", func() bool {
		rts := port.rtsCalls()
		return len(rts) == 2 && rts[0] && !rts[1]
	})
	status, err := tb.Status(context.Background())
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if status["firmware_resets"] != int64(1) {
		t.Fatalf("firmware_resets = %v, want 1", status["firmware_resets"])
	}
}

func TestLeavesAnsweringFirmwareAlone(t *testing.T) {
	tb := newTestBridge(t, &fakeSource{state: "idle"})
	tb.start()
	eventually(t, "the first write", func() bool {
		ports := tb.ports()
		return len(ports) > 0 && len(ports[0].writes()) > 0
	})
	port := tb.latestPort(t)

	// Echo every write, as the reference firmware does.
	seen := 0
	deadline := time.Now().Add(3 * tb.unresponsiveAfter)
	for time.Now().Before(deadline) {
		writes := port.writes()
		for ; seen < len(writes); seen++ {
			port.reply(t, "received line: "+writes[seen])
		}
		time.Sleep(2 * time.Millisecond)
	}
	if rts := port.rtsCalls(); len(rts) != 0 {
		t.Fatalf("answering firmware was reset (RTS calls %v)", rts)
	}
}

func TestNeverResetsFirmwareThatNeverAnswers(t *testing.T) {
	tb := newTestBridge(t, &fakeSource{state: "idle"})
	tb.start()
	time.Sleep(3 * tb.unresponsiveAfter)

	for _, p := range tb.ports() {
		if rts := p.rtsCalls(); len(rts) != 0 {
			t.Fatalf("silent firmware was reset (RTS calls %v)", rts)
		}
	}
}

func TestIsFirmwareReply(t *testing.T) {
	cases := map[string]bool{
		"ready":                               true,
		"received line: {\"state\":\"idle\"}": true,
		"-> mode: idle\r":                     true,
		"waiting for download":                false,
		"rst:0x1 (POWERON_RESET),boot:0x3":    false,
		"":                                    false,
	}
	for line, want := range cases {
		if got := isFirmwareReply(line); got != want {
			t.Errorf("isFirmwareReply(%q) = %v, want %v", line, got, want)
		}
	}
}
