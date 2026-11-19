// Package ledbridge provides the viam:conversation-bundle:led-bridge model: a
// generic resource that polls another resource's Status() "state" field and
// drives a USB-serial LED indicator firmware to match. The source (named by
// status_source, e.g. voice-command) knows nothing about LEDs; the firmware (a
// separate ESP32 sketch) owns the visuals.
//
// The link is treated as lossy: the USB-serial adapter can drop off the bus
// and the ESP32 can stop running its sketch, and a strip keeps showing its
// last colour until it is told otherwise. So the bridge resends the current
// state every resendInterval rather than only on change, drops the port as
// soon as a read fails, and hard-resets an ESP32 that has stopped answering.
package ledbridge

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.bug.st/serial"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	generic "go.viam.com/rdk/services/generic"
)

var Model = resource.NewModel("viam", "conversation-bundle", "led-bridge")

const (
	defaultBaudRate     = 115200
	defaultPollInterval = 200 * time.Millisecond
	defaultStateKey     = "state"
	// reconnectBackoff throttles reconnect attempts while the device is down.
	// Opening the port resets the ESP, so unthrottled retries would reboot-loop it.
	reconnectBackoff = 2 * time.Second
	// resendInterval is how often the current state is written again while it
	// is unchanged. A line can be lost while the adapter re-enumerates or the
	// ESP32 boots, and the firmware treats a repeated state as a no-op.
	resendInterval = 3 * time.Second
	// unresponsiveAfter is how long writes may go unanswered before the ESP32
	// is reset. It spans several resends and the firmware's ~2.5 s boot.
	unresponsiveAfter = 10 * time.Second
	// resetCooldown keeps a board that never comes back from being reset in a
	// tight loop.
	resetCooldown = 30 * time.Second
	// resetPulse is how long EN is held low, as esptool does.
	resetPulse = 100 * time.Millisecond
)

// errBackoff means the port is disconnected and we're waiting out
// reconnectBackoff before redialing — not a real I/O failure to report.
var errBackoff = errors.New("serial reconnect backoff")

func init() {
	resource.RegisterService(generic.API, Model,
		resource.Registration[resource.Resource, *Config]{
			Constructor: newService,
		},
	)
}

type Config struct {
	// SerialPort is the USB-serial device path (e.g. /dev/ttyUSB0, or
	// /dev/cu.usbserial-XXXX on macOS).
	SerialPort string `json:"serial_port"`

	// StatusSource is the resource to watch; led-bridge reacts to the StateKey
	// field of its Status(). Any resource exposing that key as a string works.
	StatusSource string `json:"status_source"`

	// StateKey is the field in status_source's Status() map to read. Optional;
	// defaults to "state".
	StateKey string `json:"state_key,omitempty"`

	// BaudRate is optional; defaults to 115200 to match the firmware.
	BaudRate int `json:"baud_rate,omitempty"`

	// PollIntervalMs is optional; how often to poll Status(). Defaults to 200ms.
	PollIntervalMs int `json:"poll_interval_ms,omitempty"`
}

func (cfg *Config) Validate(path string) ([]string, []string, error) {
	if cfg.SerialPort == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "serial_port")
	}
	if cfg.StatusSource == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "status_source")
	}
	// Bare name resolves under generic.API; listing it as a dep makes Viam
	// build status_source before us.
	return []string{cfg.StatusSource}, nil, nil
}

// actionForState maps a lifecycle state to the firmware payload. It forwards
// the state word so the firmware owns the visuals; it also allow-lists known
// states (ok=false means "leave the LED as-is"). Return a fuller payload here
// to drive richer behavior from Go instead.
func actionForState(state string) (map[string]interface{}, bool) {
	switch state {
	case "idle", "listening", "thinking", "responding", "error":
		return map[string]interface{}{"state": state}, true
	default:
		return nil, false
	}
}

// isFirmwareReply reports whether a line read from the port is the reference
// firmware talking: it prints "ready" after boot and echoes every line it
// handles. Matching these, rather than any bytes, keeps the ESP32 ROM
// bootloader's output (e.g. "waiting for download") from passing as a reply.
func isFirmwareReply(line string) bool {
	line = strings.TrimSpace(line)
	return line == "ready" ||
		strings.HasPrefix(line, "received line:") ||
		strings.HasPrefix(line, "-> ")
}

type bridge struct {
	resource.AlwaysRebuild

	name   resource.Name
	logger logging.Logger

	source       resource.Resource
	pollInterval time.Duration
	stateKey     string

	// Timings default to the package constants; tests shorten them.
	reconnectBackoff  time.Duration
	resendInterval    time.Duration
	unresponsiveAfter time.Duration
	resetCooldown     time.Duration
	openPort          func(path string, mode *serial.Mode) (serial.Port, error)

	// connMu guards the serial connection (port, lastDial) and serializes
	// writes so bytes don't interleave. port is nil while disconnected.
	connMu     sync.Mutex
	port       serial.Port
	lastDial   time.Time
	serialPort string // connection params: kept for reconnect + Status.
	baudRate   int

	// mu guards the state last written and the firmware liveness tracking.
	mu          sync.Mutex
	lastState   string
	lastWriteAt time.Time
	// firmwareReplied is set once the firmware has answered at all. Only then
	// is silence treated as a hang, so firmware that never echoes is not reset.
	firmwareReplied bool
	lastReplyAt     time.Time
	// awaitingReplySince is when the oldest unanswered write went out; zero
	// when every write has been answered.
	awaitingReplySince time.Time
	lastResetAt        time.Time
	firmwareResets     int64

	// statsMu guards the diagnostic counters. Separate from connMu so a Status
	// query never blocks behind a slow port.Write.
	statsMu      sync.Mutex
	messagesSent int64
	bytesSent    int64
	lastSentAt   time.Time
	lastError    string
	lastErrorAt  time.Time

	workerCtx    context.Context
	workerCancel context.CancelFunc
	workerWG     sync.WaitGroup
	// readerWG tracks the reply reader of each opened port.
	readerWG sync.WaitGroup
}

func newService(ctx context.Context, deps resource.Dependencies, rawConf resource.Config, logger logging.Logger) (resource.Resource, error) {
	conf, err := resource.NativeConfig[*Config](rawConf)
	if err != nil {
		return nil, err
	}
	baud := conf.BaudRate
	if baud <= 0 {
		baud = defaultBaudRate
	}
	pollInterval := defaultPollInterval
	if conf.PollIntervalMs > 0 {
		pollInterval = time.Duration(conf.PollIntervalMs) * time.Millisecond
	}
	stateKey := conf.StateKey
	if stateKey == "" {
		stateKey = defaultStateKey
	}

	source, err := deps.Lookup(generic.Named(conf.StatusSource))
	if err != nil {
		return nil, fmt.Errorf("status_source %q not found: %w", conf.StatusSource, err)
	}

	b := newBridge(rawConf.ResourceName(), logger, source, conf.SerialPort, baud, pollInterval, stateKey)
	// Connect now so a healthy device works immediately and a bad port surfaces
	// in the logs — but don't fail construction if the device is absent; the
	// poll loop reconnects when it appears.
	b.connMu.Lock()
	_, oerr := b.openLocked()
	b.connMu.Unlock()
	if oerr != nil {
		logger.Warnw("led-bridge serial port unavailable; will keep retrying",
			"port", conf.SerialPort, "err", oerr)
	}
	b.start()
	return b, nil
}

func newBridge(
	name resource.Name,
	logger logging.Logger,
	source resource.Resource,
	serialPort string,
	baud int,
	pollInterval time.Duration,
	stateKey string,
) *bridge {
	workerCtx, workerCancel := context.WithCancel(context.Background())
	return &bridge{
		name:              name,
		logger:            logger,
		source:            source,
		pollInterval:      pollInterval,
		stateKey:          stateKey,
		reconnectBackoff:  reconnectBackoff,
		resendInterval:    resendInterval,
		unresponsiveAfter: unresponsiveAfter,
		resetCooldown:     resetCooldown,
		openPort:          serial.Open,
		serialPort:        serialPort,
		baudRate:          baud,
		workerCtx:         workerCtx,
		workerCancel:      workerCancel,
	}
}

func (b *bridge) start() {
	b.workerWG.Add(1)
	go b.run()
}

func (b *bridge) Name() resource.Name { return b.name }

// run polls the source's Status() and keeps the firmware showing its state.
func (b *bridge) run() {
	defer b.workerWG.Done()
	ticker := time.NewTicker(b.pollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-b.workerCtx.Done():
			return
		case <-ticker.C:
			b.tick()
		}
	}
}

// tick writes the source's state when it changes, and again every
// resendInterval while it doesn't, then resets the ESP32 if it has stopped
// answering.
func (b *bridge) tick() {
	defer b.resetIfUnresponsive()

	status, err := b.source.Status(b.workerCtx)
	if err != nil {
		b.logger.Warnw("status_source Status() failed", "err", err)
		return
	}
	state, _ := status[b.stateKey].(string)
	if state == "" {
		return
	}
	payload, ok := actionForState(state)

	b.mu.Lock()
	changed := state != b.lastState
	due := changed || (ok && time.Since(b.lastWriteAt) >= b.resendInterval)
	b.mu.Unlock()
	if !due {
		return
	}

	if ok {
		if err := b.write(payload); err != nil {
			// Don't commit lastState — retry on the next tick so a transient
			// serial hiccup self-heals. write() logs the error.
			return
		}
	} else if changed {
		b.logger.Debugw("no action mapped; leaving LED unchanged", "state", state)
	}

	b.mu.Lock()
	b.lastState = state
	b.lastWriteAt = time.Now()
	b.mu.Unlock()
}

// resetIfUnresponsive hard-resets the ESP32 when writes have gone unanswered
// for unresponsiveAfter. A sketch that hung, or a board left in its ROM
// bootloader by a USB brownout, still accepts bytes through the adapter but
// never acts on them, so the strip stays on its last colour.
func (b *bridge) resetIfUnresponsive() {
	now := time.Now()
	b.mu.Lock()
	hung := b.firmwareReplied &&
		!b.awaitingReplySince.IsZero() &&
		now.Sub(b.awaitingReplySince) >= b.unresponsiveAfter &&
		now.Sub(b.lastResetAt) >= b.resetCooldown
	var silentFor time.Duration
	if hung {
		silentFor = now.Sub(b.awaitingReplySince)
		b.lastResetAt = now
		b.awaitingReplySince = time.Time{}
		b.firmwareResets++
	}
	b.mu.Unlock()
	if !hung {
		return
	}

	b.connMu.Lock()
	defer b.connMu.Unlock()
	if b.port == nil {
		return
	}
	b.logger.Warnw("LED firmware stopped answering; resetting the ESP32",
		"port", b.serialPort, "unanswered_for", silentFor.String())
	if err := pulseReset(b.port); err != nil {
		b.closeLocked()
		b.recordError(fmt.Errorf("reset ESP32: %w", err))
	}
}

// pulseReset reboots an ESP32 dev board into its sketch through the adapter's
// auto-reset circuit: RTS drives EN and DTR drives GPIO0, so holding RTS with
// DTR released resets the chip with GPIO0 high (normal boot). This is
// esptool's hard-reset sequence.
func pulseReset(port serial.Port) error {
	if err := port.SetDTR(false); err != nil {
		return err
	}
	if err := port.SetRTS(true); err != nil {
		return err
	}
	time.Sleep(resetPulse)
	return port.SetRTS(false)
}

// write sends payload as line-delimited JSON over serial, (re)connecting the
// port as needed, and records the outcome in the diagnostic counters surfaced
// by Status().
func (b *bridge) write(payload map[string]interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}
	line := append(data, '\n')

	b.connMu.Lock()
	port, err := b.openLocked()
	if err != nil {
		b.connMu.Unlock()
		if errors.Is(err, errBackoff) {
			return err // device known-down; waiting out the backoff, stay quiet
		}
		b.recordError(err)
		return err
	}
	// Trailing newline is the firmware's line delimiter.
	n, werr := port.Write(line)
	if werr != nil {
		// Stale handle (the device reset/reconnected) — drop it so the next
		// write reconnects to the live device.
		b.closeLocked()
	}
	b.connMu.Unlock()

	if werr != nil {
		b.recordError(werr)
		return fmt.Errorf("write to serial port: %w", werr)
	}
	b.recordSuccess(n)
	b.mu.Lock()
	if b.awaitingReplySince.IsZero() {
		b.awaitingReplySince = time.Now()
	}
	b.mu.Unlock()
	return nil
}

// openLocked returns a healthy serial port, opening it if needed. Caller must
// hold connMu. Reconnect attempts are throttled by reconnectBackoff so an
// absent device isn't dialed every tick.
func (b *bridge) openLocked() (serial.Port, error) {
	if b.port != nil {
		return b.port, nil
	}
	if time.Since(b.lastDial) < b.reconnectBackoff {
		return nil, errBackoff
	}
	b.lastDial = time.Now()
	port, err := b.openPort(b.serialPort, &serial.Mode{BaudRate: b.baudRate})
	if err != nil {
		return nil, err
	}
	b.port = port
	// Writes sent before the drop were never going to be answered; judge the
	// firmware only on what it is sent from here on.
	b.mu.Lock()
	b.awaitingReplySince = time.Time{}
	b.mu.Unlock()
	b.readerWG.Add(1)
	go b.readReplies(port)
	b.logger.Infow("led-bridge serial port opened", "port", b.serialPort, "baud", b.baudRate)
	return port, nil
}

// readReplies reads the firmware's lines from port until the port fails or
// is closed. A failed read is how an unplugged adapter shows up, so the port
// is dropped at once instead of on the next write.
func (b *bridge) readReplies(port serial.Port) {
	defer b.readerWG.Done()
	scanner := bufio.NewScanner(port)
	for scanner.Scan() {
		if isFirmwareReply(scanner.Text()) {
			b.mu.Lock()
			b.firmwareReplied = true
			b.lastReplyAt = time.Now()
			b.awaitingReplySince = time.Time{}
			b.mu.Unlock()
		}
	}

	b.connMu.Lock()
	defer b.connMu.Unlock()
	if b.port != port {
		return // closed deliberately, or already replaced
	}
	b.closeLocked()
	err := scanner.Err()
	if err == nil {
		err = errors.New("serial port closed")
	}
	b.recordError(fmt.Errorf("read from serial port: %w", err))
}

// closeLocked closes and clears the port so the next openLocked reconnects.
// Caller must hold connMu.
func (b *bridge) closeLocked() {
	if b.port != nil {
		_ = b.port.Close()
		b.port = nil
	}
}

func (b *bridge) recordSuccess(n int) {
	b.statsMu.Lock()
	b.messagesSent++
	b.bytesSent += int64(n)
	b.lastSentAt = time.Now()
	b.statsMu.Unlock()
}

// recordError updates the diagnostic counters and warns — but only when the
// error changes, so a persistently failing port doesn't flood the logs.
func (b *bridge) recordError(err error) {
	msg := err.Error()
	b.statsMu.Lock()
	if msg != b.lastError {
		b.logger.Warnw("led serial write failed", "err", msg)
	}
	b.lastError = msg
	b.lastErrorAt = time.Now()
	b.statsMu.Unlock()
}

// Status reports the last state acted on plus connection/serial-write health
// (connected, port, baud, message/byte counts, last send, last error) and
// firmware liveness (last reply, resets) so an operator can confirm the LED
// is wired up and acting on what it is sent.
func (b *bridge) Status(ctx context.Context) (map[string]interface{}, error) {
	b.mu.Lock()
	lastState := b.lastState
	lastReplyAt := b.lastReplyAt
	firmwareResets := b.firmwareResets
	b.mu.Unlock()

	b.connMu.Lock()
	connected := b.port != nil
	b.connMu.Unlock()

	b.statsMu.Lock()
	defer b.statsMu.Unlock()
	status := map[string]interface{}{
		"last_state":      lastState,
		"connected":       connected,
		"serial_port":     b.serialPort,
		"baud_rate":       b.baudRate,
		"messages_sent":   b.messagesSent,
		"bytes_sent":      b.bytesSent,
		"firmware_resets": firmwareResets,
	}
	if !b.lastSentAt.IsZero() {
		status["last_sent_at"] = b.lastSentAt.Format(time.RFC3339Nano)
	}
	if !lastReplyAt.IsZero() {
		status["last_reply_at"] = lastReplyAt.Format(time.RFC3339Nano)
	}
	if b.lastError != "" {
		status["last_error"] = b.lastError
		status["last_error_at"] = b.lastErrorAt.Format(time.RFC3339Nano)
	}
	return status, nil
}

// DoCommand pushes a raw payload to the firmware for bench testing, independent
// of the source. Pass {"payload": {...}}.
func (b *bridge) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	raw, ok := cmd["payload"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unknown command; supported: payload")
	}
	if err := b.write(raw); err != nil {
		return nil, err
	}
	return map[string]interface{}{"sent": raw}, nil
}

func (b *bridge) Close(ctx context.Context) error {
	if b.workerCancel != nil {
		b.workerCancel()
	}
	b.workerWG.Wait()
	b.connMu.Lock()
	var err error
	if b.port != nil {
		err = b.port.Close()
		b.port = nil
	}
	b.connMu.Unlock()
	// The reader takes connMu once its read fails, so wait for it unlocked.
	b.readerWG.Wait()
	return err
}
