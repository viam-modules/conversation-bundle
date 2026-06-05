// Package ledbridge provides the viam:conversation-bundle:led-bridge model: a
// generic resource that polls another resource's Status() "state" field and
// drives a USB-serial LED indicator firmware to match. The source (named by
// status_source, e.g. voice-command) knows nothing about LEDs; the firmware (a
// separate ESP32 sketch) owns the visuals.
package ledbridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	case "idle", "listening", "thinking", "responding":
		return map[string]interface{}{"state": state}, true
	default:
		return nil, false
	}
}

type bridge struct {
	resource.AlwaysRebuild

	name   resource.Name
	logger logging.Logger

	source       resource.Resource
	pollInterval time.Duration
	stateKey     string

	// connMu guards the serial connection (port, lastDial) and serializes
	// writes so bytes don't interleave. port is nil while disconnected.
	connMu     sync.Mutex
	port       serial.Port
	lastDial   time.Time
	serialPort string // connection params: kept for reconnect + Status.
	baudRate   int

	// mu guards lastState (the state we last acted on; used to write on change).
	mu        sync.Mutex
	lastState string

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

	workerCtx, workerCancel := context.WithCancel(context.Background())
	b := &bridge{
		name:         rawConf.ResourceName(),
		logger:       logger,
		source:       source,
		pollInterval: pollInterval,
		stateKey:     stateKey,
		serialPort:   conf.SerialPort,
		baudRate:     baud,
		workerCtx:    workerCtx,
		workerCancel: workerCancel,
	}
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
	b.workerWG.Add(1)
	go b.run()
	return b, nil
}

func (b *bridge) Name() resource.Name { return b.name }

// run polls the source's Status() and writes to the firmware whenever the
// reported state changes.
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

func (b *bridge) tick() {
	status, err := b.source.Status(b.workerCtx)
	if err != nil {
		b.logger.Warnw("status_source Status() failed", "err", err)
		return
	}
	state, _ := status[b.stateKey].(string)
	if state == "" {
		return
	}

	b.mu.Lock()
	changed := state != b.lastState
	b.mu.Unlock()
	if !changed {
		return
	}

	payload, ok := actionForState(state)
	if ok {
		if err := b.write(payload); err != nil {
			// Don't commit lastState — retry on the next tick so a transient
			// serial hiccup self-heals. write() logs the error.
			return
		}
	} else {
		b.logger.Debugw("no action mapped; leaving LED unchanged", "state", state)
	}

	// Commit only after handling, so we don't re-process the same state.
	b.mu.Lock()
	b.lastState = state
	b.mu.Unlock()
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
	return nil
}

// openLocked returns a healthy serial port, opening it if needed. Caller must
// hold connMu. Reconnect attempts are throttled by reconnectBackoff so an
// absent device isn't dialed every tick.
func (b *bridge) openLocked() (serial.Port, error) {
	if b.port != nil {
		return b.port, nil
	}
	if time.Since(b.lastDial) < reconnectBackoff {
		return nil, errBackoff
	}
	b.lastDial = time.Now()
	port, err := serial.Open(b.serialPort, &serial.Mode{BaudRate: b.baudRate})
	if err != nil {
		return nil, err
	}
	b.port = port
	b.logger.Infow("led-bridge serial port opened", "port", b.serialPort, "baud", b.baudRate)
	return port, nil
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
// (connected, port, baud, message/byte counts, last send, last error) so an
// operator can confirm the LED is wired up and receiving data.
func (b *bridge) Status(ctx context.Context) (map[string]interface{}, error) {
	b.mu.Lock()
	lastState := b.lastState
	b.mu.Unlock()

	b.connMu.Lock()
	connected := b.port != nil
	b.connMu.Unlock()

	b.statsMu.Lock()
	defer b.statsMu.Unlock()
	status := map[string]interface{}{
		"last_state":    lastState,
		"connected":     connected,
		"serial_port":   b.serialPort,
		"baud_rate":     b.baudRate,
		"messages_sent": b.messagesSent,
		"bytes_sent":    b.bytesSent,
	}
	if !b.lastSentAt.IsZero() {
		status["last_sent_at"] = b.lastSentAt.Format(time.RFC3339Nano)
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
	defer b.connMu.Unlock()
	if b.port != nil {
		return b.port.Close()
	}
	return nil
}
