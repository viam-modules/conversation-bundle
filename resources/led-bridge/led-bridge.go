// Package ledbridge provides the viam:conversation-bundle:led-bridge model: a
// generic resource that polls another resource's Status() "state" field and
// drives a USB-serial LED indicator firmware to match. The source (named by
// status_source, e.g. voice-command) knows nothing about LEDs; the firmware (a
// separate ESP32 sketch) owns the visuals.
package ledbridge

import (
	"context"
	"encoding/json"
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
)

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

	port         serial.Port
	source       resource.Resource
	pollInterval time.Duration
	stateKey     string

	// serialPort and baudRate are kept for Status reporting.
	serialPort string
	baudRate   int

	// writeMu serializes serial writes so bytes don't interleave.
	writeMu sync.Mutex

	// mu guards lastState (the state we last acted on; used to write on change).
	mu        sync.Mutex
	lastState string

	// statsMu guards the diagnostic counters. Separate from writeMu so a Status
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

	port, err := serial.Open(conf.SerialPort, &serial.Mode{BaudRate: baud})
	if err != nil {
		return nil, fmt.Errorf("open serial port %q at %d baud: %w", conf.SerialPort, baud, err)
	}
	logger.Infow("led-bridge serial port opened", "port", conf.SerialPort, "baud", baud, "status_source", conf.StatusSource)

	workerCtx, workerCancel := context.WithCancel(context.Background())
	b := &bridge{
		name:         rawConf.ResourceName(),
		logger:       logger,
		port:         port,
		source:       source,
		pollInterval: pollInterval,
		stateKey:     stateKey,
		serialPort:   conf.SerialPort,
		baudRate:     baud,
		workerCtx:    workerCtx,
		workerCancel: workerCancel,
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

// write sends payload as line-delimited JSON over serial and records the
// outcome in the diagnostic counters surfaced by Status().
func (b *bridge) write(payload map[string]interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}
	b.writeMu.Lock()
	// Trailing newline is the firmware's line delimiter.
	n, werr := b.port.Write(append(data, '\n'))
	b.writeMu.Unlock()

	b.statsMu.Lock()
	if werr != nil {
		msg := werr.Error()
		// Warn only when the error changes, so a persistently broken port
		// (retried every tick) doesn't flood the logs.
		if msg != b.lastError {
			b.logger.Warnw("led serial write failed", "err", msg)
		}
		b.lastError = msg
		b.lastErrorAt = time.Now()
	} else {
		b.messagesSent++
		b.bytesSent += int64(n)
		b.lastSentAt = time.Now()
	}
	b.statsMu.Unlock()

	if werr != nil {
		return fmt.Errorf("write to serial port: %w", werr)
	}
	return nil
}

// Status reports the last state acted on plus serial-write health (port, baud,
// message/byte counts, last send, last error) so an operator can confirm the
// LED is wired up and receiving data.
func (b *bridge) Status(ctx context.Context) (map[string]interface{}, error) {
	b.mu.Lock()
	lastState := b.lastState
	b.mu.Unlock()

	b.statsMu.Lock()
	defer b.statsMu.Unlock()
	status := map[string]interface{}{
		"last_state":    lastState,
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
	if b.port != nil {
		return b.port.Close()
	}
	return nil
}
