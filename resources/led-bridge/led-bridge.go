// Package ledbridge provides the viam:conversation-bundle:led-bridge model,
// a generic resource that PULLS lifecycle state from another resource and
// drives a USB-serial-attached LED indicator firmware to match. The firmware
// is a separate ESP32 sketch flashed to the indicator hardware, not part of
// this repo.
//
// The component it watches is named by `status_source` and is depended on,
// so it is built first. led-bridge polls that source's Status() on an
// interval, reads the "state" field, and writes the corresponding payload to
// the firmware over serial whenever the state changes. The source (e.g.
// voice-command) knows nothing about LEDs; all of the "what does this state
// look like" opinion lives here, in actionForState.
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
)

func init() {
	resource.RegisterService(generic.API, Model,
		resource.Registration[resource.Resource, *Config]{
			Constructor: newService,
		},
	)
}

type Config struct {
	// SerialPort is the path to the USB-serial device the indicator firmware
	// is reachable at. On Linux this typically looks like /dev/ttyUSB0 or
	// /dev/ttyACM0; on macOS it's /dev/cu.usbserial-XXXX.
	SerialPort string `json:"serial_port"`

	// StatusSource is the name of the resource to watch. led-bridge polls
	// its Status() and reacts to the "state" field. Typically the
	// voice-command service, but any generic resource whose Status() reports
	// a "state" string works — point it at a different one for a different
	// flow.
	StatusSource string `json:"status_source"`

	// BaudRate is optional; defaults to 115200, matching the indicator
	// firmware. Override only if you've modified the firmware.
	BaudRate int `json:"baud_rate,omitempty"`

	// PollIntervalMs is optional; defaults to 200ms. How often to read the
	// source's Status() looking for a state change.
	PollIntervalMs int `json:"poll_interval_ms,omitempty"`
}

func (cfg *Config) Validate(path string) ([]string, []string, error) {
	if cfg.SerialPort == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "serial_port")
	}
	if cfg.StatusSource == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "status_source")
	}
	// status_source is resolved under generic.API; a bare name defaults to
	// that API in the dep resolver, and listing it here makes Viam build it
	// before us and inject the handle into deps.
	return []string{cfg.StatusSource}, nil, nil
}

// actionForState maps a source lifecycle state to the JSON payload written to
// the LED firmware, and acts as an allow-list: the bool is false for states
// we don't recognize, which the caller treats as "leave the LED as-is".
//
// By design the payload just forwards the state word ({"state": "<state>"});
// the firmware owns the visuals (color, animation, brightness) and matches on
// that word. This keeps led-bridge decoupled from the firmware — the strip can
// be redesigned without touching this code. To make a state drive richer
// behavior from the Go side instead, return a fuller payload here.
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

	port   serial.Port
	source resource.Resource
	poll   time.Duration

	// serialPort and baudRate are retained for Status reporting; the
	// underlying serial.Port doesn't expose them back.
	serialPort string
	baudRate   int

	// writeMu serializes writes so the poll loop and any manual DoCommand
	// don't interleave bytes on the wire.
	writeMu sync.Mutex

	// mu guards lastState, the most recent state we acted on — used to write
	// only on change and to answer Status().
	mu        sync.Mutex
	lastState string

	// statsMu protects the diagnostic counters below. Kept separate from
	// writeMu so a Status query never blocks behind a slow port.Write —
	// operators need diagnostics most when the wire is misbehaving.
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
	poll := defaultPollInterval
	if conf.PollIntervalMs > 0 {
		poll = time.Duration(conf.PollIntervalMs) * time.Millisecond
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
		poll:         poll,
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
// reported "state" changes.
func (b *bridge) run() {
	defer b.workerWG.Done()
	ticker := time.NewTicker(b.poll)
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
	state, _ := status["state"].(string)
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
			// Leave lastState unchanged so the next tick retries the write —
			// a transient serial hiccup self-heals once the port recovers.
			// write() records and logs the error.
			return
		}
	} else {
		b.logger.Debugw("no action mapped for state; leaving LED unchanged", "state", state)
	}

	// Commit the state only once we've handled it (written, or deliberately
	// ignored) so we don't re-process the same value every tick.
	b.mu.Lock()
	b.lastState = state
	b.mu.Unlock()
}

// write marshals a payload to line-delimited JSON, sends it over serial, and
// records the outcome in the diagnostic counters surfaced by Status().
func (b *bridge) write(payload map[string]interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}
	b.writeMu.Lock()
	// Trailing newline is the firmware's line delimiter — without it the
	// device buffers indefinitely waiting for end-of-line.
	n, werr := b.port.Write(append(data, '\n'))
	b.writeMu.Unlock()

	b.statsMu.Lock()
	if werr != nil {
		msg := werr.Error()
		// Dedupe the log: only warn when the error changes, so a persistently
		// broken port (retried every tick) doesn't flood the logs. Status()
		// still surfaces the sticky last_error/last_error_at for diagnosis.
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

// Status reports the last state we acted on plus serial-write health — which
// port/baud we're on, how many messages/bytes have gone out, when the last
// one was, and the last error if any. Lets an operator confirm from the
// dashboard whether the LED is actually wired up and receiving data.
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

// DoCommand allows manually pushing a raw payload to the firmware, mostly for
// bench testing the strip independent of the source. Pass {"payload": {...}}.
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
