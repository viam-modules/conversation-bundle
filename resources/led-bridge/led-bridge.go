// Package ledbridge provides the viam:conversation-bundle:led-bridge model,
// a generic resource that PULLS lifecycle state from another resource and
// drives a USB-serial-attached LED indicator firmware (see
// firmware/led-indicator/ for the ESP32 sketch) to match.
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

	// BaudRate is optional; defaults to 115200, matching the firmware
	// sketches in this repo. Override only if you've modified the firmware.
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
// the LED firmware. This is the entire "opinion" of led-bridge — what each
// state looks like on the strip. The bool is false for states we choose to
// ignore (no write).
//
// TODO(you): design the real payloads your firmware understands. The four
// states voice-command emits are "idle", "listening", "thinking", and
// "responding". The placeholders below just forward the state name so the
// pipeline works end-to-end; replace them with whatever your firmware
// expects (colors, animations, brightness, etc.).
func actionForState(state string) (map[string]interface{}, bool) {
	switch state {
	case "idle":
		return map[string]interface{}{"state": "idle"}, true
	case "listening":
		return map[string]interface{}{"state": "listening"}, true
	case "thinking":
		return map[string]interface{}{"state": "thinking"}, true
	case "responding":
		return map[string]interface{}{"state": "responding"}, true
	default:
		// Unknown state — leave the LED as-is rather than guessing.
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

	// writeMu serializes writes so the poll loop and any manual DoCommand
	// don't interleave bytes on the wire.
	writeMu sync.Mutex

	// mu guards lastState, the most recent state we acted on — used to write
	// only on change and to answer Status().
	mu        sync.Mutex
	lastState string

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
	b.lastState = state
	b.mu.Unlock()
	if !changed {
		return
	}

	payload, ok := actionForState(state)
	if !ok {
		b.logger.Debugw("no action mapped for state; leaving LED unchanged", "state", state)
		return
	}
	if err := b.write(payload); err != nil {
		b.logger.Warnw("led write failed", "state", state, "err", err)
	}
}

// write marshals a payload to line-delimited JSON and sends it over serial.
func (b *bridge) write(payload map[string]interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}
	b.writeMu.Lock()
	defer b.writeMu.Unlock()
	// Trailing newline is the firmware's line delimiter — without it the
	// device buffers indefinitely waiting for end-of-line.
	if _, err := b.port.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("write to serial port: %w", err)
	}
	return nil
}

func (b *bridge) Status(ctx context.Context) (map[string]interface{}, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return map[string]interface{}{"last_state": b.lastState}, nil
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
