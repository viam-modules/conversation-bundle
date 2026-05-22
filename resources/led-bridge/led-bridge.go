// Package ledbridge provides the viam:conversation-bundle:led-bridge model,
// a thin generic resource that forwards DoCommand payloads as line-delimited
// JSON to a USB-serial-attached LED indicator firmware (see
// firmware/led-indicator/ for the ESP32 sketch). The firmware decides what
// each payload means visually; this component just owns the serial pipe so
// the rest of the module (notably voice-command) can stay hardware-agnostic.
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

const defaultBaudRate = 115200

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

	// BaudRate is optional; defaults to 115200, which matches the firmware
	// sketches in this repo. Override only if you've modified the firmware.
	BaudRate int `json:"baud_rate,omitempty"`
}

func (cfg *Config) Validate(path string) ([]string, []string, error) {
	if cfg.SerialPort == "" {
		return nil, nil, resource.NewConfigValidationFieldRequiredError(path, "serial_port")
	}
	return nil, nil, nil
}

type bridge struct {
	resource.AlwaysRebuild

	name   resource.Name
	logger logging.Logger

	// serialPort and baudRate are retained for Status reporting; the
	// underlying os.File path may not be recoverable from serial.Port.
	serialPort string
	baudRate   int

	port serial.Port

	// writeMu serializes writes so concurrent DoCommand calls don't
	// interleave bytes on the wire.
	writeMu sync.Mutex

	// statsMu protects the diagnostic counters below. Kept separate from
	// writeMu so a Status query doesn't block behind a slow port.Write —
	// operators need diagnostics most when the wire is misbehaving.
	statsMu      sync.Mutex
	messagesSent int64
	bytesSent    int64
	lastSentAt   time.Time
	lastError    string
	lastErrorAt  time.Time
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
	port, err := serial.Open(conf.SerialPort, &serial.Mode{BaudRate: baud})
	if err != nil {
		return nil, fmt.Errorf("open serial port %q at %d baud: %w", conf.SerialPort, baud, err)
	}
	logger.Infow("led-bridge serial port opened", "port", conf.SerialPort, "baud", baud)
	return &bridge{
		name:       rawConf.ResourceName(),
		logger:     logger,
		serialPort: conf.SerialPort,
		baudRate:   baud,
		port:       port,
	}, nil
}

func (b *bridge) Name() resource.Name { return b.name }

// Status reports the configured serial port, baud rate, and cumulative
// write diagnostics. Useful when triaging "is the LED wired up correctly?"
// and "have any payloads actually made it out?". Exposed via DoCommand's
// reserved "status" key — Status is not on resource.Resource, so DoCommand
// is the only way to surface it to clients.
func (b *bridge) Status(ctx context.Context) (map[string]interface{}, error) {
	b.statsMu.Lock()
	defer b.statsMu.Unlock()
	status := map[string]interface{}{
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

func (b *bridge) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	// Log the current status after every DoCommand so a debug-level tail
	// shows the counters and last_error advancing in real time. Deferred
	// so the snapshot reflects any stats updated by the call itself.
	defer func() {
		s, _ := b.Status(ctx)
		b.logger.Debugw("led-bridge do_command", "status", s)
	}()

	// "status" is reserved as a diagnostic query — Status is not on
	// resource.Resource so DoCommand is the only entry point. The LED
	// protocol uses {"state": "..."} payloads, so this doesn't collide
	// with any forwarded firmware command.
	if _, ok := cmd["status"]; ok {
		return b.Status(ctx)
	}
	payload, err := json.Marshal(cmd)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}
	line := append(payload, '\n')
	b.writeMu.Lock()
	// Trailing newline is the firmware's line delimiter — without it the
	// device buffers indefinitely waiting for end-of-line.
	n, werr := b.port.Write(line)
	b.writeMu.Unlock()

	b.statsMu.Lock()
	if werr != nil {
		b.lastError = werr.Error()
		b.lastErrorAt = time.Now()
	} else {
		b.messagesSent++
		b.bytesSent += int64(n)
		b.lastSentAt = time.Now()
	}
	b.statsMu.Unlock()

	if werr != nil {
		return nil, fmt.Errorf("write to serial port: %w", werr)
	}
	return map[string]interface{}{"sent": string(payload)}, nil
}

func (b *bridge) Close(ctx context.Context) error {
	if b.port != nil {
		return b.port.Close()
	}
	return nil
}
