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

	port serial.Port

	// writeMu serializes writes so concurrent DoCommand calls don't
	// interleave bytes on the wire.
	writeMu sync.Mutex
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
		name:   rawConf.ResourceName(),
		logger: logger,
		port:   port,
	}, nil
}

func (b *bridge) Name() resource.Name { return b.name }

func (b *bridge) Status(ctx context.Context) (map[string]interface{}, error) {
	return map[string]interface{}{}, nil
}

func (b *bridge) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	payload, err := json.Marshal(cmd)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}
	b.writeMu.Lock()
	defer b.writeMu.Unlock()
	// Trailing newline is the firmware's line delimiter — without it the
	// device buffers indefinitely waiting for end-of-line.
	if _, err := b.port.Write(append(payload, '\n')); err != nil {
		return nil, fmt.Errorf("write to serial port: %w", err)
	}
	return map[string]interface{}{"sent": string(payload)}, nil
}

func (b *bridge) Close(ctx context.Context) error {
	if b.port != nil {
		return b.port.Close()
	}
	return nil
}
