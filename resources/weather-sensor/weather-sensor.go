// Package weathersensor provides the viam:conversation-bundle:weather-sensor
// model — a lightweight sensor that reports current weather conditions for
// a configured latitude/longitude by calling the Open-Meteo API (free, no
// key required). Designed to be attached to voice-command as a context
// sensor so Claude can answer weather-aware questions.
package weathersensor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"go.viam.com/rdk/components/sensor"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
)

var Model = resource.NewModel("viam", "conversation-bundle", "weather-sensor")

func init() {
	resource.RegisterComponent(sensor.API, Model,
		resource.Registration[sensor.Sensor, *Config]{
			Constructor: newSensor,
		},
	)
}

// Default location when neither latitude nor longitude is set in config.
// Null Island (0,0) is treated as "unset" — nobody realistically wants
// weather for that point, so we fall back to NYC.
const (
	defaultLatitude  = 40.7128  // New York City
	defaultLongitude = -74.0060
)

type Config struct {
	Latitude    float64 `json:"latitude,omitempty"`
	Longitude   float64 `json:"longitude,omitempty"`
	Unit        string  `json:"unit,omitempty"`           // "imperial" or "metric" — default "imperial"
	CacheTTLSec float64 `json:"cache_ttl_sec,omitempty"`  // default 60s
}

func (c *Config) Validate(path string) ([]string, []string, error) {
	if c.Latitude < -90 || c.Latitude > 90 {
		return nil, nil, fmt.Errorf("%s: latitude must be in [-90, 90]", path)
	}
	if c.Longitude < -180 || c.Longitude > 180 {
		return nil, nil, fmt.Errorf("%s: longitude must be in [-180, 180]", path)
	}
	if c.Unit != "" && c.Unit != "imperial" && c.Unit != "metric" {
		return nil, nil, fmt.Errorf("%s: unit must be \"imperial\" or \"metric\"", path)
	}
	return nil, nil, nil
}

type weatherSensor struct {
	resource.AlwaysRebuild

	name   resource.Name
	logger logging.Logger

	latitude  float64
	longitude float64
	unit      string
	cacheTTL  time.Duration

	httpClient *http.Client

	mu        sync.Mutex
	cached    map[string]interface{}
	cachedAt  time.Time
	cachedErr error
}

func newSensor(ctx context.Context, deps resource.Dependencies, rawConf resource.Config, logger logging.Logger) (sensor.Sensor, error) {
	conf, err := resource.NativeConfig[*Config](rawConf)
	if err != nil {
		return nil, err
	}
	unit := conf.Unit
	if unit == "" {
		unit = "imperial"
	}
	ttl := time.Duration(conf.CacheTTLSec * float64(time.Second))
	if ttl <= 0 {
		ttl = 60 * time.Second
	}
	lat, lon := conf.Latitude, conf.Longitude
	if lat == 0 && lon == 0 {
		lat, lon = defaultLatitude, defaultLongitude
		logger.Infow("weather-sensor: no lat/long configured; defaulting to NYC", "latitude", lat, "longitude", lon)
	}
	return &weatherSensor{
		name:       rawConf.ResourceName(),
		logger:     logger,
		latitude:   lat,
		longitude:  lon,
		unit:       unit,
		cacheTTL:   ttl,
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}, nil
}

func (s *weatherSensor) Name() resource.Name { return s.name }

// Readings returns the most recent weather snapshot, refetching from
// Open-Meteo if the cached value is older than cacheTTL. Upstream errors
// are cached too (for the same TTL) so a flaky API doesn't hammer the
// network on every voice turn.
func (s *weatherSensor) Readings(ctx context.Context, extra map[string]interface{}) (map[string]interface{}, error) {
	s.mu.Lock()
	if !s.cachedAt.IsZero() && time.Since(s.cachedAt) < s.cacheTTL {
		defer s.mu.Unlock()
		if s.cachedErr != nil {
			return nil, s.cachedErr
		}
		return copyReadings(s.cached), nil
	}
	s.mu.Unlock()

	fresh, err := s.fetch(ctx)

	s.mu.Lock()
	s.cachedAt = time.Now()
	s.cached = fresh
	s.cachedErr = err
	s.mu.Unlock()

	if err != nil {
		return nil, err
	}
	return copyReadings(fresh), nil
}

func (s *weatherSensor) fetch(ctx context.Context) (map[string]interface{}, error) {
	q := url.Values{}
	q.Set("latitude", strconv.FormatFloat(s.latitude, 'f', -1, 64))
	q.Set("longitude", strconv.FormatFloat(s.longitude, 'f', -1, 64))
	q.Set("current", "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m")
	if s.unit == "imperial" {
		q.Set("temperature_unit", "fahrenheit")
		q.Set("wind_speed_unit", "mph")
	}
	endpoint := "https://api.open-meteo.com/v1/forecast?" + q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("open-meteo fetch: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("open-meteo returned %d: %s", resp.StatusCode, body)
	}

	var payload struct {
		Current struct {
			Time          string  `json:"time"`
			Temperature2m float64 `json:"temperature_2m"`
			Humidity2m    int     `json:"relative_humidity_2m"`
			WeatherCode   int     `json:"weather_code"`
			WindSpeed10m  float64 `json:"wind_speed_10m"`
		} `json:"current"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("parse open-meteo response: %w", err)
	}

	return map[string]interface{}{
		"temperature":         payload.Current.Temperature2m,
		"humidity_pct":        payload.Current.Humidity2m,
		"wind_speed":          payload.Current.WindSpeed10m,
		"weather_description": wmoCodeDescription(payload.Current.WeatherCode),
		"as_of":               payload.Current.Time,
		"unit":                s.unit,
	}, nil
}

func (s *weatherSensor) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return nil, fmt.Errorf("weather-sensor does not support DoCommand")
}

// Status mirrors Readings for the resource.Resource contract, so dashboards
// and status polls see the same view as voice-command does.
func (s *weatherSensor) Status(ctx context.Context) (map[string]interface{}, error) {
	return s.Readings(ctx, nil)
}

func (s *weatherSensor) Close(ctx context.Context) error { return nil }

// copyReadings returns a shallow copy of the map so callers don't mutate
// the cached snapshot.
func copyReadings(in map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// wmoCodeDescription translates Open-Meteo's WMO weather-condition code
// to a short English description. Codes from
// https://open-meteo.com/en/docs#weathervariables.
func wmoCodeDescription(code int) string {
	switch code {
	case 0:
		return "Clear sky"
	case 1:
		return "Mainly clear"
	case 2:
		return "Partly cloudy"
	case 3:
		return "Overcast"
	case 45:
		return "Foggy"
	case 48:
		return "Depositing rime fog"
	case 51:
		return "Light drizzle"
	case 53:
		return "Moderate drizzle"
	case 55:
		return "Dense drizzle"
	case 56:
		return "Light freezing drizzle"
	case 57:
		return "Dense freezing drizzle"
	case 61:
		return "Slight rain"
	case 63:
		return "Moderate rain"
	case 65:
		return "Heavy rain"
	case 66:
		return "Light freezing rain"
	case 67:
		return "Heavy freezing rain"
	case 71:
		return "Slight snowfall"
	case 73:
		return "Moderate snowfall"
	case 75:
		return "Heavy snowfall"
	case 77:
		return "Snow grains"
	case 80:
		return "Slight rain showers"
	case 81:
		return "Moderate rain showers"
	case 82:
		return "Violent rain showers"
	case 85:
		return "Slight snow showers"
	case 86:
		return "Heavy snow showers"
	case 95:
		return "Thunderstorm"
	case 96:
		return "Thunderstorm with slight hail"
	case 99:
		return "Thunderstorm with heavy hail"
	default:
		return fmt.Sprintf("WMO code %d", code)
	}
}
