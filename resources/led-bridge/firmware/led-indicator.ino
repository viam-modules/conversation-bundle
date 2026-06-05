// led-indicator.ino — reference firmware for the led-bridge LED indicator.
//
// This sketch runs on an ESP32 driving an 80-pixel Adafruit NeoPixel strip.
// It is NOT built, flashed, or otherwise used by this repo at runtime — it
// lives here purely for reference alongside ../led-bridge.go, the host-side
// Viam module that drives it over USB serial.
//
// Hardware / wiring:
//   - NeoPixel data pin: GPIO 18
//   - Strip length:      80 pixels
//   - Serial baud:       115200  (must match led-bridge.go's defaultBaudRate)
//
// Serial protocol: led-bridge.go writes one '\n'-terminated line per update,
// each a JSON object of the form {"state":"<mode>"} (see actionForState in
// led-bridge.go). This sketch does a substring match on that line, so the
// JSON wrapper is matched transparently — it only cares that one of these
// keywords appears:
//   - "idle"        -> LEDs off
//   - "listening"   -> solid blue
//   - "thinking"    -> orange sine pulse (~800ms period)
//   - "responding"  -> solid orange
// Any line without a recognized keyword is logged and ignored.

#include <Adafruit_NeoPixel.h>

const int  LED_PIN    = 18;
const int  NUM_PIXELS = 80;
const long BAUD_RATE  = 115200;
const uint8_t BRIGHTNESS = 80;

Adafruit_NeoPixel strip(NUM_PIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

enum Mode { MODE_IDLE, MODE_LISTENING, MODE_THINKING, MODE_RESPONDING };
Mode currentMode = MODE_IDLE;
String inputBuffer;

void setup() {
  Serial.begin(BAUD_RATE);
  strip.begin();
  strip.setBrightness(BRIGHTNESS);

  // Boot self-test: R G B on first 3 LEDs for 2 seconds.
  strip.setPixelColor(0, strip.Color(255, 0, 0));
  strip.setPixelColor(1, strip.Color(0, 255, 0));
  strip.setPixelColor(2, strip.Color(0, 0, 255));
  strip.show();
  delay(2000);
  strip.clear();
  strip.show();

  Serial.println("ready");
}

void loop() {
  readSerial();
  render();
  delay(20);
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleLine(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}

void handleLine(const String &line) {
  Serial.print("received line: ");
  Serial.println(line);
  if (line.indexOf("listening") >= 0) {
    currentMode = MODE_LISTENING;
    Serial.println("-> mode: listening");
  } else if (line.indexOf("thinking") >= 0) {
    currentMode = MODE_THINKING;
    Serial.println("-> mode: thinking");
  } else if (line.indexOf("idle") >= 0) {
    currentMode = MODE_IDLE;
    Serial.println("-> mode: idle");
  } else if (line.indexOf("responding") >= 0) {
    currentMode = MODE_RESPONDING;
    Serial.println("-> mode: responding");
  } else {
    Serial.println("-> unrecognized, ignoring");
  }
}

// Render a single-color sine pulse across the whole strip.
// period_ms controls speed (smaller = faster). r, g, b set the hue.
void pulseColor(uint16_t period_ms, uint8_t r, uint8_t g, uint8_t b) {
  float phase = (millis() % period_ms) / (float)period_ms;
  float wave = (sin(phase * 2.0f * PI) + 1.0f) * 0.5f;
  float scale_lo = 10.0f / 255.0f;
  float scale = scale_lo + wave * (1.0f - scale_lo);
  uint32_t color = strip.Color(
    (uint8_t)(r * scale),
    (uint8_t)(g * scale),
    (uint8_t)(b * scale));
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

void solidColor(uint8_t r, uint8_t g, uint8_t b) {
  uint32_t color = strip.Color(r, g, b);
  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

void render() {
  switch (currentMode) {
    case MODE_IDLE:
      strip.clear();
      strip.show();
      break;
    case MODE_LISTENING:
      solidColor(0, 0, 255);
      break;
    case MODE_THINKING:
      pulseColor(800, 255, 140, 0);
      break;
    case MODE_RESPONDING:
      solidColor(255, 140, 0);
      break;
  }
}
