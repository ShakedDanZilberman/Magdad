#define STEP_PIN 7
#define DIR_PIN 6


void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  delay(500);

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, HIGH);
}

void loop() {
  digitalWrite(STEP_PIN, HIGH);
  delay(10);
  digitalWrite(LED_BUILTIN, HIGH);
  digitalWrite(STEP_PIN, LOW);
  delay(10);
  digitalWrite(LED_BUILTIN, LOW);
}