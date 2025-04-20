const int dirPin = 2;
const int stepPin = 3;
const int speed = 1000;
void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    long steps = Serial.parseInt(); // Read number of steps
    if (steps != 0) {
      digitalWrite(dirPin, steps > 0 ? HIGH : LOW);
      steps = abs(steps);
      for (long i = 0; i < steps; i++) {
        digitalWrite(stepPin, HIGH);
        delayMicroseconds(speed);  // Adjust speed here
        digitalWrite(stepPin, LOW);
        delayMicroseconds(speed);
      }
      Serial.println("Done");
    }
  }
}
