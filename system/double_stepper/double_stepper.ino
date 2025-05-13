// Pins
const int gunPin_I = 10;
const int gunPin_II = 9;
const int dirPin_A = 4;
const int stepPin_A = 7;
const int dirPin_B = 2;
const int stepPin_B = 5;
const int enablePin = 8;
const int SHOOT_COOLDOWN = 2;  // ms
const int speed = 1000;

void setup() {
  Serial.begin(9600);

  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, HIGH);  // Disable the driver

  pinMode(gunPin_I, OUTPUT);
  pinMode(gunPin_II, OUTPUT);
  pinMode(dirPin_A, OUTPUT);
  pinMode(dirPin_B, OUTPUT);
  pinMode(stepPin_A, OUTPUT);
  pinMode(stepPin_B, OUTPUT);
}

void shoot(int gunPin) {
  digitalWrite(gunPin, HIGH);
  Serial.println("Shooting.");
  delay(SHOOT_COOLDOWN);
  digitalWrite(gunPin, LOW);
  Serial.println("Done");  // Critical for communication with Python
}

void rotate(int dirPin, int stepPin, int enablePin, long steps) {
  Serial.print("Number of steps: ");
  Serial.println(steps);
  if (steps == 0) {
    Serial.println("Done");
    return;
  }
  digitalWrite(enablePin, LOW);  // Enable the driver
  int direction = steps > 0 ? HIGH : LOW;
  Serial.print("Rotating in direction (0=+, 1=-): ");
  Serial.println(direction);

  digitalWrite(dirPin, direction);
  steps = abs(steps);

  for (long i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(speed);  // Adjust speed here
    digitalWrite(stepPin, LOW);
    delayMicroseconds(speed);
  }

  digitalWrite(enablePin, HIGH);  // Disable the driver
  Serial.println("Done");
}


void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "SHOOT1") {
      shoot(gunPin_I);
    } else if (command == "SHOOT2") {
      shoot(gunPin_II);
    } else if (command.startsWith("ROTATEA:")) {
      long steps = command.substring(8).toInt();
      rotate(dirPin_A, stepPin_A, enablePin, steps);
    } else if (command.startsWith("ROTATEB:")) {
      long steps = command.substring(8).toInt();
      rotate(dirPin_B, stepPin_B, enablePin, steps);
    }
  }
}
