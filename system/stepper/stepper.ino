#include <Servo.h>

const int gunPin = 4;
const int dirPin = 3;
const int stepPin = 2;
const int enablePin = 5;
const int speed = 1000; // less is faster
const int SHOOT_COOLDOWN = 200;  // ms

void setup() {
  Serial.begin(9600);

  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, HIGH);  // Disable the driver

  pinMode(gunPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(stepPin, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "SHOOT") {
      digitalWrite(gunPin, HIGH);
      Serial.println("Pew Pew.");
      delay(SHOOT_COOLDOWN);
      digitalWrite(gunPin, LOW);
      Serial.println("Done");
    }
    else if (command == "FLIP") {
      digitalWrite(enablePin, !digitalRead(enablePin));
      Serial.println("Flipped, now the enable pin is: ");
      Serial.print(digitalRead(enablePin));
    }
    else if (command.startsWith("ROTATE:")) {
      digitalWrite(enablePin, LOW);  // Enable the driver
      long steps = command.substring(7).toInt(); // Read number of steps
      Serial.print("Number of steps: ");
      Serial.println(steps);
      if (steps != 0) {
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
      }
      digitalWrite(enablePin, HIGH);  // Disable the driver
      Serial.println("Done");
    }
  }
}
