#include <AccelStepper.h>

// Pins
const int gunPin = 4;
const int dirPin = 2;
const int stepPin = 3;
const int enablePin = 5;
const int SHOOT_COOLDOWN = 2;  // ms

// Stepper setup: DRIVER interface type = 1
AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);

void setup() {
  Serial.begin(9600);

  pinMode(gunPin, OUTPUT);
  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, HIGH);  // Disable driver


  // Stepper setup
  stepper.setMaxSpeed(2000);        // steps per second
  stepper.setAcceleration(1000);     // steps per second^2
  stepper.setCurrentPosition(0);    // reset position
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "SHOOT") {
      digitalWrite(gunPin, HIGH);
      Serial.println("Shooting.");
      delay(SHOOT_COOLDOWN);
      digitalWrite(gunPin, LOW);
      Serial.println("Done");
    }
    else if (command == "FLIP") {
      digitalWrite(enablePin, !digitalRead(enablePin));
      Serial.print("Flipped, now the enable pin is: ");
      Serial.println(digitalRead(enablePin));
    }
    else if (command.startsWith("ROTATE:")) {
      long steps = command.substring(7).toInt();
      Serial.print("Requested steps: ");
      Serial.println(steps);

      digitalWrite(enablePin, LOW);  // Enable driver
      long target = stepper.currentPosition() + steps;
      stepper.moveTo(target);

      // Run motor until it reaches the target
      while (stepper.distanceToGo() != 0) {
        stepper.run();
      }

      digitalWrite(enablePin, HIGH);  // Disable driver after move
      Serial.println("Done"); // Critical for communication with Python
    }
  }
}
