#include <AccelStepper.h>

// Pins
const int gunPin_I = 10;
const int gunPin_II = 9;
const int dirPin_A = 4;
const int stepPin_A = 7;
const int dirPin_B = 2;
const int stepPin_B = 5;
const int enablePin = 8;
const int SHOOT_COOLDOWN = 2;  // ms
const int MAX_SPEED = 3000;  // steps per second
const int ACCELERATION = 1000;  // steps per second^2

// Stepper setup: DRIVER interface type = 1
AccelStepper stepper_A(AccelStepper::DRIVER, stepPin_A, dirPin_A);
AccelStepper stepper_B(AccelStepper::DRIVER, stepPin_B, dirPin_B);

void setup() {
  Serial.begin(9600);

  pinMode(gunPin_I, OUTPUT);
  pinMode(gunPin_II, OUTPUT);
  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, HIGH);  // Disable driver


  // Stepper setup
  stepper_A.setMaxSpeed(MAX_SPEED);        
  stepper_A.setAcceleration(ACCELERATION);     
  stepper_A.setCurrentPosition(0);    // reset position
  stepper_B.setMaxSpeed(MAX_SPEED);
  stepper_B.setAcceleration(ACCELERATION);
  stepper_B.setCurrentPosition(0);    // reset position
}

void shoot(int gunPin) {
  digitalWrite(gunPin, HIGH);
  Serial.println("Shooting.");
  delay(SHOOT_COOLDOWN);
  digitalWrite(gunPin, LOW);
  Serial.println("Done"); // Critical for communication with Python
}

void rotate(AccelStepper stepper, int enablePin, long steps) {
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
      rotate(stepper_A, enablePin, steps);
    } else if (command.startsWith("ROTATEB:")) {
      long steps = command.substring(8).toInt();
      rotate(stepper_B, enablePin, steps);
    }
  }
}
