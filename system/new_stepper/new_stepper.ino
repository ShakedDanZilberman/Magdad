#include <AccelStepper.h>

/*
CNC pins Z - 
const int gunPin = 12;
const int dirPin = 4;
const int stepPin = 7;
const int enablePin = 8;



bodoboard - 
const int gunPin = 4;
const int dirPin = 3;
const int stepPin = 2;
const int enablePin = 5;
*/

const int gunPin = 12;
const int dirPin = 4;
const int stepPin = 7;
const int enablePin = 8;
const int SHOOT_COOLDOWN = 2;  // ms
const int MAX_SPEED = 3000;  // steps per second
const int ACCELERATION = 10000;  // steps per second^2
long count = 0;

// Stepper setup: DRIVER interface type = 1
AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);


void setup() {
  Serial.begin(9600);
  pinMode(gunPin, OUTPUT);
  pinMode(enablePin, OUTPUT);

  stepper.setEnablePin(enablePin);
  // stepper.enableOutputs();
  stepper.setPinsInverted(false, false, true);
  stepper.disableOutputs();

  // Stepper setup
  stepper.setMaxSpeed(MAX_SPEED);        
  stepper.setAcceleration(ACCELERATION);     
  stepper.setCurrentPosition(0);    // reset position
}


void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "SHOOT") {
      digitalWrite(gunPin, HIGH);
      Serial.println("Pew Pew");
      delay(SHOOT_COOLDOWN);
      digitalWrite(gunPin, LOW);
      Serial.println("Done"); // Critical for communication with Python
    } 
    else if (command.startsWith("ROTATE:")) {
      stepper.enableOutputs();
      long steps = command.substring(7).toInt();
      Serial.print("Requested steps: ");
      Serial.println(steps);
      stepper.move(steps);
      Serial.println("Done"); // Critical for communication with Python
    }else if(command =="OFF"){
      stepper.disableOutputs();
      Serial.println("Turned off.");
    }else if(command =="ON"){
      stepper.enableOutputs();
      Serial.println("Turned on.");
    }
  }
  if (stepper.currentPosition() == stepper.targetPosition()){
    stepper.disableOutputs();
  }
  stepper.run();
}
