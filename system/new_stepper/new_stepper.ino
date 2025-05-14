#include <AccelStepper.h>

// Pins
const int gunPin = 2;
const int dirPin = 4;
const int stepPin = 7;
const int enablePin = 8;
const int SHOOT_COOLDOWN = 2;  // ms
const int MAX_SPEED = 2000;  // steps per second
const int ACCELERATION = 10000;  // steps per second^2

// Stepper setup: DRIVER interface type = 1
AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);


void setup() {
  Serial.begin(9600);

  pinMode(gunPin, OUTPUT);
  pinMode(enablePin, OUTPUT);

  // stepper.setEnablePin(enablePin);

  // stepper.disableOutputs();

  // Stepper setup
  stepper.setMaxSpeed(MAX_SPEED);        
  stepper.setAcceleration(ACCELERATION);     
  stepper.setCurrentPosition(0);    // reset position
}


void loop() {
  // if (stepper.distanceToGo() == 0){
  //   stepper.disableOutputs();
  // }else{
  //   stepper.enableOutputs();
  // }
  stepper.run();
    // if (stepper.run()){
    //   stepper.enableOutputs();
    // } 
    // else{
    //   stepper.disableOutputs();
    // }

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
      long steps = command.substring(7).toInt();
      Serial.print("Requested steps: ");
      Serial.println(steps);
      stepper.move(steps);
    }
  }
}
