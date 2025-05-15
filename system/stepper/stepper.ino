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

const int gunPin = 4;
const int dirPin = 3;
const int stepPin = 2;
const int enablePin = 5;
const int speed = 1000;
const int SHOOT_COOLDOWN = 200;  // ms
const float factor = 2/1.9*0.99;

void setup() {
  Serial.begin(9600);  
  
  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, HIGH);  // DISable the driver

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
      Serial.println("Shooting.");
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
      long steps = factor * command.substring(7).toInt(); // Read number of steps
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
      delayMicroseconds(10000);
      digitalWrite(enablePin, HIGH);  // Disable the driver
      Serial.println("Done");
    }
  }
}
