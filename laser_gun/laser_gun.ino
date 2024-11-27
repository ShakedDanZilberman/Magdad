#include <Servo.h> // include servo library
 
Servo servo1; // define servos
Servo servo2;
 
int joyX = 0; // give variable to joystick readings
int joyY = 1;
 
int joyVal; // create variable for joystick value
 
void setup()
{
  servo1.attach(4); // start servos
  servo2.attach(3);
  pinMode(8,OUTPUT);
  digitalWrite(8,HIGH);
  delay(1000);
  digitalWrite(8,LOW);
}
 
 
void loop()
{

  joyVal = analogRead(joyX); // read value from joystick
  joyVal = map(joyVal, 0, 1023, 180, 0); // change value range to 0-180
  servo1.write(joyVal); // write value to servo

 
  joyVal = analogRead(joyY); // repeat same for y axis
  joyVal = map(joyVal, 0, 1023, 0, 180);
  servo2.write(joyVal);
  delay(20); // add small delay to reduce noise
 
}