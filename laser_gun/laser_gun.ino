#include <Servo.h> // include servo library
 
Servo servo1; // define servos
Servo servo2;
 
int joyX = 0; // give variable to joystick readings
int joyY = 1;
 
int joyVal; // create variable for joystick value
 
void setup()
{
  servo1.attach(9); // start servos
  servo2.attach(3);
  pinMode(8,OUTPUT);

}
 
 
void loop()
{

  servo1.write(0); // write value to servo
  delay(2000); // add small delay to reduce noise
  servo1.write(180); // write value to servo
  delay(2000);
}