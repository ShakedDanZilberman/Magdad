// Define pin connections & motor's steps per revolution
const int dirPin = 2;
const int stepPin = 3;
const int stepsPerRevolution = 200;

void setup()
{
	// Declare pins as Outputs
	pinMode(stepPin, OUTPUT);
	pinMode(dirPin, OUTPUT);
  	// Set motor direction clockwise
	digitalWrite(dirPin, HIGH);
}
void loop()
{
  digitalWrite(stepPin, HIGH);
  delay(5000);
  digitalWrite(stepPin, LOW);
  delay(5000);
}