// This code successfully moves a 6-wire stepper motor forward and backward.
// Choose the direction by changing the value of the dir variable in the doStep function in the loop.

#define PIN1 5
#define PIN2 6
#define PIN3 4
#define PIN4 3


#define DELAY_TIME 10 // Adjust to change speed

int step;

// one phase mode
// int truthtable[4][4] = {
//   {LOW, HIGH, HIGH, HIGH},
//   {HIGH, LOW, HIGH, HIGH},
//   {HIGH, HIGH, LOW, HIGH},
//   {HIGH, HIGH, HIGH, LOW}
// };

// Supposed to have higher torque:
int truthtable[4][4] = {
  {LOW, LOW, HIGH, HIGH},
  {HIGH, LOW, LOW, HIGH},
  {HIGH, HIGH, LOW, LOW},
  {LOW, HIGH, HIGH, LOW}
};

void setup()
{
    Serial.begin(9600);
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(PIN1, OUTPUT);
    pinMode(PIN2, OUTPUT);
    pinMode(PIN3, OUTPUT);
    pinMode(PIN4, OUTPUT);

    digitalWrite(PIN1, LOW);
    digitalWrite(PIN2, LOW);
    digitalWrite(PIN3, LOW);
    digitalWrite(PIN4, LOW);

    step = 0;
}

void loop()
{
    for (int i = 0; i < 4; i++) {
      write(i, truthtable[step][i]);
      Serial.print(i);
      Serial.print(':');
      Serial.println(truthtable[step][i]);
    }

    step += 1;
    step %= 4;
    Serial.println();
    delay(DELAY_TIME);
}


void write(int index, int voltage) {
  switch (index) {
    case 1:
      digitalWrite(PIN1, voltage);
      break;
    case 2:
      digitalWrite(PIN2, voltage);
      break;
    case 3:
      digitalWrite(PIN3, voltage);
      break;
    case 4:
      digitalWrite(PIN4, voltage);
      break;
  }
}