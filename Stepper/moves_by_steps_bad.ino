#define RED 7 // Coil 1 - A
#define BLU 6 // Coil 1 - B
#define BLK 5 // Coil 2 - A'
#define GRN 4 // Coil 2 - B'

#define POSITIVE 1
#define NEGATIVE 0

#define DELAY_TIME 3 // Adjust to change speed

void setup()
{
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(RED, OUTPUT);
    pinMode(BLU, OUTPUT);
    pinMode(BLK, OUTPUT);
    pinMode(GRN, OUTPUT);

    digitalWrite(RED, LOW);
    digitalWrite(BLU, LOW);
    digitalWrite(BLK, LOW);
    digitalWrite(GRN, LOW);
}

void loop()
{
    // float angle = 90;
    // float steps = angle / 1.8;
    // int i_steps = (int)steps;
    int i_steps = 200;

    for (int i = 0; i < i_steps; i++)
    {
        doStep(NEGATIVE);
    }
    delay(500);
}

void doStep(int dir)
{
    stepMotorWrapper(0, dir);
    delay(DELAY_TIME);
    stepMotorWrapper(1, dir);
    delay(DELAY_TIME);
    stepMotorWrapper(2, dir);
    delay(DELAY_TIME);
    stepMotorWrapper(3, dir);
    delay(DELAY_TIME);
}

void stepMotorWrapper(int step, int dir)
{
    if (dir == POSITIVE)
    {
        stepMotor(step);
    }
    else
    {
        stepMotor(3 - step);
    }
}

void stepMotor(int step)
{
    switch (step)
    {
    case 0:
        digitalWrite(RED, HIGH);
        digitalWrite(BLU, LOW);
        digitalWrite(BLK, HIGH);
        digitalWrite(GRN, LOW);
        break;
    case 1:
        digitalWrite(RED, LOW);
        digitalWrite(BLU, HIGH);
        digitalWrite(BLK, HIGH);
        digitalWrite(GRN, LOW);
        break;
    case 2:
        digitalWrite(RED, LOW);
        digitalWrite(BLU, HIGH);
        digitalWrite(BLK, LOW);
        digitalWrite(GRN, HIGH);
        break;
    case 3:
        digitalWrite(RED, HIGH);
        digitalWrite(BLU, LOW);
        digitalWrite(BLK, LOW);
        digitalWrite(GRN, HIGH);
        break;
    }
}
