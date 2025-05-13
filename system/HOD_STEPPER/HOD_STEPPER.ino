#include <Servo.h>

// Pins
const int gunPin_I = 12;
const int gunPin_II = 11;
const int dirPin_A = 4;
const int stepPin_A = 7;
const int dirPin_B = 2;
const int stepPin_B = 5;
const int enablePin = 8;
const int SHOOT_COOLDOWN = 2;  // ms
const int speed = 1000;        // Delay between steps in us

// --- Queue definitions ---
const int MAX_QUEUE_SIZE = 30;

struct QueueItem {
  bool isShoot;   // true = SHOOT, false = ROTATE
  long steps;     // for ROTATE (positive/negative), 0 for SHOOT
};

// A queue
QueueItem A_queue[MAX_QUEUE_SIZE];
int A_head = 0, A_tail = 0, A_count = 0;
bool A_busy = false;       // Is A currently executing a command
long A_remainingSteps = 0; // Steps left in current rotation

// B queue
QueueItem B_queue[MAX_QUEUE_SIZE];
int B_head = 0, B_tail = 0, B_count = 0;
bool B_busy = false;
long B_remainingSteps = 0;

// --- Queue helpers ---
void appendA(QueueItem item) {
  if (A_count < MAX_QUEUE_SIZE) {
    A_queue[A_head] = item;
    A_head = (A_head + 1) % MAX_QUEUE_SIZE;
    A_count++;
  } else {
    Serial.println("A_queue full!");
  }
}

void appendB(QueueItem item) {
  if (B_count < MAX_QUEUE_SIZE) {
    B_queue[B_head] = item;
    B_head = (B_head + 1) % MAX_QUEUE_SIZE;
    B_count++;
  } else {
    Serial.println("B_queue full!");
  }
}

QueueItem popA() {
  QueueItem item = A_queue[A_tail];
  A_tail = (A_tail + 1) % MAX_QUEUE_SIZE;
  A_count--;
  return item;
}

QueueItem popB() {
  QueueItem item = B_queue[B_tail];
  B_tail = (B_tail + 1) % MAX_QUEUE_SIZE;
  B_count--;
  return item;
}

// --- Setup ---
void setup() {
  Serial.begin(9600);

  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, HIGH);  // Disable motor driver

  pinMode(gunPin_I, OUTPUT);
  pinMode(gunPin_II, OUTPUT);
  pinMode(dirPin_A, OUTPUT);
  pinMode(dirPin_B, OUTPUT);
  pinMode(stepPin_A, OUTPUT);
  pinMode(stepPin_B, OUTPUT);
}

// --- Shoot ---
void shoot(int gunPin) {
  digitalWrite(gunPin, HIGH);
  delay(SHOOT_COOLDOWN);
  digitalWrite(gunPin, LOW);
  Serial.println("Shoot done");
}

// --- Rotate one step ---
void doOneStep(int dirPin, int stepPin, int enablePin, int direction) {
  digitalWrite(enablePin, LOW);  // Enable driver
  digitalWrite(dirPin, direction);
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(speed);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(speed);
}

// --- Process A queue ---
void processA() {
  if (!A_busy && A_count > 0) {
    QueueItem cmd = popA();
    if (cmd.isShoot) {
      shoot(gunPin_I);
    } else {
      A_remainingSteps = abs(cmd.steps);
      int dir = cmd.steps > 0 ? HIGH : LOW;
      digitalWrite(dirPin_A, dir);
      A_busy = true;
    }
  }

  if (A_busy && A_remainingSteps > 0) {
    doOneStep(dirPin_A, stepPin_A, enablePin, digitalRead(dirPin_A));
    A_remainingSteps--;
    if (A_remainingSteps == 0) {
      A_busy = false;
      digitalWrite(enablePin, HIGH);  // Disable driver when done
      Serial.println("A rotation done");
    }
  }
}

// --- Process B queue ---
void processB() {
  if (!B_busy && B_count > 0) {
    QueueItem cmd = popB();
    if (cmd.isShoot) {
      shoot(gunPin_II);
    } else {
      B_remainingSteps = abs(cmd.steps);
      int dir = cmd.steps > 0 ? HIGH : LOW;
      digitalWrite(dirPin_B, dir);
      B_busy = true;
    }
  }

  if (B_busy && B_remainingSteps > 0) {
    doOneStep(dirPin_B, stepPin_B, enablePin, digitalRead(dirPin_B));
    B_remainingSteps--;
    if (B_remainingSteps == 0) {
      B_busy = false;
      digitalWrite(enablePin, HIGH);  // Disable driver
      Serial.println("B rotation done");
    }
  }
}

// --- Loop ---
void loop() {
  // Handle incoming commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("ROTATE_A:")) {
      long steps = command.substring(9).toInt();
      QueueItem item = {false, steps};
      appendA(item);
    } else if (command.startsWith("ROTATE_B:")) {
      long steps = command.substring(9).toInt();
      QueueItem item = {false, steps};
      appendB(item);
    } else if (command == "SHOOT_A") {
      QueueItem item = {true, 0};
      appendA(item);
    } else if (command == "SHOOT_B") {
      QueueItem item = {true, 0};
      appendB(item);
    }
  }

  // Process A and B queues step-by-step
  processA();
  processB();
}
