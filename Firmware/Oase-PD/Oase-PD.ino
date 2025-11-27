#include <bluefruit.h>
#include <Adafruit_LSM6DS3.h>

// Create a LSM6DS3 object
Adafruit_LSM6DS3 myIMU;

// BLE Service & Characteristics
// UUIDs in reverse order
uint8_t service_uuid[16] = { 0x14, 0x12, 0x8A, 0x76, 0x04, 0xD1, 0x6C, 0x4F, 0x7E, 0x53, 0xF2, 0xE8, 0x00, 0x00, 0xB1, 0x19 };
uint8_t control_uuid[16] = { 0x14, 0x12, 0x8A, 0x76, 0x04, 0xD1, 0x6C, 0x4F, 0x7E, 0x53, 0xF2, 0xE8, 0x07, 0x00, 0xB1, 0x19 };
uint8_t data_uuid[16]    = { 0x14, 0x12, 0x8A, 0x76, 0x04, 0xD1, 0x6C, 0x4F, 0x7E, 0x53, 0xF2, 0xE8, 0x08, 0x00, 0xB1, 0x19 };

BLEService        sensorService(service_uuid);
BLECharacteristic controlCharacteristic(control_uuid);
BLECharacteristic dataCharacteristic(data_uuid);

#define START_CMD 0x01
#define STOP_CMD 0x02

// Force Sensor
#define FORCE_SENSOR_PIN A0
#define FORCE_THRESHOLD_HIGH 500
#define FORCE_THRESHOLD_LOW 400

// Data Buffer
#define MAX_SAMPLES 100
#define PACKET_SAMPLES 20
float imu_buffer[MAX_SAMPLES][7];
int sample_count = 0;

// State Machine
enum State {
  IDLE,
  WAITING_FOR_FORCE,
  RECORDING,
  SENDING
};
State currentState = IDLE;

void control_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len);
void connect_callback(uint16_t conn_handle);
void disconnect_callback(uint16_t conn_handle, uint8_t reason);


void setup() {
  Serial.begin(9600);
  while (!Serial);

  pinMode(FORCE_SENSOR_PIN, INPUT);

  // Initialize IMU sensor
  if (!myIMU.begin_I2C()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Configure IMU
  myIMU.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  myIMU.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
  myIMU.setAccelDataRate(LSM6DS_RATE_416_HZ);
  myIMU.setGyroDataRate(LSM6DS_RATE_416_HZ);

  // Set up BLE
  Bluefruit.begin();
  Bluefruit.setName("Oase-PD");
  Bluefruit.setTxPower(4);
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);

  // Setup Service
  sensorService.begin();

  // Setup Characteristics
  controlCharacteristic.setProperties(CHR_PROPS_READ | CHR_PROPS_WRITE | CHR_PROPS_NOTIFY);
  controlCharacteristic.setPermission(SECMODE_OPEN, SECMODE_OPEN);
  controlCharacteristic.setFixedLen(1);
  controlCharacteristic.setWriteCallback(control_write_callback);
  controlCharacteristic.begin();
  controlCharacteristic.write8(STOP_CMD);

  dataCharacteristic.setProperties(CHR_PROPS_READ | CHR_PROPS_NOTIFY);
  dataCharacteristic.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  dataCharacteristic.setMaxLen(PACKET_SAMPLES * 6 * sizeof(float));
  dataCharacteristic.begin();

  // Setup Advertising
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(sensorService);
  Bluefruit.Advertising.addName();
  Bluefruit.Advertising.start(0); // 0 = Advertise forever

  Serial.println("Bluetooth device active, waiting for connections...");
}

void loop() {
  switch (currentState) {
    case IDLE:
      // Waiting for START_CMD in callback
      break;

    case WAITING_FOR_FORCE: {
      int forceValue = analogRead(FORCE_SENSOR_PIN);
      Serial.print("Force: ");
      Serial.println(forceValue);
      if (forceValue > FORCE_THRESHOLD_HIGH) {
        currentState = RECORDING;
        sample_count = 0;
        Serial.println("State: RECORDING");
      }
      // Check for STOP_CMD is handled in the callback
      break;
    }

    case RECORDING: {
      int forceValue = analogRead(FORCE_SENSOR_PIN);
      if (analogRead(FORCE_SENSOR_PIN) < FORCE_THRESHOLD_LOW) {
        currentState = SENDING;
        Serial.println("State: SENDING (force low)");
      } else if (sample_count < MAX_SAMPLES) {
        sensors_event_t accel;
        sensors_event_t gyro;
        sensors_event_t temp;
        myIMU.getEvent(&accel, &gyro, &temp);
        imu_buffer[sample_count][0] = accel.acceleration.x;
        imu_buffer[sample_count][1] = accel.acceleration.y;
        imu_buffer[sample_count][2] = accel.acceleration.z;
        imu_buffer[sample_count][3] = gyro.gyro.x;
        imu_buffer[sample_count][4] = gyro.gyro.y;
        imu_buffer[sample_count][5] = gyro.gyro.z;
        imu_buffer[sample_count][6] = forceValue;
        sample_count++;
      } else { // Buffer is full
        currentState = SENDING;
        Serial.println("State: SENDING (buffer full)");
      }
      break;
    }

    case SENDING: {
      Serial.println("Sending data...");
      int num_packets = (sample_count + PACKET_SAMPLES - 1) / PACKET_SAMPLES;
      for (int i = 0; i < num_packets; i++) {
        int offset = i * PACKET_SAMPLES;
        int num_samples_in_packet = min(PACKET_SAMPLES, sample_count - offset);
        dataCharacteristic.notify((uint8_t*)&imu_buffer[offset][0], num_samples_in_packet * 6 * sizeof(float));
      }
      Serial.println("Data sent.");
      currentState = WAITING_FOR_FORCE;
      break;
    }
  }
}

void control_write_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint8_t* data, uint16_t len) {
  if (len > 0) {
    if (data[0] == START_CMD) {
      if (currentState == IDLE) {
        currentState = WAITING_FOR_FORCE;
        Serial.println("State: WAITING_FOR_FORCE");
        controlCharacteristic.write8(STOP_CMD); // Acknowledge
      }
    } else if (data[0] == STOP_CMD) {
      if (currentState == WAITING_FOR_FORCE || currentState == RECORDING) {
        currentState = IDLE;
        Serial.println("State: IDLE");
      }
    }
  }
}

void connect_callback(uint16_t conn_handle) {
  Serial.print("Connected to central: ");
  char central_name[32] = { 0 };
  BLEConnection* conn = Bluefruit.Connection(conn_handle);
  if (conn) {
    conn->getPeerName(central_name, sizeof(central_name));
  }
  Serial.println(central_name);
}

void disconnect_callback(uint16_t conn_handle, uint8_t reason) {
  (void) conn_handle;
  (void) reason;
  Serial.println("Disconnected");
  currentState = IDLE; // Reset state on disconnect
}
