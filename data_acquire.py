import asyncio
from bleak import BleakScanner, BleakClient

# 앞서 ESP32 코드에서 설정한 UUID와 일치해야 합니다.
# 16-bit UUID 0x1234는 표준 128-bit UUID 형태로 변환하여 사용합니다.
# Base UUID: 0000xxxx-0000-1000-8000-00805f9b34fb
ECHO_CHARACTERISTIC_UUID = "00001234-0000-1000-8000-00805f9b34fb"
DEVICE_NAME = "ESP-NimBLE-Echo"

# 데이터 수신 시 호출될 콜백 함수 (Notify 핸들러)
async def notification_handler(sender, data):
    """
    ESP32로부터 Notify된 데이터를 처리하는 함수
    """
    decoded_str = data.decode('utf-8')
    print(f"\n[RX] Echo Received from ESP32: {decoded_str}")
    print("Enter message: ", end="", flush=True)

async def main():
    print(f"Scanning for device named '{DEVICE_NAME}'...")

    # 1. 이름으로 장치 검색
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name == DEVICE_NAME
    )

    if not device:
        print(f"Device '{DEVICE_NAME}' not found.")
        return

    print(f"Found {device.name} ({device.address}). Connecting...")

    # 2. 장치 연결
    async with BleakClient(device) as client:
        print(f"Connected: {client.is_connected}")

        # 3. Notify(구독) 시작
        # 데이터가 들어오면 notification_handler 함수가 자동으로 실행됨
        await client.start_notify(ECHO_CHARACTERISTIC_UUID, notification_handler)

        print("Type a message and press Enter (type 'exit' to quit).")

        # 4. 데이터 전송 루프
        while True:
            # 비동기 루프 내에서 input을 받기 위해 run_in_executor 사용 (블로킹 방지)
            msg = await asyncio.get_event_loop().run_in_executor(None, input, "Enter message: ")

            if msg.lower() == 'exit':
                break

            # ESP32로 데이터 전송 (Write)
            # response=True는 ESP32가 수신 확인(ACK)을 보낼 때까지 기다림
            await client.write_gatt_char(ECHO_CHARACTERISTIC_UUID, msg.encode('utf-8'), response=True)

            # Echo가 돌아올 시간을 잠시 대기 (사실 Notify는 즉시 오지만 루프 안정성을 위해)
            await asyncio.sleep(0.1)

        # 5. Notify 중지 및 종료
        await client.stop_notify(ECHO_CHARACTERISTIC_UUID)
        print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())