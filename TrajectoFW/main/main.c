#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_bt.h"             // 컨트롤러
#include "esp_nimble_hci.h"     // [필수] HCI 인터페이스
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"

static const char *TAG = "NimBLE_FINAL";

// 스캔 재시작을 위한 전방 선언
static void ble_app_scan(void);

// GAP 이벤트 핸들러 (스캔 결과 처리)
static int ble_gap_event(struct ble_gap_event *event, void *arg) {
    struct ble_hs_adv_fields fields;
    int rc;

    switch (event->type) {
        case BLE_GAP_EVENT_DISC:
            rc = ble_hs_adv_parse_fields(&fields, event->disc.data, event->disc.length_data);
            if (rc != 0) return 0;

            // 장치 발견 로그
            ESP_LOGI(TAG, "Device: %02x:%02x:%02x:%02x:%02x:%02x | RSSI: %d | Name: %.*s",
                     event->disc.addr.val[5], event->disc.addr.val[4],
                     event->disc.addr.val[3], event->disc.addr.val[2],
                     event->disc.addr.val[1], event->disc.addr.val[0],
                     event->disc.rssi,
                     fields.name_len, fields.name);
            return 0;

        case BLE_GAP_EVENT_DISC_COMPLETE:
            ESP_LOGI(TAG, "스캔 완료. 재시작합니다...");
            ble_app_scan(); // 무한 스캔
            return 0;

        default:
            return 0;
    }
}

static void ble_app_scan(void) {
    struct ble_gap_disc_params disc_params;
    
    disc_params.filter_duplicates = 1; // 중복 제거
    disc_params.passive = 0;           // Active Scan
    disc_params.itvl = 0;
    disc_params.window = 0;
    disc_params.filter_policy = 0;
    disc_params.limited = 0;

    int rc = ble_gap_disc(BLE_OWN_ADDR_PUBLIC, 5000, &disc_params, ble_gap_event, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "스캔 시작 실패 rc=%d", rc);
    }
}

static void ble_app_on_sync(void) {
    int rc;
    // 주소 타입 결정
    rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);
    
    // 스캔 시작
    ble_app_scan();
}

void ble_host_task(void *param) {
    ESP_LOGI(TAG, "NimBLE Host Task Started");
    nimble_port_run();
    nimble_port_freertos_deinit();
}

void app_main(void) {
    esp_err_t ret;

    // 1. NVS 초기화
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. 컨트롤러 초기화 (아까 성공한 그 코드)
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));

    // 3. 컨트롤러 활성화
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_BLE));

    // 4. [핵심] HCI 초기화 (이게 추가됨!)
    ESP_ERROR_CHECK(esp_nimble_hci_init());

    // 5. NimBLE 포트 초기화
    nimble_port_init();

    // 6. 태스크 시작
    ble_hs_cfg.sync_cb = ble_app_on_sync;
    nimble_port_freertos_init(ble_host_task);
    
    ESP_LOGI(TAG, "시스템 정상 가동 중...");
}