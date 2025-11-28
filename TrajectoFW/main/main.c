#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_bt.h"
#include "esp_nimble_hci.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"

static const char *TAG = "NimBLE_Scan";

// Forward declaration for the scan function
static void start_scan(void);

// GAP event handler
static int gap_event_handler(struct ble_gap_event *event, void *arg) {
    struct ble_hs_adv_fields fields;

    switch (event->type) {
        case BLE_GAP_EVENT_DISC:
            // A new device was discovered
            ESP_LOGI(TAG, "Discovered device: addr=%02x:%02x:%02x:%02x:%02x:%02x, rssi=%d",
                     event->disc.addr.val[5], event->disc.addr.val[4],
                     event->disc.addr.val[3], event->disc.addr.val[2],
                     event->disc.addr.val[1], event->disc.addr.val[0],
                     event->disc.rssi);

            // Parse advertising data
            if (ble_hs_adv_parse_fields(&fields, event->disc.data, event->disc.length_data) == 0) {
                if (fields.name != NULL && fields.name_len > 0) {
                    ESP_LOGI(TAG, "  Name: %.*s", fields.name_len, fields.name);
                }
            }
            return 0;

        case BLE_GAP_EVENT_DISC_COMPLETE:
            ESP_LOGI(TAG, "Scan complete.");
            // You could start a new scan here if you want continuous scanning
            return 0;

        default:
            return 0;
    }
}

// Function to start the BLE scan
static void start_scan(void) {
    struct ble_gap_disc_params disc_params;

    // Configure the scan parameters
    disc_params.filter_duplicates = 1; // Report each device only once
    disc_params.passive = 1;           // Use passive scanning (don't send scan requests)
    disc_params.itvl = 0;              // Default interval
    disc_params.window = 0;            // Default window
    disc_params.filter_policy = 0;     // No filter policy
    disc_params.limited = 0;           // Don't limit the scan to limited discoverable mode

    // Start the scan
    int rc = ble_gap_disc(BLE_OWN_ADDR_PUBLIC, BLE_HS_FOREVER, &disc_params, gap_event_handler, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "Failed to start scan; rc=%d", rc);
    } else {
        ESP_LOGI(TAG, "Started scanning for BLE devices...");
    }
}

// This function is called when the NimBLE stack is synchronized
static void on_sync(void) {
    // Ensure we have a public address
    int rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);

    // Start scanning
    start_scan();
}

// The main task for the NimBLE host
void host_task(void *param) {
    ESP_LOGI(TAG, "NimBLE Host Task running");
    nimble_port_run(); // This function will block until nimble_port_stop() is called
    nimble_port_freertos_deinit();
}

void app_main(void) {
    esp_err_t ret;

    // Initialize NVS flash
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialize the ESP Bluetooth controller
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_BLE));

    // Initialize the NimBLE stack
    ESP_ERROR_CHECK(esp_nimble_hci_init());

    ESP_LOGI(TAG, "Free Heap after BT Enable: %lu", esp_get_free_heap_size());

    ret = nimble_port_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "NimBLE Init failed: %d", ret);
        return;
    }

    // Configure the sync callback
    ble_hs_cfg.sync_cb = on_sync;

    // Initialize the NimBLE host task
    nimble_port_freertos_init(host_task);

    ESP_LOGI(TAG, "Application main started");
}