#include "ble_task.hpp"
#include "esp_log.h"
#include "esp_nimble_hci.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "console/console.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char *TAG = "BLE_TASK";

static RingBuffer *ble_data_buffer = NULL;

static uint16_t sensor_data_handle; // GATT characteristic handle for sensor data

// Advertising parameters
static struct ble_hs_adv_fields adv_fields;
static uint8_t own_addr_type;

// Forward declarations for NimBLE callbacks
static int ble_gap_event(struct ble_gap_event *event, void *arg);
static void ble_app_on_sync(void);
static void ble_app_on_reset(int reason);
static int gatt_svr_chr_access_sensor_data(uint16_t conn_handle, uint16_t attr_handle,
                                           struct ble_gatt_access_ctxt *ctxt, void *arg);

// GATT Service and Characteristic Definitions
static const struct ble_gatt_svc_def gatt_svr_svcs[] = {
    {
        /*** Service: Sensor Data. */
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = BLE_UUID16_DECLARE(BLE_SVC_UUID_SENSOR_DATA),
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                /*** Characteristic: Sensor Data. */
                .uuid = BLE_UUID16_DECLARE(BLE_CHR_UUID_SENSOR_DATA),
                .access_cb = gatt_svr_chr_access_sensor_data,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
                .val_handle = &sensor_data_handle,
            }, {
                0, /* No more characteristics in this service. */
            }
        },
    }, {
        0, /* No more services. */
    }
};

static int gatt_svr_chr_access_sensor_data(uint16_t conn_handle, uint16_t attr_handle,
                                           struct ble_gatt_access_ctxt *ctxt, void *arg) {
    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        // For read requests, we can just return a dummy value or the last sample
        // For actual streaming, client will subscribe to notifications
        ESP_LOGI(TAG, "GATT Read: Sensor Data characteristic.");
        uint8_t dummy_data[sizeof(sensor_sample_t)] = {0};
        int rc = os_mbuf_append(ctxt->om, &dummy_data, sizeof(dummy_data));
        return rc == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
    }
    return BLE_ATT_ERR_UNLIKELY;
}

static void ble_host_task(void *param) {
    ESP_LOGI(TAG, "BLE Host Task Started");
    nimble_port_run(); // This function will return only when nimble_port_stop() is called
    nimble_port_freertos_deinit();
}

static void gatt_svr_register_cb(struct ble_gatt_register_ctxt *ctxt, void *arg) {
    char buf[BLE_UUID_STR_LEN];

    switch (ctxt->op) {
    case BLE_GATT_REGISTER_OP_SVC:
        MODLOG_DFLT(DEBUG, "registered service %s, handle=%d\n",
                    ble_uuid_to_str(ctxt->svc.svc_def->uuid, buf),
                    ctxt->svc.handle);
        break;

    case BLE_GATT_REGISTER_OP_CHR:
        MODLOG_DFLT(DEBUG, "registered characteristic %s, handle=%d def_handle=%d\n",
                    ble_uuid_to_str(ctxt->chr.chr_def->uuid, buf),
                    ctxt->chr.val_handle,
                    ctxt->chr.def_handle);
        break;

    case BLE_GATT_REGISTER_OP_DSC:
        MODLOG_DFLT(DEBUG, "registered descriptor %s, handle=%d\n",
                    ble_uuid_to_str(ctxt->dsc.dsc_def->uuid, buf),
                    ctxt->dsc.handle);
        break;

    default:
        assert(0);
        break;
    }
}

static void ble_app_on_sync(void) {
    int rc;

    // Set own address
    rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);

    // Figure out address to use for advertising (public or random)
    rc = ble_hs_id_infer_auto(&own_addr_type);
    assert(rc == 0);

    ESP_LOGI(TAG, "BLE Host Synced. Own address type: %d", own_addr_type);

    // Begin advertising
    struct ble_gap_ext_adv_params adv_params;
    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.connectable = 1;
    adv_params.scannable = 1;
    adv_params.legacy_pdu = 1; // Use legacy advertising for broader compatibility
    adv_params.itvl_min = BLE_GAP_ADV_ITVL_MS(100);
    adv_params.itvl_max = BLE_GAP_ADV_ITVL_MS(100);

    // Set advertising data
    adv_fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    adv_fields.name_len = (uint8_t)strlen("TrajectoIMU");
    adv_fields.name = (uint8_t *)"TrajectoIMU";
    adv_fields.uuids16 = (ble_uuid16_t[]){BLE_UUID16_INIT(BLE_SVC_UUID_SENSOR_DATA)};
    adv_fields.num_uuids16 = 1;
    adv_fields.uuids16_is_complete = 1;

    rc = ble_gap_adv_set_fields(&adv_fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "Error setting advertising data; rc=%d", rc);
        return;
    }

    rc = ble_gap_ext_adv_start(own_addr_type, 0, NULL, ble_gap_event, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "Error enabling advertising; rc=%d", rc);
        return;
        }

    ESP_LOGI(TAG, "BLE Advertising started.");
}

static void ble_app_on_reset(int reason) {
    ESP_LOGE(TAG, "BLE Host reset, reason: %d", reason);
}

// GAP event callback
static int ble_gap_event(struct ble_gap_event *event, void *arg) {
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        ESP_LOGI(TAG, "BLE Connection %s: handle=%d, peer_addr=%s",
                 event->connect.status == 0 ? "established" : "failed",
                 event->connect.conn_handle,
                 addr_str(&event->connect.peer_id_addr));
        if (event->connect.status != 0) {
            ble_app_on_reset(event->connect.status);
        }
        break;

    case BLE_GAP_EVENT_DISCONNECT:
        ESP_LOGI(TAG, "BLE Disconnect: reason=%d", event->disconnect.reason);
        // Start advertising again
        ble_app_on_sync();
        break;

    case BLE_GAP_EVENT_SUBSCRIBE:
        ESP_LOGI(TAG, "BLE Subscribe: conn_handle=%d, attr_handle=%d, reason=%d, subscribed=%d",
                 event->subscribe.conn_handle, event->subscribe.attr_handle, event->subscribe.reason,
                 event->subscribe.cur_notifications);
        break;

    case BLE_GAP_EVENT_ADV_COMPLETE:
        ESP_LOGI(TAG, "BLE Advertise Complete.");
        ble_app_on_sync();
        break;

    default:
        break;
    }
    return 0;
}

// Main BLE streaming task
void ble_streaming_task(void *pvParameters) {
    ESP_LOGI(TAG, "BLE Streaming Task started.");
    sensor_sample_t samples[BLE_SAMPLES_PER_NOTIFICATION];
    size_t bytes_to_send = BLE_SAMPLES_PER_NOTIFICATION * sizeof(sensor_sample_t);
    int rc;

    while (1) {
        // Only proceed if there's enough data and the BLE host is synced
        if (ble_data_buffer->available_items() >= BLE_SAMPLES_PER_NOTIFICATION &&
            ble_hs_is_enabled() && !ble_hs_is_reset()) {

            // Iterate through all connected devices
            ble_gap_conn_iter_t conn_it;
            ble_gap_conn_iter_init(&conn_it);
            struct ble_gap_conn_desc desc;

            while (ble_gap_conn_iterate(&conn_it)) {
                if (ble_gap_conn_extract(&conn_it, &desc) == 0) {
                    // Check if characteristic is subscribed for notifications
                    if (ble_gatt_chr_is_subscribed(desc.conn_handle, sensor_data_handle)) {
                        // Read batch of samples
                        for (size_t i = 0; i < BLE_SAMPLES_PER_NOTIFICATION; ++i) {
                            if (!ble_data_buffer->read(samples[i])) {
                                // This should ideally not happen if available_items() check is correct
                                ESP_LOGE(TAG, "Failed to read from ring buffer, buffer state unexpected!");
                                break;
                            }
                        }
                        // Send notification
                        rc = ble_gatt_notify_custom(desc.conn_handle, sensor_data_handle, samples, bytes_to_send);
                        if (rc != 0) {
                            ESP_LOGW(TAG, "BLE notification failed for conn_handle %d; rc=%d", desc.conn_handle, rc);
                        } else {
                            ESP_LOGD(TAG, "Sent %d sensor samples via BLE notification to conn_handle %d.", BLE_SAMPLES_PER_NOTIFICATION, desc.conn_handle);
                        }
                    }
                }
            }
        }
        // Always yield to avoid busy-waiting, even if no data or no connections
        vTaskDelay(pdMS_TO_TICKS(10)); // Yield to other tasks, adjust as needed
    }
}


esp_err_t ble_task_init(RingBuffer *buffer) {
    ble_data_buffer = buffer;

    // Initialize NimBLE host controller
    ESP_ERROR_CHECK(esp_nimble_hci_and_controller_init());
    nimble_port_init();

    // Configure NimBLE stack
    ble_hs_cfg.sync_cb = ble_app_on_sync;
    ble_hs_cfg.reset_cb = ble_app_on_reset;
    ble_hs_cfg.gatts_register_cb = gatt_svr_register_cb;
    ble_hs_cfg.store_status_cb = ble_store_util_status_rr;

    // Set device name
    ble_svc_gap_device_name_set("TrajectoIMU");

    // Set up GATT services
    int rc = ble_gatts_count_cfg(gatt_svr_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed: %d", rc);
        return ESP_FAIL;
    }

    rc = ble_gatts_add_svcs(gatt_svr_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed: %d", rc);
        return ESP_FAIL;
    }

    // Start NimBLE host task
    xTaskCreate(ble_host_task, "ble_host", BLE_TASK_STACK_SIZE, NULL, BLE_TASK_PRIORITY, NULL);

    // Start BLE streaming task
    xTaskCreate(ble_streaming_task, "ble_stream", BLE_TASK_STACK_SIZE, NULL, BLE_TASK_PRIORITY, NULL);

    ESP_LOGI(TAG, "BLE Task Initialized.");
    return ESP_OK;
}
