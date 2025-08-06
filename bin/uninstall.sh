#!/bin/bash

echo "Remove debug script ..."
sudo rm /bin/trt_eval_latency
sudo rm /bin/trt_eval_throughput

echo "Remove sensor script ..."
sudo rm /bin/trt_init_zeeahn
sudo rm /bin/trt_dump_sensor
sudo rm /bin/trt_enable_sensor

echo "Remove system script ..."
sudo rm /bin/trt_save_model
sudo rm /bin/trt_delete_model
sudo rm /bin/trt_boot_device
sudo rm /bin/trt_get_device_status
