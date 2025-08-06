when we reboot raspberry pi
sudo insmod ./tachy-bs.ko

how to run this script
sudo -Es
source pvenv/bin/activate
export TACHY_INTERFACE=spi:host
cd /raspberrypi_contilab/inference/July_30_PPE_RT_copy

python J_30_inference_running_ppe.py \
    --model 20250728_085213 \
    --input_shape 256x416x3 \
    --input_dir ./example_PPE \
    --model_path ./20250728_085213/model_256x416x3_inv-f.tachyrt \
    --class_json ./20250728_085213/class.json \
    --post_config ./20250728_085213/post_process_256x416.json \
    --post_process_dir ./20250728_085213 \
    --output_dir ./Results