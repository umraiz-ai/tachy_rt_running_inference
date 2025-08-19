# Raspberry Pi TACHY-BS Driver & Overlay Setup Guide

This guide summarizes the steps to build, install, and configure the TACHY-BS SPI driver and device tree overlay on Raspberry Pi.

---
to update the ssh key

ssh-keygen -R 192.168.0.206



## 1. Build the Kernel Module
with ubuntu version installed 20.04.1 

```sh
sudo apt update
sudo apt install build-essential linux-headers-$(uname -r)
sudo apt install raspberrypi-kernel-headers
cd ~/raspberrypi_contilab/driver/tachy_bs_host_driver_v6
sudo apt install linux-headers-$(uname -r)
make ARCH=arm64
```

#This is the make file

ARCH ?= arm
ccflags-y := -O3
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

obj-m := tachy-bs.o

default:
	$(MAKE) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) -C $(KDIR) M=$(PWD) modules

clean:
	find . -name "*.mod" -exec rm {} \;
	find . -name "*.mod.*" -exec rm {} \;
	find . -name "*.ko" -exec rm {} \;
	find . -name "*.o" -exec rm {} \;
	find . -name ".*.*.cmd" -exec rm {} \;
	find . -name "modules.order" -exec rm {} \;
	find . -name "Module.symvers" -exec rm {} \;



sudo insmod tachy-bs.ko
lsmod | grep tachy_bs

```
If you see output, the module is loaded. If not, check for errors with:


dmesg | tail
---

## 3. Prepare the Device Tree Overlay

- If you have a `.dtbo` file, move it to the overlays directory:
  ```sh


  ```
- If you only have a `.dts` file, compile it:
  ```sh
  dtc -I dts -O dtb -o ~/raspberrypi_contilab/tachy_bs_host.dtbo ~/raspberrypi_contilab/tachy_bs_host.dts
  sudo mv ~/raspberrypi_contilab/tachy_bs_host.dtbo /boot/firmware/overlays/

admin@raspberrypi:~/raspberrypi_contilab/driver/tachy_bs_host_driver_v6 $ lsmod | grep tachy_bs
tachy_bs               16384  0
ubuntu@ubuntu:~/raspberrypi_contilab$ sudo mv ./tachy_bs_host.dtbo /boot/firmware/overlays
  ```

---

## 4. Edit `/boot/firmware/config.txt`



Add or ensure these lines are present:
```
dtparam=spi=on
dtoverlay=tachy_bs_host
```

---

## 5. Reboot

```sh
sudo reboot
```

---

## 6. Confirm Setup

After reboot, check:

```sh
ls /sys/bus/spi/devices/
dmesg | grep tachy
```

You should see SPI devices and driver messages if everything is configured correctly.


sudo insmod tachy-bs.ko

export TACHY_INTERFACE=spi:host

1. Switch to root user
```
pi@raspberrypi:~ $ sudo -Es
root@raspberrypi:~ #
```
sudo apt install python3.8-dev

sudo apt install python3.8-venv

python3 --version
2. Create Python virtual environment

```
root@raspberrypi:~ $ python -m venv pvenv
root@raspberrypi:~ $ source ./pvenv/bin/activate
root@raspberrypi:~ $ apt update
root@raspberrypi:~ $ pip install wheel
root@raspberrypi:~ $ pip install ./tachy_rt-3.1.8-py3-none-any.whl
root@raspberrypi:~ $ pip install opencv-python
root@raspberrypi:~ $ pip install numpy==1.24  (numpy is already instlled, so no need to repeat this step, only when required.)


sudo insmod tachy-bs.ko
export TACHY_INTERFACE=spi:host


lsmod | grep tachy_bs
dmesg | tail


python boot.py




---
./bin/picture.sh
```
![image](./result_picture.png)

1-2. Run example script for sensor <br>
``` bash
./bin/sensor.sh





(pvenv) root@ubuntu:~/raspberrypi_contilab/inference/example/object_detection_yolo_coco-80cls# 

./bin/picture.sh