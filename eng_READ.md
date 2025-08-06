This is an example script that utilizes the Tachy-Runtime API.
It provides various functions for controlling a Tachy Device, which are categorized into the following four main groups:

1.  **system function**
    \=\> Provides functions related to system control.
2.  **sensor function**
    \=\> Provides functions related to sensor interface control.
3.  **debug function**
    \=\> Provides debugging functions for the model.
4.  **inference function**
    \=\> Provides functions related to inference.
5.  **example function**
    \=\> This is an example application implemented using the inference functions (item 4).

-----

## How to install

1.  Install the Runtime API (python)
    ```python
    pip install common/whl/tachy_rt-3.2.0-py3-none-any.whl
    ```
2.  Run the install script
    ```bash
    # sudo permission is required (copies the execution script to the /bin directory)
    ./bin/install.sh
    ```

-----

## How to set the interface environment

To communicate with the Tachy Device, you must set the environment variable for the interface as shown below.
The available interfaces differ for each Tachy Device, so you need to check which device you are using.
The possible interfaces are **TCP**, **spi**, or **local**. Below are the available interfaces for each board.

```bash
export TACHY_INTERFACE=spi:host
```

-----

## How to use it

### 1\. System Functions

**1-1. Get status from bs**
\=\> Used to read the current status of the Tachy Device.

```bash
trt_get_device_status
```

**1-2. HostSPI booting**
\=\> Booting using HostSPI.

```bash
# 'TACHY_INTERFACE' must be spi:<type>
trt_boot_device --path <path for firmware> --upload <true or false>
```

**1-3. Save model**
\=\> Saves the model output from the TACHY-Compiler to the BS.

```bash
trt_save_model --name <name> \
               --model <path for model> \
               --storage <memory or nor> \
               --overwrite <true or false>
```

-----

### 2\. Sensor functions

**2-1. Dump sensor image**
\=\> Used to visually check the data coming in through the Sensor Interface.
\=\> The image is FHD (1920x1080) by default, and an image of FHD / Ratio size is written according to the Ratio setting.

```bash
trt_dump_sensor --tx <TX> \
                --crop <x0> <y0> <x1> <y1> \
                --ratio <0 ~ 6> \
                --dtype <float16 or uint8> \
                --rgb <true or false> \
                --fmt <BT1120 or BT656 or Bayer> \
                --inverse_data <true or false> \
                --inverse_sync <true or false> \
                --inverse_clock <true or false> \
                --reset <true or false>
```

**2-2. Enable sensor**

```bash
trt_enable_sensor --tx <TX> \
                  --crop <x0> <y0> <x1> <y1> \
                  --ratio <0 ~ 6> \
                  --dtype <float16 or uint8> \
                  --rgb <true or false> \
                  --fmt <BT1120 or BT656 or Bayer> \
                  --inverse_data <true or false> \
                  --inverse_sync <true or false> \
                  --inverse_clock <true or false> \
                  --reset <true or false>
```

-----

### 3\. Debug functions

**3-1. Evaluate Throughput (=FPS)**
\=\> Measures the model's Throughput (FPS).

```bash
trt_eval_throughput --model <path to model> \
                    --hz <297 or 200 or 100> \
                    --show <true or false> \
                    --dump <true or false>
```

**3-2. Evaluate Latency**
\=\> Measures the model's Latency.

```bash
trt_eval_latency --model <path to model> \
                 --hz <297 or 200 or 100> \
                 --show <true or false> \
                 --dump <true or false>
```

-----

### 4\. Inference functions

TODO

-----

### 5\. Example Function

\=\> Refer to the `example` directory.