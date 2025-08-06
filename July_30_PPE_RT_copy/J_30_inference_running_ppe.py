#!/usr/bin/env python
# coding: utf-8

import os, sys
# Fix the path to point to the correct location
sys.path.append('../example/utils/common')  # Changed from '../utils/common'

import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import importlib
import tachy_rt.core.functions as rt_core

from threading import Thread
from functions import *

def create_args():
    def get_parser():
        """
        Get the parser.
        :return: parser
        """
    
        parser = argparse.ArgumentParser(description='Deep learning example script')
    
        parser.add_argument('--model', type=str,
                            help='Model to inference',
                            required=True)

        parser.add_argument('--input_shape', type=str,
                            help='Target input shape(HxWxD)',
                            default=None)

        parser.add_argument('--input_dir', type=str,
                            help='Directory for inputs',
                            required=True)

        # Add new arguments for file paths
        parser.add_argument('--model_path', type=str,
                            help='Path to model file (.tachyrt)',
                            required=True)

        parser.add_argument('--class_json', type=str,
                            help='Path to class.json file',
                            required=True)

        parser.add_argument('--post_config', type=str,
                            help='Path to post-processing config JSON file',
                            required=True)

        parser.add_argument('--post_process_dir', type=str,
                            help='Directory containing post_process.py module',
                            required=True)

        parser.add_argument('--output_dir', type=str,
                            help='Directory to save results',
                            default='./Results')

        parser.add_argument('--upload_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='true')

        parser.add_argument('--path_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='../firmware/tachy-shield')

        args = parser.parse_args()

        ENVS = os.environ
        if not "TACHY_INTERFACE" in ENVS:
            print("Environment \"TACHY_INTERFACE\" is not set")
            exit()

        args.interface = ENVS["TACHY_INTERFACE"]

        return args
    
    args = get_parser()

    args.upload_firmware = True if args.upload_firmware.lower() == 'true' else False
    args.h, args.w = 256, 416  # Fixed dimensions based on your model
    args.model_name = "object_detection_yolov9"

    # Use external arguments for file paths
    args.clss_dict = read_json(args.class_json)
    
    # Load and resize images instead of just getting paths
    image_paths = glob.glob(f'{args.input_dir}/*.jpg') + glob.glob(f'{args.input_dir}/*.png') + glob.glob(f'{args.input_dir}/*.jpeg')
    args.images_input = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (args.w, args.h))  # Resize to model input size
            args.images_input.append(img)
        else:
            print(f"Warning: Could not load image {img_path}")

    if len(args.images_input) == 0:
        print(f"Error: No valid images found in {args.input_dir}")
        exit(-1)

    return args

def boot(args):
    if 'spi' not in args.interface or not args.upload_firmware:
        return

    ''' Upload firmware to device '''
    spi_type = args.interface.split(":")[-1]
    ret = rt_core.boot(path=args.path_firmware, spi_type=spi_type)
    if ret:
        print("Success to boot. Check the status via uart or other api")
    else:
        print("Failed to boot")
        print("Error code :", rt_core.get_last_error_code())
        exit(-1)

def save_model(args):
    ''' Upload model '''
    ret = rt_core.save_model(args.interface, args.model_name, rt_core.MODEL_STORAGE_MEMORY, args.model_path, overwrite=True)
    return ret

def make_instance(args):
    ''' Make runtime config '''
    args.config = \
    {
        "global": {
            "name": args.model_name,
            "data_type": rt_core.DTYPE_FLOAT16,
            "buf_num": 5,
            "max_batch": 1,
            "npu_mask": -1
        },
        "input": [
            {
                "method": rt_core.INPUT_FMT_BINARY,
                "std": 255.0,
                "mean": 0.0
            }
        ],
        "output": {
            "reorder": True
        }
    }

    ''' Make engine '''
    ret = rt_core.make_instance(args.interface, args.model_name, args.model_name, "frame_split", args.config)

    return ret

def connect_instance(args):
    ''' Connect instance '''
    ret, args.instance = rt_core.connect_instance(args.interface, args.model_name)
    if not ret:
        print("connect instance fail")
        print("error :", rt_core.get_last_error_code())
        exit()

    return ret

def inference(args):
    ''' Load Input & Do inference'''
    args.predicts = []
    
    # Load post-processing config once (outside the loop)
    args.post_config_data = read_json(args.post_config)
    sys.path.append(args.post_process_dir)
    
    post_process_module = importlib.import_module('post_process')
    args.post = post_process_module.Decoder(args.post_config_data)
    
    # # DEBUG: Print ALL attributes of the decoder instance
    # print(f"ALL Decoder attributes:")
    # for attr in dir(args.post):
    #     if not attr.startswith('_'):
    #         value = getattr(args.post, attr)
    #         print(f"  {attr}: {value}")
    
    for i, img in enumerate(args.images_input):
        image = img.reshape(-1, args.h, args.w, 3)
        args.instance.process([[image]])
        args.ret = args.instance.get_result()

        # Convert to float32
        output_data = args.ret['buf'].view(np.float32)
        
        args.anno = args.post.main(output_data, np.array([[0, 0, args.w-1, args.h-1]], dtype=np.float32))

        print(f"Image {i+1}: Found {len(args.anno)} detections")
        for j, box in enumerate(args.anno):
            confidence = box[0]
            class_id = int(box[1].astype(np.int32))
            print(f"  Detection {j+1}: confidence={confidence:.3f}, class_id={class_id}")

        for box in args.anno:
            class_id = int(box[1].astype(np.int32))
            # Bounds checking for class ID
            if str(class_id) in args.clss_dict:
                _cls = args.clss_dict[str(class_id)]
            else:
                print(f"Warning: Unknown class ID {class_id}, mapping to class 0")
                _cls = args.clss_dict["0"]  # Default to first class
            
            x0,y0,x1,y1 = box[2:6].astype(np.int32)
            img = cv2.rectangle(img.copy(), (x0,y0), (x1,y1), (255,0,0), 3)
            img = cv2.putText(img, _cls, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        args.predicts.append(img)

    rt_core.deinit_instance(args.interface, args.model_name)

def display(args):
    inputs    = np.concatenate(args.images_input, axis=0)
    total_img = inputs
    predicts = np.concatenate(args.predicts, axis=0)
    total_img = np.concatenate([total_img, predicts], axis=1)

    #cv2.imshow('', total_img)
    #cv2.waitKey(0)
    plt.imshow(total_img)
    plt.axis('off')
    plt.show()
    
    # Use external argument for output directory
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(f'{args.output_dir}/result.png', total_img)

if __name__ == '__main__':

    import time
    script_start = time.time()
    ''' Parse arguments '''
    args = create_args()

    ''' Boot tachy-bs '''
    # boot(args)

    ''' Save model to tachy-bs '''
    save_model(args)

    ''' Make instance for inference '''
    make_instance(args)

    ''' Connect to instance for inference '''
    connect_instance(args)

    ''' Do inference '''
    inference(args)

    ''' Display result '''
    display(args)
    script_total = time.time() - script_start
    print(f"\n=== SCRIPT EXECUTION TIME ===")
    print(f"Total script execution: {script_total:.3f} seconds")
    print(f"=============================")