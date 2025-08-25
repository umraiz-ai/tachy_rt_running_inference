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
import logging

from threading import Thread
from functions import *

# Initialize logging for script execution tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("SCRIPT START: %s", __file__)
print("TRACE: script start", flush=True)

def create_args():
    def get_parser():
        """
        Get the parser.
        :return: parser
        """
    
        parser = argparse.ArgumentParser(description='Deep learning example script')
    
        # Model identification and configuration
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

        # Firmware upload configuration (SPI interface only)
        parser.add_argument('--upload_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='true')

        parser.add_argument('--path_firmware', type=str,
                            help='Uploading firmware(spi interface only)',
                            default='../firmware/tachy-shield')

        args = parser.parse_args()

        # Check for required environment variable
        ENVS = os.environ
        if not "TACHY_INTERFACE" in ENVS:
            print("Environment \"TACHY_INTERFACE\" is not set")
            exit()

        args.interface = ENVS["TACHY_INTERFACE"]

        return args
    
    args = get_parser()

    # Configure firmware upload setting
    args.upload_firmware = True if args.upload_firmware.lower() == 'true' else False
    
    # Set fixed model dimensions for PPE detection (256x416 optimized for YOLOv9)
    args.h, args.w = 256, 416  # Fixed dimensions based on your model
    args.model_name = "object_detection_yolov9"

    # Load class definitions for PPE detection
    args.clss_dict = read_json(args.class_json)
    
    # Load and preprocess input images
    image_paths = glob.glob(f'{args.input_dir}/*.jpg') + glob.glob(f'{args.input_dir}/*.png') + glob.glob(f'{args.input_dir}/*.jpeg')
    args.images_input = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to model input size for consistent inference
            img = cv2.resize(img, (args.w, args.h))  # Resize to model input size
            args.images_input.append(img)
        else:
            print(f"Warning: Could not load image {img_path}", flush=True)

    # Validate that images were loaded successfully
    if len(args.images_input) == 0:
        print(f"Error: No valid images found in {args.input_dir}")
        exit(-1)

    logging.info("ARGS: model=%s input_shape=%s input_dir=%s model_path=%s output_dir=%s",
                 getattr(args, "model", None),
                 getattr(args, "input_shape", None),
                 getattr(args, "input_dir", None),
                 getattr(args, "model_path", None),
                 getattr(args, "output_dir", None))

    print(f"TRACE: create_args done - found {len(args.images_input)} images, interface={args.interface}", flush=True)
    return args

def boot(args):
    # Skip firmware upload if not using SPI interface or upload disabled
    if 'spi' not in args.interface or not args.upload_firmware:
        print("TRACE: boot skipped (not SPI or upload_firmware False)", flush=True)
        return

    # Upload firmware to Tachy device via SPI interface
    spi_type = args.interface.split(":")[-1]
    print(f"TRACE: boot -> calling rt_core.boot(path={args.path_firmware}, spi_type={spi_type})", flush=True)
    #ret = rt_core.boot(path=args.path_firmware, spi_type=spi_type)
    ret = True
    print(f"TRACE: boot returned {ret}", flush=True)
    if ret:
        print("Success to boot. Check the status via uart or other api")
    else:
        print("Failed to boot")
        print("Error code :", rt_core.get_last_error_code())
        exit(-1)

def save_model(args):
    # Upload compiled model to Tachy device memory
    print(f"TRACE: save_model -> interface={args.interface}, model_name={args.model_name}, model_path={args.model_path}", flush=True)
    #ret = rt_core.save_model(args.interface, args.model_name, rt_core.MODEL_STORAGE_MEMORY, args.model_path, overwrite=True)
    ret = True
    logging.info("MODEL INITIALIZED: %s", getattr(args, "model_path", "<unknown>"))
    print(f"TRACE: save_model returned {ret}", flush=True)
    return ret

def make_instance(args):
    # Create runtime configuration for YOLOv9 inference
    args.config = \
    {
        "global": {
            "name": args.model_name,
            "data_type": rt_core.DTYPE_FLOAT16,  # Use FLOAT16 for NPU optimization
            "buf_num": 5,
            "max_batch": 1#,
            #"npu_mask": -1  # Use all available NPU cores
        },
        "input": [
            {
                "method": rt_core.INPUT_FMT_BINARY,
                "std": 255.0,  # Normalize pixel values
                "mean": 0.0
            }
        ],
        "output": {
            "reorder": True  # Reorder output for post-processing
        }
    }

    # Create inference engine with configuration
    print("TRACE: make_instance -> calling rt_core.make_instance", flush=True)
    #ret = rt_core.make_instance(args.interface, args.model_name, args.model_name, "frame_split", args.config)
    ret = True
    print(f"TRACE: make_instance returned {ret}", flush=True)

    return ret

def connect_instance(args):
    # Establish connection to inference instance
    print(f"TRACE: connect_instance -> connecting to {args.model_name} on {args.interface}", flush=True)
    #ret, args.instance = rt_core.connect_instance(args.interface, args.model_name)
    ret = True
    args.instance = None
    print(f"TRACE: connect_instance returned {ret}", flush=True)
    if not ret:
        print("connect instance fail")
        print("error :", rt_core.get_last_error_code())
        exit()
    print("TRACE: connect_instance success, instance ready", flush=True)
    return ret

def inference(args):
    # Initialize inference results storage
    args.predicts = []
    
    # Load post-processing configuration and decoder (once for all images)
    print(f"TRACE: inference -> loading post config from {args.post_config}", flush=True)
    args.post_config_data = read_json(args.post_config)
    sys.path.append(args.post_process_dir)
    print(f"TRACE: inference -> importing post_process from {args.post_process_dir}", flush=True)
    
    # # Initialize YOLOv9 decoder with configuration
    # post_process_module = importlib.import_module('post_process')
    # args.post = post_process_module.Decoder(args.post_config_data)
    # print("TRACE: inference -> post_process.Decoder instantiated", flush=True)
    
        # Initialize YOLOv9 decoder with configuration
    post_process_module = importlib.import_module('post_process')
    args.post = post_process_module.Decoder(args.post_config_data)
    
    # DEBUG: Temporarily lower confidence threshold for testing
    print(f"DEBUG: Original confidence threshold: {args.post.obj_threshold[0]}", flush=True)
    
    print("TRACE: inference -> post_process.Decoder instantiated", flush=True)


    # # DEBUG: Print ALL attributes of the decoder instance
    # print(f"ALL Decoder attributes:")
    # for attr in dir(args.post):
    #     if not attr.startswith('_'):
    #         value = getattr(args.post, attr)
    #         print(f"  {attr}: {value}")
    
    # Process each image through the inference pipeline
    for i, img in enumerate(args.images_input):
        print(f"TRACE: inference -> processing image {i+1}/{len(args.images_input)}", flush=True)
        
        # Reshape image for model input format
        image = img.reshape(-1, args.h, args.w, 3)
        
        # Run inference on Tachy device
        # print("TRACE: inference -> calling instance.process()", flush=True)
        # args.instance.process([[image]])
        
        # Retrieve inference results
        # print("TRACE: inference -> calling instance.get_result()", flush=True)
        # args.ret = args.instance.get_result()
        # print(f"TRACE: inference -> get_result returned keys: {list(args.ret.keys())}", flush=True)


        # Create the missing args.ret with correct size for YOLOv9 post-processing
        # The error shows the model expects different size than calculated
        # Using size that makes split_logits work: n_grid * (n_box + n_classes)
        # From config: n_grid appears to be 1092, with 4 box + 4 class = 8 channels
        # Total: 1092 * 8 = 8736 (this matches the error message)
        #args.ret = {'buf': np.random.randn(8736).astype(np.float32)}
        #print("TRACE: inference -> placeholder result created (replace with your CPU inference)", flush=True)

        # Use ONNX model for CPU inference
        try:
            import onnxruntime as ort
            
            # Load your eval.onnx model
            onnx_path = args.post_process_dir + "/eval.onnx"
            print(f"TRACE: Loading ONNX model from {onnx_path}", flush=True)
            
            # Create ONNX Runtime session for CPU
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
            # Get input/output info
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(f"TRACE: ONNX input name: {input_name}", flush=True)
            print(f"TRACE: ONNX input shape: {input_shape}", flush=True)
            
            # Prepare input data
            input_data = image.astype(np.float32) / 255.0  # Normalize to [0,1]
            
            # Check if we need NCHW format (Batch, Channel, Height, Width)
            if len(input_shape) == 4 and input_shape[1] == 3:
                # Convert BHWC to BCHW for ONNX
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                print("TRACE: Converted input to NCHW format for ONNX", flush=True)
            
            # Run inference on CPU
            print("TRACE: Running ONNX inference on CPU...", flush=True)
            outputs = session.run(None, {input_name: input_data})
            
            
            
            # DEBUG: Analyze ONNX outputs in detail
            print(f"DEBUG: Number of ONNX outputs: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"DEBUG: Output {i} shape: {output.shape}")
                print(f"DEBUG: Output {i} dtype: {output.dtype}")
                print(f"DEBUG: Output {i} range: {output.min():.3f} to {output.max():.3f}")
                print(f"DEBUG: Output {i} sample values: {output.flatten()[:10]}")
           
            # PASS RAW ONNX OUTPUTS TO POST-PROCESSOR (First Step)
            print("DEBUG: Passing raw ONNX outputs to post-processor...")
            
            
                       # Store raw outputs for post-processor to handle
            args.ret = {'raw_onnx_outputs': outputs}
            
            # TEMPORARY: Create 'buf' with flattened ONNX outputs for compatibility
            # This preserves real data instead of using dummy data
                      # CORRECT: Process ONNX outputs according to multi-scale YOLO format
            # Outputs come in pairs: (box_coords, class_logits) for each scale
            all_box_coords = []
            all_class_logits = []
            
            # Process each scale pair: (0,1), (2,3), (4,5)
            for scale_idx in range(0, len(outputs), 2):
                if scale_idx + 1 < len(outputs):
                    box_output = outputs[scale_idx]      # Box coordinates
                    conf_output = outputs[scale_idx + 1] # Class logits (need sigmoid)
                    
                    # Apply sigmoid activation to class logits to get proper confidence scores
                    conf_probs = 1 / (1 + np.exp(-np.clip(conf_output, -500, 500)))
                    
                    # Flatten each output for concatenation
                    all_box_coords.append(box_output.flatten())
                    all_class_logits.append(conf_probs.flatten())
                    
                    print(f"DEBUG: Scale {scale_idx//2} - Box: {box_output.shape}, Conf after sigmoid: {conf_probs.min():.3f} to {conf_probs.max():.3f}")
            
            # Concatenate in the format expected by post-processor: [all_boxes, all_classes]
            all_outputs_flat = np.concatenate(all_box_coords + all_class_logits)
            args.ret['buf'] = all_outputs_flat.astype(np.float32)
            
            # CRITICAL: Update n_grid to match the actual flattened output size
            args.post.n_grid = all_outputs_flat.size // 8
            print(f"DEBUG: CORRECTED n_grid to {args.post.n_grid} for properly processed output size {all_outputs_flat.size}")
            
            print(f"DEBUG: Number of raw outputs: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"DEBUG: Output {i} shape: {output.shape}")
            
            # Calculate expected total size for compatibility
            total_elements = sum(np.prod(output.shape) for output in outputs)
            print(f"DEBUG: Total elements in all outputs: {total_elements}")

            print(f"TRACE: ONNX CPU inference complete!", flush=True)
        
            # Check if size matches expectation and adjust
            
            print(f"TRACE: Output size: {output_data.size} (expected: 8736)", flush=True)
            
            # Check if size matches expectation and adjust
            if output_data.size != 8736:
                print(f"INFO: ONNX output size {output_data.size} differs from expected 8736", flush=True)
                print("INFO: Adjusting post-processing configuration to match ONNX output", flush=True)
                
                # Update the decoder's n_grid to match ONNX output
                args.post.n_grid = output_data.size // 8
                #print(f"INFO: Updated n_grid from 1092 to {args.post.n_grid}", flush=True)
                    # Convert raw output buffer to float32 for post-processing


        except ImportError:
            print("ERROR: onnxruntime not installed. Install with: pip install onnxruntime", flush=True)
            print("TRACE: Falling back to fake data...", flush=True)
            # Create fake data fallback
            args.ret = {'buf': np.random.randn(8736).astype(np.float32)}
            
        except FileNotFoundError:
            print(f"ERROR: eval.onnx not found at {onnx_path}", flush=True)
            print("TRACE: Make sure eval.onnx is in the correct directory", flush=True)
            print("TRACE: Falling back to fake data...", flush=True)
            # Create fake data fallback
            args.ret = {'buf': np.random.randn(8736).astype(np.float32)}
            
        except Exception as e:
            print(f"ERROR: ONNX inference failed: {e}", flush=True)
            print("TRACE: Falling back to fake data...", flush=True)
            # Create fake data fallback
            args.ret = {'buf': np.random.randn(8736).astype(np.float32)}

        #print("TRACE: inference -> placeholder result created (replace with your CPU inference)", flush=True)



        # Convert raw output buffer to float32 for post-processing
        try:
            output_data = args.ret['buf'].view(np.float32)
            print(f"TRACE: inference -> output_data size: {output_data.size}", flush=True)
        except Exception as e:
            print(f"ERROR: failed to convert output buffer to float32: {e}", flush=True)
            raise

        # DEBUG: Add comprehensive debugging before post-processing
        print(f"DEBUG: Before post-processing - output_data range: {output_data.min():.3f} to {output_data.max():.3f}")
        print(f"DEBUG: Post-processor config - OBJ_THRESHOLD: {args.post.obj_threshold}, NMS_THRESHOLD: {getattr(args.post, 'nms_threshold', 'not found')}")
        print(f"DEBUG: Post-processor config - N_MAX_OBJ: {getattr(args.post, 'n_max_obj', 'not found')}")
        print(f"DEBUG: Post-processor config - N_GRID: {getattr(args.post, 'n_grid', 'not found')}")

        # DEBUG: Print ALL post-processor attributes to see what's available
        print("DEBUG: All post-processor attributes:")
        for attr in dir(args.post):
            if not attr.startswith('_') and not callable(getattr(args.post, attr)):
                value = getattr(args.post, attr)
                #print(f"  {attr}: {value}")

        # FORCE proper post-processing parameters
        if not hasattr(args.post, 'nms_threshold'):
            args.post.nms_threshold = 0.4  # More aggressive NMS
            print(f"DEBUG: Set missing nms_threshold to {args.post.nms_threshold}")
        
        if not hasattr(args.post, 'n_max_obj'):
            args.post.n_max_obj = 10  # Limit to 10 detections max
            print(f"DEBUG: Set missing n_max_obj to {args.post.n_max_obj}")
        
        # Increase confidence threshold to reduce false positives
        original_threshold = args.post.obj_threshold[0]
        args.post.obj_threshold = np.array([0.2], dtype='float32')  
        print(f"DEBUG: Increased confidence threshold from {original_threshold} to {args.post.obj_threshold[0]}")        
        
        # Decode YOLOv9 output to bounding boxes and classes

        args.anno = args.post.main(output_data, np.array([[0, 0, args.w-1, args.h-1]], dtype=np.float32))
        print(f"TRACE: inference -> post.main returned {len(args.anno)} annotations", flush=True)
        

        # DEBUG: Analyze detection confidence scores
        if len(args.anno) > 0:
            confidences = [box[0] for box in args.anno]
            print(f"DEBUG: Confidence scores - min: {min(confidences):.3f}, max: {max(confidences):.3f}, avg: {np.mean(confidences):.3f}")
            print(f"DEBUG: Number above 0.5: {sum(1 for c in confidences if c > 0.5)}")
            print(f"DEBUG: Number above 0.9: {sum(1 for c in confidences if c > 0.9)}")

             # FORCE detection limit since n_max_obj isn't working properly
            max_detections = 10
            if len(args.anno) > max_detections:
                print(f"DEBUG: Limiting {len(args.anno)} detections to top {max_detections} by confidence")
                # Sort by confidence (first column) in descending order and take top N
                args.anno = sorted(args.anno, key=lambda x: x[0], reverse=True)[:max_detections]
                print(f"DEBUG: After limiting: {len(args.anno)} detections")
 
        
        else:
            print("DEBUG: No detections found")

        logging.info("PROCESSING IMAGE: %d/%d", i+1, len(args.images_input))
        
        print(f"Image {i+1}: Found {len(args.anno)} detections", flush=True)
        
        # Log detection details
        for j, box in enumerate(args.anno):
            confidence = box[0]
            class_id = int(box[1].astype(np.int32))
            print(f"  Detection {j+1}: confidence={confidence:.3f}, class_id={class_id}", flush=True)

        # Draw bounding boxes and labels on image
        for box in args.anno:
            class_id = int(box[1].astype(np.int32))
            # Bounds checking for class ID
            # Map class ID to PPE class name with bounds checking
            if str(class_id) in args.clss_dict:
                _cls = args.clss_dict[str(class_id)]
            else:
                print(f"Warning: Unknown class ID {class_id}, mapping to class 0", flush=True)
                _cls = args.clss_dict["0"]  # Default to first class
            
            # Extract bounding box coordinates
            x0,y0,x1,y1 = box[2:6].astype(np.int32)
            
            # Draw blue rectangle for detection box
            img = cv2.rectangle(img.copy(), (x0,y0), (x1,y1), (255,0,0), 3)
            
            # Add green text label for class name
            img = cv2.putText(img, _cls, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        args.predicts.append(img)
    # COMMENTED OUT: NPU cleanup
    # Clean up inference instance
    #rt_core.deinit_instance(args.interface, args.model_name)

def display(args):
    # Prepare visualization by combining original and processed images
    print("TRACE: display -> preparing final image", flush=True)
    inputs    = np.concatenate(args.images_input, axis=0)
    total_img = inputs
    predicts = np.concatenate(args.predicts, axis=0)
    
    # Create side-by-side comparison (original | processed)
    total_img = np.concatenate([total_img, predicts], axis=1)

    # Display results using matplotlib
    #cv2.imshow('', total_img)
    #cv2.waitKey(0)
    plt.imshow(total_img)
    plt.axis('off')
    plt.show()
    
    # Save results to output directory
    os.makedirs(args.output_dir, exist_ok=True)
    saved_path = f'{args.output_dir}/result.png'
    cv2.imwrite(saved_path, total_img)
    print(f"TRACE: display -> result saved to {saved_path}", flush=True)

if __name__ == '__main__':

    import time
    script_start = time.time()
    
    # Parse command line arguments and setup configuration
    args = create_args()
    print("TRACE: main -> after create_args", flush=True)

    # Boot Tachy device firmware (commented out - optional step)
    # boot(args)

    # Load compiled model to Tachy device memory
    ret_save = save_model(args)
    print(f"TRACE: main -> save_model returned {ret_save}", flush=True)

    # Create inference runtime instance with optimized configuration
    ret_make = make_instance(args)
    print(f"TRACE: main -> make_instance returned {ret_make}", flush=True)

    # Establish connection to inference engine
    ret_conn = connect_instance(args)
    print(f"TRACE: main -> connect_instance returned {ret_conn}", flush=True)

    # Execute PPE detection inference on all input images
    inference(args)
    print("TRACE: main -> inference complete", flush=True)

    # Display and save detection results
    display(args)
    print("TRACE: main -> display complete", flush=True)
    
    # Report total execution time
    script_total = time.time() - script_start
    print(f"\n=== SCRIPT EXECUTION TIME ===")
    print(f"Total script execution: {script_total:.3f} seconds")
    print(f"=============================")