# PPE Detection Model Configuration Guide

This guide explains how to adapt the PPE detection system when using a new `.tachyrt` model file with different image dimensions or parameters.

## Overview

When you receive a new trained model (`.tachyrt` file), you typically need to update several configuration files and parameters to match the new model's specifications.

## Files That Need Updates

### 1. **Post-Processing Configuration** (`post_process_256x416.json`)
**Location**: `./20250728_085213/post_process_256x416.json` (or your model folder)

This is the **most critical file** to update when changing models.

#### Key Parameters:

```json
{
    "SHAPES_GRID": [[39, 28]],        // Model output grid dimensions
    "SHAPES_OUTPUT": [[39, 28, 16]],  // Full output tensor shape
    "SHAPES_INPUT": [256, 416],       // Input image dimensions (H, W)
    "OBJ_THRESHOLD": 0.9,             // Confidence threshold (0.0-1.0)
    "NMS_THRESHOLD": 0.2,             // Non-maximum suppression overlap
    "PRE_THRESHOLD": 0.8,             // Early filtering threshold
    "N_BOX_LOGIT": 4,                 // Box coordinates (always 4)
    "N_CLASSES": 4,                   // Number of detection classes
    "N_MAX_OBJ": 5                    // Maximum detections per image
}
```

#### How to Calculate Grid Dimensions:

**Step 1**: Run the model once to get the output tensor size
```bash
# Look for this output in terminal:
Model output shape: (17472,)  # Total elements
```

**Step 2**: Calculate grid dimensions
```
Total elements = 17,472
Elements per cell = Total / Grid cells
Grid cells = Height × Width

# For our example:
17,472 ÷ 16 = 1,092 grid cells
39 × 28 = 1,092 ✓
```

**Step 3**: Update configuration
```json
"SHAPES_GRID": [[39, 28]],
"SHAPES_OUTPUT": [[39, 28, 16]],
"SHAPES_INPUT": [256, 416]  // Your new input dimensions
```

### 2. **Main Script** (`J_30_inference_running_ppe.py`)
**Location**: `./J_30_inference_running_ppe.py`

#### Parameters to Update:

**Line ~78**: Input dimensions
```python
args.h, args.w = 256, 416  # Change to your new dimensions
```

**Command line arguments** when running:
```bash
python J_30_inference_running_ppe.py \
    --model your_new_model_name \
    --input_shape 256x416x3 \                    # New dimensions
    --model_path ./path/to/new_model.tachyrt \   # New model file
    --class_json ./path/to/class.json \          # Class definitions
    --post_config ./path/to/post_process_config.json
```

### 3. **Class Definitions** (`class.json`)
**Location**: `./20250728_085213/class.json` (or your model folder)

Update if your new model has different classes:

```json
{
    "0": "class_name_1",
    "1": "class_name_2",
    "2": "class_name_3",
    "3": "class_name_4"
}
```

**Update N_CLASSES** in post_process config to match the number of classes.

### 4. **Post-Processing Code** (`post_process.py`)
**Location**: `./20250728_085213/post_process.py`

#### Default Grid Size (Line ~71):
```python
grid_sizes = np.asarray(configs['SHAPES_GRID'], dtype='float32') if 'SHAPES_GRID' in configs else np.array([
    [39,28],  # Update to match your new grid dimensions
], dtype='float32')
```

## Step-by-Step Adaptation Process

### When You Get a New Model:

1. **Update Input Dimensions**
   - Edit `J_30_inference_running_ppe.py` line ~78
   - Update `SHAPES_INPUT` in post_process config

2. **Determine Output Grid Size**
   - Run the script once to see tensor size
   - Calculate grid dimensions using the formula above
   - Update `SHAPES_GRID` and `SHAPES_OUTPUT`

3. **Update Class Information**
   - Modify `class.json` if classes changed
   - Update `N_CLASSES` in post_process config

4. **Tune Detection Quality**
   - Adjust `OBJ_THRESHOLD` (0.5-0.9) for detection sensitivity
   - Modify `N_MAX_OBJ` (1-20) for max detections per image
   - Tweak `NMS_THRESHOLD` (0.1-0.5) for overlap removal

## Common Issues and Solutions

### Issue 1: "cannot reshape array" Error
**Cause**: Grid dimensions don't match model output
**Solution**: Recalculate `SHAPES_GRID` and `SHAPES_OUTPUT` based on actual tensor size

### Issue 2: Too Many/Few Detections
**Cause**: Threshold settings
**Solution**: Adjust these parameters in post_process config:
- `OBJ_THRESHOLD`: Higher = fewer detections
- `N_MAX_OBJ`: Direct limit on detection count
- `NMS_THRESHOLD`: Lower = more overlap removal

### Issue 3: "KeyError" for Class ID
**Cause**: Model predicting classes not in class.json
**Solution**: Add bounds checking or update class definitions

## Example: 320x320 Model Adaptation

If you receive a 320x320 input model:

1. **Update input dimensions**:
   ```python
   args.h, args.w = 320, 320
   ```

2. **Update config**:
   ```json
   "SHAPES_INPUT": [320, 320]
   ```

3. **Determine new grid** (example):
   ```
   If output is 20,480 elements:
   20,480 ÷ 16 = 1,280 grid cells
   40 × 32 = 1,280
   ```

4. **Update grid config**:
   ```json
   "SHAPES_GRID": [[40, 32]],
   "SHAPES_OUTPUT": [[40, 32, 16]]
   ```

## Quick Reference: File Locations

| Parameter | File | Line | Description |
|-----------|------|------|-------------|
| Input dimensions | `J_30_inference_running_ppe.py` | ~78 | `args.h, args.w` |
| Grid size | `post_process_config.json` | 2-3 | `SHAPES_GRID`, `SHAPES_OUTPUT` |
| Input shape | `post_process_config.json` | 4 | `SHAPES_INPUT` |
| Class count | `post_process_config.json` | 9 | `N_CLASSES` |
| Detection limits | `post_process_config.json` | 5-7,10 | Thresholds and `N_MAX_OBJ` |
| Class names | `class.json` | All | Class ID to name mapping |
| Default grid | `post_process.py` | ~71 | Fallback grid dimensions |

## Testing New Configuration

1. Run with debug output first
2. Check tensor shapes match expectations
3. Verify detection quality and adjust thresholds
4. Test on multiple images before production use

---

**Note**: Always backup working configurations before making changes!
