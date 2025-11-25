# Multi-Camera Vehicle Tracking with Homography Transformation
A computer vision system for multi-camera vehicle tracking and parking analysis using homography transformation to convert camera coordinates to global world coordinates.

## Overview 
This project implements a robust homography-based approach for transforming local camera views into a unified global coordinate system, enabling vehicle tracking across multiple cameras. The system is based on the method described in paper "Precise Top View Image Generation without Global Metric Information" by Kano et al. (2008), with enhancements for global coordinate alignment and multi-level optimization.

## Key features
- **Automatic Square Detection**: Detects calibration patterns in camera views
- **Multi-Level Optimization**: Uses flexible optimization methods with adaptive bounds
- **Homography Optimization**: Implements Kano's algorithm with enhancements for global coordinate mapping
- **Anchor Point Constraints**: Incorporates known reference points for precise global positioning
- **Multi-Camera Support**: Processes and stitches multiple camera views
- **Optuna Integration**: Automatic hyperparameter optimization
- **Reprojection Error Visualization**: Comprehensive error analysis and visualization tools

## Project Structure
```text
project/
├── src/
│   ├── pipeline.py          # Core processing classes
│   ├── optuna_search.py     # Hyperparameter optimization
│   ├── stitching.py         # Image stitching functions
│   └── save_params.py       # Data saving utilities
├── data/
│   ├── calibration_images/  # Source images
│   └── anchor_points/       # Calibration points
├── examples/
│   ├── basic_usage.py       # Single camera example
│   └── multi_camera_stitch.py # Multi-camera example
└── results/
    └── EXPERIMENT_NAME/
        ├── cameras/         # Data for each camera
        ├── panorama_stitching/ # Stitching results
        └── configurations/  # Optuna configurations
```

## Installation 
Clone the repository:
```bash
git clone https://github.com/SamKholodov/multi-camera-tracking.git
cd multi-camera-tracking

Install required dependencies
pip install opencv-python numpy scipy matplotlib optuna
```
## Usage   
Step-by-Step Calibration Process:
 - Prepare Calibration Patterns: Place square patterns in camera's field of view
- Define Coordinate Systems: Map camera points to world coordinates
- Run Optimization: Use multi-level homography optimization
- Validate Results: Check reprojection errors and visual alignment

## Quick start 

### Basic Homography Calculation:

```python
import os 
import cv2 as cv
import numpy as np
from src.pipeline import *

processor = HomographyProcessor()
root = os.getcwd()
imgPath = os.path.join(root, 'data/calibration_images/img.jpg')

img = cv.imread(imgPath)
imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

EXPERIMENT_NAME = "img"
results_dir = os.path.join(root, f'results/{EXPERIMENT_NAME}')
os.makedirs(results_dir, exist_ok=True)

src = np.array([[x1, y1], [x2, y2], ...])  # Camera coordinates
dst = np.array([[X1, Y1], [X2, Y2], ...])  # World coordinates
anch_src = np.array([[x11, y11], [x22, y22], ...])  # Camera coordinates
anch_dst = np.array([[X11, Y11], [X22, Y22], ...])  # World coordinates

method = ['Powell', 'Powell', 'Powell'] # Powell, L-BFGS-B or TNC
weight = [0.9, 0.5, 0.4]
bound_mode = ['adaptive', 'strict', 'none']
square_size = 100 # Real square size
square_area = 2000 # Min square area in pixels

initial_H, optimized_H, squares, initial_warp, optimized_warp = processor.process_image(
    img_path=imgPath,
    src_points=src,
    dst_points=dst,
    anchor_points=anch_src,
    anchor_targets=anch_dst,
    weight=weight,
    method=method,
    bound_mode=bound_mode,
    output_size=(1600, 1200),
    square_size=square_size,
    square_area=square_area,
    verbose=False,
    optimization_levels=3)
    
cv.imwrite(os.path.join(root, 'homography/results/initial_homo.jpg'), initial_warp)
cv.imwrite(os.path.join(root, 'homography/results/improved_homo.jpg'), optimized_warp)
```