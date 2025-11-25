import os 
import cv2 as cv
import numpy as np
from src.pipeline import *
import numpy as np

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

method = ['Powell', 'Powell', 'Powell']
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