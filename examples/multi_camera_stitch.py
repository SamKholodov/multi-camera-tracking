import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
import numpy as np
from src.pipeline import *
from src.optuna_search import *
from src.stitching import *
from src.save_params import * 

processor = HomographyProcessor()
root = os.getcwd()

# Camera configuration - easily scalable to multiple cameras
CAMERAS = {
    'camera1': {
        'image_path': os.path.join(root, 'data/calibration_images/img1.jpg'),
        'points_file': 'img1',  # Will load img1_src.npy and img1_dst.npy
        'point_indices': {
            'optimization_points': [0, 4, 37, 40, 21, 4, 18, 23, 36],
            'anchor_points': [0, 4, 37, 40, 21, 4, 18, 23, 36],
            'reprojection_points': [2, 31]
        }
    },
    'camera2': {
        'image_path': os.path.join(root, 'data/calibration_images/img2.jpg'),
        'points_file': 'img2',  # Will load img2_src.npy and img2_dst.npy
        'point_indices': {
            'optimization_points': [0, 3, 35, 38, 20, 1, 14, 19, 31],
            'anchor_points': [0, 3, 35, 38, 20, 1, 14, 19, 31],
            'reprojection_points': [11, 29]
        }
    }
}

EXPERIMENT_NAME = "img1_img2_11"
experiment_dir = os.path.join(root, 'results', EXPERIMENT_NAME)

# Create directory structure
os.makedirs(experiment_dir, exist_ok=True)
cameras_dir = os.path.join(experiment_dir, 'cameras')
stitching_dir = os.path.join(experiment_dir, 'panorama_stitching')
configs_dir = os.path.join(experiment_dir, 'configurations')

os.makedirs(cameras_dir, exist_ok=True)
os.makedirs(stitching_dir, exist_ok=True)
os.makedirs(configs_dir, exist_ok=True)

# Load anchor points directory
anchor_dir = os.path.join(root, 'data/anchor_points')
square_size = 56

# Process each camera
camera_configs = {}
cameras_data = {
    'experiment_name': EXPERIMENT_NAME,
    'cameras': {}
}

# Store homographies, anchor points aand warped images for stitching
homographies_dict = {}
anchor_points_dict = {}
warped_images_dict = {}

for cam_name, cam_config in CAMERAS.items():
    print(f"\n{'='*50}")
    print(f"PROCESSING {cam_name.upper()}")
    print(f"{'='*50}")
    
    # Load image
    img = cv.imread(cam_config['image_path'])
    if img is None:
        print(f"Error: Could not load image {cam_config['image_path']}")
        continue
    
    # Load points for this camera
    try:
        src_all, dst_all = load_points_from_config(anchor_dir, cam_config)
        print(f"Loaded points for {cam_name}: {len(src_all)} source points, {len(dst_all)} destination points")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading points for {cam_name}: {e}")
        continue
    
    # Extract points using indices
    idxs = cam_config['point_indices']
    src_points = src_all[idxs['optimization_points']]
    dst_points = dst_all[idxs['optimization_points']]
    anchor_src = src_all[idxs['anchor_points']]
    anchor_dst = dst_all[idxs['anchor_points']]
    rep_src = src_all[idxs['reprojection_points']]
    rep_dst = dst_all[idxs['reprojection_points']]
    
    # Store anchor points for stitching
    anchor_points_dict[cam_name] = anchor_src
    
    # Run optimization
    results = test_multiple_methods(
        processor, 
        cam_config['image_path'], 
        src_points, 
        dst_points,
        rep_src, 
        rep_dst,
        anchor_src=anchor_src,
        anchor_dst=anchor_dst,
        square_size=square_size,
        verbose=False,
        use_optuna=True, 
        n_trials=100      
    )
    
    _, _, best_config = results_summary(results, f'{cam_name} - Optuna Optimized')
    camera_configs[cam_name] = best_config
    
    # Process image
    initial_H, optimized_H, squares, initial_warp, optimized_warp = processor.process_image(
        img_path=cam_config['image_path'],
        src_points=src_points,
        dst_points=dst_points,
        anchor_points=anchor_src,
        anchor_targets=anchor_dst,
        weight=best_config['weights'],
        method=best_config['methods'],
        bound_mode=best_config['bounds_modes'],
        output_size=(1600, 1200),
        square_size=square_size,
        square_area=best_config['square_area'],
        verbose=True,
        optimization_levels=best_config['optimization_levels'])
    
    # Store homographies and warped images for stitching
    homographies_dict[cam_name] = {
        'initial': initial_H,
        'optimized': optimized_H
    }
    
    warped_images_dict[cam_name] = {
        'initial': initial_warp,
        'optimized': optimized_warp
    }
    
    # Calculate reprojection errors
    mask = np.ones(len(src_all), dtype=bool)
    mask[idxs['optimization_points']] = False
    rep_img = src_all[mask]
    rep_dst = dst_all[mask]
    
    initial_error, _, _ = GeometryUtils.calculate_reprojection_error(initial_H, rep_img, rep_dst)
    optimized_error, _, _ = GeometryUtils.calculate_reprojection_error(optimized_H, rep_img, rep_dst)
    
    print(f'{cam_name} - init = {initial_error:.2f}, opt = {optimized_error:.2f}')
    
    # Prepare camera data for saving
    camera_data = prepare_camera_data(
        cam_name, idxs, src_points, dst_points,
        initial_error, optimized_error, initial_H, optimized_H,
        initial_warp, optimized_warp, square_size
    )
    
    # Save camera data
    camera_dir = os.path.join(cameras_dir, cam_name)
    save_camera_data(
        camera_dir, cam_name,
        camera_data['params'],
        camera_data['results'],
        camera_data['homographies'],
        camera_data['warped_images']
    )
    
    cameras_data['cameras'][cam_name] = camera_data

# Save configurations
save_camera_configs(configs_dir, camera_configs)

# Stitching (for multiple cameras)
if len(CAMERAS) >= 2:
    print(f"\n{'='*50}")
    print("STITCHING IMAGES")
    print(f"{'='*50}")
    
    # Get warped images
    initial_warp1 = warped_images_dict['camera1']['initial']
    optimized_warp1 = warped_images_dict['camera1']['optimized']
    initial_warp2 = warped_images_dict['camera2']['initial']
    optimized_warp2 = warped_images_dict['camera2']['optimized']
    
    # Get homographies
    optimized_H1 = homographies_dict['camera1']['optimized']
    optimized_H2 = homographies_dict['camera2']['optimized']
    
    # Get anchor points
    anch_src1 = anchor_points_dict['camera1']
    anch_src2 = anchor_points_dict['camera2']
    
    # Perform all stitching operations
    stitched_initial = stitch_with_overlap_detection(initial_warp1, initial_warp2)
    cv.imwrite(os.path.join(stitching_dir, 'stitched_initial.jpg'), stitched_initial)

    stitched_optimized = stitch_with_overlap_detection(optimized_warp1, optimized_warp2)
    cv.imwrite(os.path.join(stitching_dir, 'stitched_optimized.jpg'), stitched_optimized)

    blended_initial = advanced_blend_images(initial_warp1, initial_warp2, alpha=0.5)
    cv.imwrite(os.path.join(stitching_dir, 'blended_initial.jpg'), blended_initial)

    blended_optimized = advanced_blend_images(optimized_warp1, optimized_warp2, alpha=0.5)
    cv.imwrite(os.path.join(stitching_dir, 'blended_optimized.jpg'), blended_optimized)

    seam_mask = find_optimal_seam_mask(optimized_warp1, optimized_warp2)
    cv.imwrite(os.path.join(stitching_dir, 'seam_mask.jpg'), seam_mask * 255)

    stitched_optimal_seam = stitch_with_optimal_seam(optimized_warp1, optimized_warp2, blend_width=20)
    cv.imwrite(os.path.join(stitching_dir, 'stitched_optimal_seam.jpg'), stitched_optimal_seam)

    stitched_multiband = stitch_with_multiband_blending(optimized_warp1, optimized_warp2, levels=5)
    cv.imwrite(os.path.join(stitching_dir, 'stitched_multiband.jpg'), stitched_multiband)

    # Create stitched image with anchor points
    stitched_with_points = stitched_optimized.copy()

    anchor_points1_transformed = cv.perspectiveTransform(
        anch_src1.reshape(-1, 1, 2).astype(np.float32), 
        optimized_H1
    ).reshape(-1, 2)

    anchor_points2_transformed = cv.perspectiveTransform(
        anch_src2.reshape(-1, 1, 2).astype(np.float32),
        optimized_H2
    ).reshape(-1, 2)

    # Draw points from first camera (blue)
    for i, point in enumerate(anchor_points1_transformed):
        point_int = tuple(point.astype(int))
        cv.circle(stitched_with_points, point_int, 8, (255, 0, 0), -1)  # Blue
        cv.putText(stitched_with_points, f'L{i+1}', 
                  (point_int[0] + 10, point_int[1] - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw points from second camera (green)
    for i, point in enumerate(anchor_points2_transformed):
        point_int = tuple(point.astype(int))
        cv.circle(stitched_with_points, point_int, 8, (0, 255, 0), -1)  # Green
        cv.putText(stitched_with_points, f'R{i+1}', 
                  (point_int[0] + 10, point_int[1] - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.imwrite(os.path.join(stitching_dir, 'stitched_with_anchor_points.jpg'), stitched_with_points)

    # Save stitching summary
    stitching_data = {
        'num_images': 2,
        'method': 'advanced_stitching',
        'generated_files': [
            'stitched_initial.jpg',
            'stitched_optimized.jpg', 
            'blended_initial.jpg',
            'blended_optimized.jpg',
            'seam_mask.jpg',
            'stitched_with_anchor_points.jpg'
        ]
    }
    
    np.save(os.path.join(stitching_dir, 'stitching_summary.npy'), stitching_data)
    
    with open(os.path.join(stitching_dir, 'stitching_summary.txt'), 'w') as f:
        f.write("STITCHING SUMMARY\n")
        f.write("=================\n")
        f.write(f"Number of images: {stitching_data['num_images']}\n")
        f.write(f"Method: {stitching_data['method']}\n")
        f.write("Generated files:\n")
        for file in stitching_data['generated_files']:
            f.write(f"  - {file}\n")
    
    print("All stitching images saved successfully!")

# Save experiment summary
save_experiment_summary(experiment_dir, cameras_data)

print(f"\n{'='*50}")
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print(f"{'='*50}")
print(f"All data saved to: {experiment_dir}")
print(f"Processed {len(CAMERAS)} cameras: {list(CAMERAS.keys())}")
print(f"Stitching results: {len(os.listdir(stitching_dir))} files in {stitching_dir}")