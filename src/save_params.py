import numpy as np
import os
import cv2
from datetime import datetime


def load_points_from_config(anchor_dir, cam_config):
    """
    Load points for a camera based on configuration
    
    Args:
        anchor_dir: Directory with anchor points
        cam_config: Camera configuration dictionary
    
    Returns:
        tuple: (src_points, dst_points)
    """
    # Check if points are provided directly or via file paths
    if 'points_file' in cam_config:
        # Load from file
        points_file = cam_config['points_file']
        src_path = os.path.join(anchor_dir, f"{points_file}_src.npy")
        dst_path = os.path.join(anchor_dir, f"{points_file}_dst.npy")
        
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source points file not found: {src_path}")
        if not os.path.exists(dst_path):
            raise FileNotFoundError(f"Destination points file not found: {dst_path}")
        
        src_points = np.load(src_path)
        dst_points = np.load(dst_path)
        
    elif 'src_points' in cam_config and 'dst_points' in cam_config:
        # Points provided directly in config
        src_points = np.array(cam_config['src_points'])
        dst_points = np.array(cam_config['dst_points'])
        
    else:
        raise ValueError("Camera configuration must contain either 'points_file' or both 'src_points' and 'dst_points'")
    
    return src_points, dst_points

def save_camera_data(camera_dir, camera_name, params, results, homographies, warped_images):
    """
    Save data for a single camera
    
    Args:
        camera_dir: Directory for camera data
        camera_name: Name of the camera
        params: Camera parameters dictionary
        results: Results dictionary (errors, improvements)
        homographies: Homography matrices dictionary
        warped_images: Warped images dictionary
    """
    os.makedirs(camera_dir, exist_ok=True)
    
    # Save camera parameters
    np.save(os.path.join(camera_dir, f'{camera_name}_params.npy'), params)
    with open(os.path.join(camera_dir, f'{camera_name}_params.txt'), 'w') as f:
        f.write(f"{camera_name.upper()} PARAMETERS\n")
        f.write("=" * (len(camera_name) + 12) + "\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    # Save results
    np.save(os.path.join(camera_dir, f'{camera_name}_results.npy'), results)
    with open(os.path.join(camera_dir, f'{camera_name}_results.txt'), 'w') as f:
        f.write(f"{camera_name.upper()} RESULTS\n")
        f.write("=" * (len(camera_name) + 9) + "\n")
        f.write(f"Initial error: {results['initial_error']:.4f} px\n")
        f.write(f"Optimized error: {results['optimized_error']:.4f} px\n")
        f.write(f"Improvement: {results['improvement']:.4f} px ({results['improvement_percent']:.2f}%)\n")
    
    # Save homography matrices
    np.save(os.path.join(camera_dir, f'{camera_name}_homography_initial.npy'), homographies['initial'])
    np.save(os.path.join(camera_dir, f'{camera_name}_homography_optimized.npy'), homographies['optimized'])
    
    # Save images
    cv2.imwrite(os.path.join(camera_dir, f'{camera_name}_initial.jpg'), warped_images['initial'])
    cv2.imwrite(os.path.join(camera_dir, f'{camera_name}_optimized.jpg'), warped_images['optimized'])
    
    print(f"  {camera_name} data saved to {camera_dir}")

def save_experiment_summary(experiment_dir, cameras_data, stitching_results=None):
    """
    Save overall experiment summary
    
    Args:
        experiment_dir: Main experiment directory
        cameras_data: Dictionary with data for all cameras
        stitching_results: Stitching results (optional)
    """
    summary = {
        'experiment_name': cameras_data['experiment_name'],
        'total_cameras': len(cameras_data['cameras']),
        'camera_names': list(cameras_data['cameras'].keys()),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add data for each camera
    for cam_name, cam_data in cameras_data['cameras'].items():
        summary[f'{cam_name}_initial_error'] = cam_data['results']['initial_error']
        summary[f'{cam_name}_optimized_error'] = cam_data['results']['optimized_error']
        summary[f'{cam_name}_improvement'] = cam_data['results']['improvement']
    
    # Add stitching results if available
    if stitching_results:
        summary.update(stitching_results)
    
    # Save to files
    np.save(os.path.join(experiment_dir, 'experiment_summary.npy'), summary)
    
    with open(os.path.join(experiment_dir, 'experiment_summary.txt'), 'w') as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("==================\n")
        f.write(f"Experiment: {summary['experiment_name']}\n")
        f.write(f"Total cameras: {summary['total_cameras']}\n")
        f.write(f"Camera names: {summary['camera_names']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("REPROJECTION ERRORS:\n")
        for cam_name in summary['camera_names']:
            f.write(f"{cam_name}: {summary[f'{cam_name}_initial_error']:.4f} -> {summary[f'{cam_name}_optimized_error']:.4f} px ")
            f.write(f"(improvement: {summary[f'{cam_name}_improvement']:.4f} px)\n")
    
    print(f"Experiment summary saved to {experiment_dir}")

def save_stitching_results(stitching_dir, stitching_data):
    """
    Save stitching results
    
    Args:
        stitching_dir: Stitching results directory
        stitching_data: Dictionary with stitching results
    """
    os.makedirs(stitching_dir, exist_ok=True)
    
    # Save stitching parameters
    np.save(os.path.join(stitching_dir, 'stitching_params.npy'), stitching_data)
    
    # Save images
    for img_name, img_data in stitching_data['images'].items():
        cv2.imwrite(os.path.join(stitching_dir, f'{img_name}.jpg'), img_data)
    
    with open(os.path.join(stitching_dir, 'stitching_results.txt'), 'w') as f:
        f.write("STITCHING RESULTS\n")
        f.write("=================\n")
        f.write(f"Number of images stitched: {stitching_data.get('num_images', 'N/A')}\n")
        f.write(f"Stitching method: {stitching_data.get('method', 'N/A')}\n")
        f.write(f"Generated files: {list(stitching_data['images'].keys())}\n")
    
    print(f"Stitching results saved to {stitching_dir}")

def save_camera_configs(configs_dir, camera_configs):
    """
    Save Optuna configurations for all cameras
    
    Args:
        configs_dir: Directory for configurations
        camera_configs: Dictionary with configurations for each camera
    """
    os.makedirs(configs_dir, exist_ok=True)
    
    for cam_name, config in camera_configs.items():
        np.save(os.path.join(configs_dir, f'{cam_name}_best_config.npy'), config)
        
        with open(os.path.join(configs_dir, f'{cam_name}_best_config.txt'), 'w') as f:
            f.write(f"{cam_name.upper()} BEST CONFIGURATION\n")
            f.write("=" * (len(cam_name) + 21) + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
    
    print(f"Camera configurations saved to {configs_dir}")

def prepare_camera_data(camera_name, point_indices, src_points, dst_points, 
                       initial_error, optimized_error, initial_H, optimized_H, 
                       initial_warp, optimized_warp, square_size):
    """
    Prepare camera data for saving
    
    Args:
        camera_name: Name of the camera
        point_indices: Dictionary with point indices
        src_points: Source points
        dst_points: Destination points
        initial_error: Initial reprojection error
        optimized_error: Optimized reprojection error
        initial_H: Initial homography matrix
        optimized_H: Optimized homography matrix
        initial_warp: Initial warped image
        optimized_warp: Optimized warped image
        square_size: Square size used
    
    Returns:
        Dictionary with prepared camera data
    """
    improvement = initial_error - optimized_error
    improvement_percent = (improvement / initial_error * 100) if initial_error > 0 else 0
    
    params = {
        'camera_name': camera_name,
        'square_size': square_size,
        'point_indices': point_indices,
        'num_points': len(point_indices.get('optimization_points', [])),
        'num_anchors': len(point_indices.get('anchor_points', [])),
        'num_rep_points': len(point_indices.get('reprojection_points', [])),
        'src_points_shape': src_points.shape,
        'dst_points_shape': dst_points.shape
    }
    
    results = {
        'initial_error': initial_error,
        'optimized_error': optimized_error,
        'improvement': improvement,
        'improvement_percent': improvement_percent
    }
    
    homographies = {
        'initial': initial_H,
        'optimized': optimized_H
    }
    
    warped_images = {
        'initial': initial_warp,
        'optimized': optimized_warp
    }
    
    return {
        'params': params,
        'results': results,
        'homographies': homographies,
        'warped_images': warped_images
    }