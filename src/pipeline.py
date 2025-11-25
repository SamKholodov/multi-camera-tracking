import os 
import cv2 as cv
import numpy as np
from scipy.optimize import minimize

class SquareDetector:
    """Class for detecting and processing squares in images"""    
    @staticmethod
    def sort_corners(corners):
        """
        Sort quadrilateral corners to [top-left, top-right, bottom-right, bottom-left].
        
        Args:
            corners: numpy array of shape (4, 2) with corner coordinates
            
        Returns:
            numpy array of shape (4, 2) with sorted corners
        """
        corners = corners.reshape(4, 2)
        
        center = np.mean(corners, axis=0)
        
        left = corners[corners[:, 0] < center[0]]
        right = corners[corners[:, 0] >= center[0]]
        
        left = left[left[:, 1].argsort()]
        right = right[right[:, 1].argsort()]
        
        if len(left) == 2 and len(right) == 2:
            return np.array([left[0], right[0], right[1], left[1]])
        else:
            vectors = corners - center
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            return corners[np.argsort(angles)]
    
    def detect_squares(
            self, 
            img_path, 
            square_area=2000, 
            save_picture=True, 
            file_path=None):
        """
        Detect and refine square contours in an image.
        
        Args:
            img_path: Path to input image
            square_area: Minimum area for square detection
            save_picture: Whether to save visualization
            file_path: Path for saving visualization
            
        Returns:
            numpy.array: Refined square corners as array of shape (N, 4, 2) where 
                        N is number of squares, each with 4 corners (x, y)
        """
        img = cv.imread(img_path)
        assert img is not None, "file could not be read, check with os.path.exists()"

        imgOrig = img.copy()
        imgGray = cv.cvtColor(imgOrig, cv.COLOR_BGR2GRAY)
        imgBlur = cv.GaussianBlur(imgGray, (5,5), 0)

        ret, thresh = cv.threshold(imgBlur, 140, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        squares = []
        for i, cnt in enumerate(contours):
            if i == 0:
                continue
            if hierarchy[0][i][3] == -1:  # skipping ext contour
                continue

            epsilon = 0.02 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv.isContourConvex(approx) and cv.contourArea(approx) > square_area:
                x, y, w, h = cv.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.6 <= aspect_ratio <= 3:
                    sorted_corners = self.sort_corners(approx)
                    squares.append(sorted_corners)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_squares = []
        for square_idx, square in enumerate(squares):
            gray_points = np.array(square, dtype=np.float32)
            refined_corners = cv.cornerSubPix(imgGray, gray_points, (5,5), (-1,-1), criteria)
            refined_squares.append(refined_corners)
            cv.drawContours(imgOrig, [refined_corners.astype(int)], -1, (0, 255, 0), 3)
            for corner_idx, corner in enumerate(refined_corners):
                x, y = corner.astype(int)
                
                cv.circle(imgOrig, (x, y), 8, (255, 0, 0), -1)
                
                label = f"p{corner_idx + 1}"
                cv.putText(imgOrig, label, (x + 10, y - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv.putText(imgOrig, f"S{square_idx + 1}", 
                          (int(np.mean(refined_corners[:, 0])), 
                           int(np.mean(refined_corners[:, 1]))), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        if save_picture:
            if file_path is not None:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            else:
                root = os.getcwd() 
                file_path = os.path.join(root, 'homography_pipeline/results/detected.jpg')
                os.makedirs(os.path.dirname(file_path), exist_ok=True)     
            cv.imwrite(file_path, imgOrig)

        return np.array([square.reshape(4, 2) for square in refined_squares])


class GeometryUtils:
    """Class of geometry utils"""

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point as numpy array [x, y]
            point2: Second point as numpy array [x, y]
            
        Returns:
            float: Euclidean distance between the points
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    @staticmethod
    def angle_between_vectors(v1, v2):
        """
        Calculate angle between two vectors in degrees.
        
        Args:
            v1: First vector as numpy array
            v2: Second vector as numpy array
            
        Returns:
            float: Angle between vectors in degrees
        """
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        angle = np.degrees(np.arccos(cos_angle))
        return angle

    @staticmethod
    def calculate_reprojection_error(H, src_points, dst_points):
        """
        Calculate reprojection error between source and destination points.
        
        Args:
            H: Homography matrix (3x3)
            src_points: Source points in image coordinates (N, 2)
            dst_points: Destination points in world coordinates (N, 2)
            
        Returns:
            tuple: (mean_error, all_errors, reprojected_points)
        """
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        src_homogeneous = np.column_stack([src_points, np.ones(len(src_points))])
        transformed = (H @ src_homogeneous.T).T
        reprojected_points = transformed[:, :2] / transformed[:, 2:3]
        
        errors = np.linalg.norm(reprojected_points - dst_points, axis=1)
        mean_error = np.mean(errors)
        
        return mean_error, errors, reprojected_points
    
    @staticmethod
    def get_coords_in_target_plane(H, src_points):
        """
        Calculate coordinates of points in target plane using homography matrix.
        
        Args:
            H: Homography matrix (3x3)
            src_points: Source points in image coordinates (N, 2)
            
        Returns:
            numpy.array: Reprojected points in target coordinates (N, 2)
        """ 
        src_points = np.array(src_points, dtype=np.float32)

        src_homogeneous = np.column_stack([src_points, np.ones(len(src_points))])
        transformed = (H @ src_homogeneous.T).T
        reprojected_points = transformed[:, :2] / transformed[:, 2:3]

        return reprojected_points
    
    @staticmethod
    def normalize_quadrilateral(quad):
        """
        Normalizes a quadrilateral: shifts to origin and aligns with X-axis.
        
        Args:
            quad: numpy array of shape (4, 2) containing quadrilateral vertex coordinates
            
        Returns:
            rotated: normalized quadrilateral after translation and rotation
            shifted: quadrilateral after translation only
            angle: rotation angle applied (in radians)
        """
        # Find the point with minimum Y-coordinate (and minimum X if there are ties)
        start_idx = np.lexsort((quad[:, 0], quad[:, 1]))[0]
        
        # Reorder points to start from the bottom-most point
        quad_rolled = np.roll(quad, -start_idx, axis=0)
        
        # Translate so the first point becomes origin (0, 0)
        shifted = quad_rolled - quad_rolled[0]
        
        # Calculate rotation angle to align the first side with X-axis
        dx, dy = shifted[1]  # Vector from point 0 to point 1
        angle = np.arctan2(dy, dx)
        
        # Create rotation matrix for alignment
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation_matrix = np.array([[cos_a, -sin_a],
                                    [sin_a, cos_a]])
        
        # Apply rotation to all points
        rotated = shifted @ rotation_matrix.T
        
        return rotated
class HomographyOptimizer:
    """Class for calculating and optimizing homography matrix"""
    
    def __init__(self):
        self.geometry_utils = GeometryUtils()
    
    @staticmethod
    def find_initial_homography(src_points, dst_points):
        """Find initial homography"""
        H, _ = cv.findHomography(src_points, dst_points, method=cv.RANSAC)
        return H
    
    def calculate_dn_error(self, H, square, square_size=100):
        """
        Calculate transformation error between detected square and ideal square.
        
        Args:
            H: Homography matrix (3x3)
            square: Detected square corners (4x2)
            square_size: Size of ideal square in pixels
            
        Returns:
            float: Combined error metric
        """
        ideal_square = np.array([
            [0, square_size],
            [square_size, square_size],
            [square_size, 0],
            [0, 0]
        ])

        square_transformed = cv.perspectiveTransform(square.reshape(1, -1, 2), 
                                                    H).reshape(-1, 2)
    
        new_square = self.geometry_utils.normalize_quadrilateral(square_transformed)

        d1 = self.geometry_utils.euclidean_distance(new_square[1], ideal_square[1])
        d2 = self.geometry_utils.euclidean_distance(new_square[2], ideal_square[2])  # Position error
        d3 = self.geometry_utils.euclidean_distance(new_square[0], ideal_square[0])  # Position error

        error = d1 + d2 + d3
        #error = d1 ** 2 + d2 ** 2 + d3 ** 2 
        return error
    
    def calculate_total_error(self, H_params, squares):
        """
        Calculate total transformation error across all squares.
        
        Args:
            H_params: Homography parameters as flattened array (8,) or matrix (3,3)
            squares: List of detected square corners
            
        Returns:
            float: Sum of transformation errors for all squares
        """
        H = H_params.reshape(3, 3)
        total_error = 0 
        for square in squares:
            total_error += self.calculate_dn_error(H, square)
        return total_error
    
    @staticmethod
    def homography_from_params(params):
        """
        Construct 3x3 homography matrix from 8 parameters.
        
        Args:
            params: Array of 8 parameters [H00, H01, H02, H10, H11, H12, H20, H21]
            
        Returns:
            numpy array: 3x3 homography matrix
        """
        H = np.eye(3)
        H[0, 0] = params[0]
        H[0, 1] = params[1]
        H[0, 2] = params[2]
        H[1, 0] = params[3]
        H[1, 1] = params[4]
        H[1, 2] = params[5]
        H[2, 0] = params[6]
        H[2, 1] = params[7]
        H[2, 2] = 1.0  # fixed
        return H
    
    @staticmethod
    def params_from_homography(H):
        """
        Extract 8 parameters from 3x3 homography matrix.
        
        Args:
            H: 3x3 homography matrix
            
        Returns:
            numpy array: 8 parameters [H00, H01, H02, H10, H11, H12, H20, H21]
        """
        params = np.zeros(8)
        params[0] = H[0, 0]
        params[1] = H[0, 1]
        params[2] = H[0, 2]
        params[3] = H[1, 0]
        params[4] = H[1, 1]
        params[5] = H[1, 2]
        params[6] = H[2, 0]
        params[7] = H[2, 1]
        return params
    
    def optimize_homography(
            self, 
            initial_homography, 
            squares, 
            anchor_points=None, 
            anchor_targets=None, 
            weight=0.3, 
            bounds_mode="adaptive", 
            method='Powell', 
            square_size=100,
            verbose=False):
        """    
        Optimize homography with both geometric and anchor constraints
        
        Args:
            initial_homography: Initial 3x3 homography matrix
            squares: List of detected square corners for geometric optimization
            anchor_points: Anchor points in source image (N, 2)
            anchor_targets: Target positions in destination coordinates (N, 2)
            weight: Weight for anchor error vs geometric error
            bounds_mode: Bounds constraint mode - "none" (no bounds), "strict" (tight bounds), 
                    "adaptive" (medium bounds based on initial estimate)
            method: Optimization method ('Powell', 'BFGS', 'L-BFGS-B')
            square_size: Square size of real square in pixels
            verbose: Display optimization process
            
        Returns:
            numpy.array: Optimized 3x3 homography matrix
        """
        def objective_function(params, squares, anchor_points=None, anchor_targets=None, weight=0.3, square_size=100):
            H = self.homography_from_params(params)

            # 1. Geometric error from squares
            geometric_error = 0
            for i, square in enumerate(squares):
                geometric_error += self.calculate_dn_error(H, square, square_size=square_size)     

            # 2. Anchor error
            anchor_error = 0      
            if anchor_points is not None and anchor_targets is not None:
                for point, target in zip(anchor_points, anchor_targets):
                    point_homogeneous = np.array([point[0], point[1], 1.0])
                    transformed_point = H @ point_homogeneous
                    transformed_point_cartesian = transformed_point[:2] / transformed_point[2]
                    position_error = np.linalg.norm(transformed_point_cartesian - target)
                    anchor_error += position_error
            
            return geometric_error + weight * anchor_error
        
        initial_params = self.params_from_homography(initial_homography)
        
        if bounds_mode == "none":
            bounds = None
        
        elif bounds_mode == "strict":
            bounds = [
                (initial_params[0] * 0.9, initial_params[0] * 1.1),
                (initial_params[1] - 0.1, initial_params[1] + 0.1),
                (initial_params[2] - 50, initial_params[2] + 50),
                (initial_params[3] - 0.1, initial_params[3] + 0.1),
                (initial_params[4] * 0.9, initial_params[4] * 1.1),
                (initial_params[5] - 50, initial_params[5] + 50),
                (-0.001, 0.001),
                (-0.001, 0.001)
            ]

        elif bounds_mode == "adaptive":
            bounds = [
                (initial_params[0] * 0.7, initial_params[0] * 1.3),
                (initial_params[1] - 0.3, initial_params[1] + 0.3),
                (initial_params[2] - 150, initial_params[2] + 150),
                (initial_params[3] - 0.3, initial_params[3] + 0.3),
                (initial_params[4] * 0.7, initial_params[4] * 1.3),
                (initial_params[5] - 150, initial_params[5] + 150),
                (-0.01, 0.01),
                (-0.01, 0.01)
            ]
        
        else:
            raise ValueError("bounds_mode must be 'none', 'strict', or 'adaptive'")
        
        if bounds is None:
            result = minimize(
                objective_function,
                initial_params,
                args=(squares, anchor_points, anchor_targets, weight, square_size),
                method=method,
                options={'disp': verbose}
            )
        else:
            result = minimize(
                objective_function,
                initial_params,
                args=(squares, anchor_points, anchor_targets, weight, square_size),
                method=method,
                bounds=bounds,
                options={'disp': verbose}
            )
        
        optimized_H = self.homography_from_params(result.x)
        return optimized_H
    
    def multi_level_optimization(self,
                                initial_H, 
                                squares,
                                levels=3, 
                                anchor_points=None,
                                anchor_targets=None, 
                                bounds_modes=None,
                                methods=None,
                                square_size=100,
                                weights=None,
                                verbose=False):
        """
        Multi-level homography optimization with coarse-to-fine approach.
        
        Args:
            initial_H: Starting homography matrix (3x3)
            squares: Detected squares data
            levels: Number of optimization levels
            anchor_points: Source constraint points (N, 2)
            anchor_targets: Target constraint points (N, 2)
            bounds_modes: Bounds mode per level: 'adaptive', 'strict', 'none'
            methods: Optimization method per level: 'Powell', 'BFGS', 'L-BFGS-B'
            square_size: Physical square size in mm
            weights: Cost function weight per level (higher = more accuracy)
            verbose: Print debug info
            
        Returns:
            Optimized homography matrix (3x3)
        """

        
        current_H = initial_H.copy()
        
        if weights is None:
            weights = [0.9, 0.5, 0.4][:levels]
            
        if bounds_modes is None:
            bounds_modes = ['adaptive', 'strict', 'none'][:levels]
        elif isinstance(bounds_modes, str):
            bounds_modes = [bounds_modes] * levels
        
        if methods is None:
            methods = ['Powell', 'Powell', 'Powell'][:levels]
        elif isinstance(methods, str):
            methods = [methods] * levels 
        

        
        if len(bounds_modes) < levels:
            bounds_modes = bounds_modes + [bounds_modes[-1]] * (levels - len(bounds_modes))
        if len(methods) < levels:
            methods = methods + [methods[-1]] * (levels - len(methods))
        if len(weights) < levels:
            weights = weights + [weights[-1]] * (levels - len(weights))

        

        
        for level in range(levels):
            bound = bounds_modes[level]
            method = methods[level]
            current_weight = weights[level]  
            
            
            current_H = self.optimize_homography(
                initial_homography=current_H,
                squares=squares,
                anchor_points=anchor_points,
                anchor_targets=anchor_targets,
                bounds_mode=bound,
                method=method,
                weight=current_weight,
                square_size=square_size,
                verbose=verbose
            )
        
        return current_H
class ImageTransformer:
    """Class for image transformations"""
    
    @staticmethod
    def rotate_and_translate(
        img, 
        angle, 
        dx, 
        dy, 
        scale=1, 
        shape=(800,600)):
        """
        Apply combined rotation, translation and scaling transformation to image.
        
        Args:
            img: Input image
            angle: Rotation angle in degrees
            dx: Translation in x-direction
            dy: Translation in y-direction
            scale: Scaling factor
            shape: Output image shape (width, height)
            
        Returns:
            numpy array: Transformed image
        """
        h, w = img.shape[:2]
        M_rotate = cv.getRotationMatrix2D((h // 2, w // 2), angle, scale) 
        M_rotate = np.vstack([M_rotate, [0, 0, 1]]) 
        M_translate = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

        M_combined = M_translate @ M_rotate
        imgFinal = cv.warpPerspective(img, M_combined, shape)
        return imgFinal
    
    @staticmethod
    def apply_homography(img, H, output_size=None):
        """
        Apply homography transformation to image.
        
        Args:
            img: Input image
            H: Homography matrix (3x3)
            output_size: Output image size (width, height)
            
        Returns:
            numpy array: Transformed image
        """
        if output_size is None:
            output_size = (img.shape[1], img.shape[0])
        return cv.warpPerspective(img, H, output_size)

class HomographyProcessor():
    """Main class for homography processing"""
    
    def __init__(self):
        self.detector = SquareDetector()
        self.optimizer = HomographyOptimizer()
        self.transformer = ImageTransformer()
        self.geometry_utils = GeometryUtils()
    
    def process_image(
            self, 
            img_path, 
            square_area=4000, 
            squares_path=None, 
            src_points=None, 
            dst_points=None, 
            anchor_points=None, 
            square_size = 100,
            anchor_targets=None, 
            output_size=(800, 700), 
            weight=0.3, 
            bound_mode='adaptive', 
            method='Powell',
            verbose=False,
            optimization_levels=3):
        """
        Complete image processing pipeline: detect squares, compute and optimize homography.
        
        Args:
            img_path: Path to input image
            square_area: Minimum area for square detection
            squares_path: Path to save squares data
            src_points: Source points for initial homography
            dst_points: Target points for initial homography  
            anchor_points: Constraint points for optimization
            anchor_targets: Target positions for constraint points
            square_size: Physical size of squares
            output_size: Size of output warped image (width, height)
            weight: Optimization cost function weight (anchor vs geometric error balance)
            bound_mode: Bounds mode for optimization
            method: Optimization method
            verbose: Print debug information
            
        Returns:
            tuple: (initial_H, optimized_H, squares, initial_warp, optimized_warp)
                initial_H: Initial homography matrix (3x3)
                optimized_H: Optimized homography matrix (3x3) 
                squares: Detected squares as array of shape (N, 4, 2)
                initial_warp: Image warped with initial homography
                optimized_warp: Image warped with optimized homography
        """
        # Square detection
        squares = self.detector.detect_squares(img_path, square_area=square_area, file_path=squares_path)
        
        # Load image for visualization
        img = cv.imread(img_path)
        
        
        # Compute initial homography
        if src_points is not None and dst_points is not None:
            initial_H = self.optimizer.find_initial_homography(src_points, dst_points)
            if anchor_points is None and anchor_targets is None:
                anchor_points = src_points
                anchor_targets = dst_points
        else:
            # Use the first square as default
            if len(squares) > 0:
                target_square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
                initial_H = self.optimizer.find_initial_homography(squares[0], target_square)
            else:
                raise ValueError("No squares detected and no anchor points provided")
        
        # Apply initial homography
        initial_warp = self.transformer.apply_homography(img, initial_H, output_size)
        
        # Optimize homography
        if optimization_levels == 1:
            optimized_H = self.optimizer.optimize_homography(
                initial_homography=initial_H,
                squares=squares,
                anchor_points=anchor_points,
                anchor_targets=anchor_targets,
                weight=weight,
                bounds_mode=bound_mode,
                method=method,
                square_size=square_size,
                verbose=verbose
            )
        else:
            optimized_H = self.optimizer.multi_level_optimization(
                initial_H=initial_H,
                squares=squares,
                anchor_points=anchor_points,
                anchor_targets=anchor_targets,
                weights=[weight] * optimization_levels if isinstance(weight, (int, float)) else weight,
                bounds_modes=[bound_mode] * optimization_levels if isinstance(bound_mode, str) else bound_mode,
                methods=[method] * optimization_levels if isinstance(method, str) else method,
                square_size=square_size,
                levels=optimization_levels,
                verbose=verbose
            )
        # Apply optimized homography
        optimized_warp = self.transformer.apply_homography(img, optimized_H, output_size)
        
        return initial_H, optimized_H, squares, initial_warp, optimized_warp


    