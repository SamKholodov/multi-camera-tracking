import numpy as np
import optuna
from src.pipeline import GeometryUtils
optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptimizationTester:
    """
    Automatic optimization parameter search using Optuna
    """
    
    def __init__(self, processor, n_trials=100, timeout=3600):
        """
        Args:
            processor: HomographyProcessor instance
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
        """
        self.processor = processor
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
    
    def objective(self, trial, img_path, src_points, dst_points, rep_img, rep_dst,
                 anchor_src=None, anchor_dst=None, square_size=114, square_area=2000, 
                 output_size=(1600, 800)):
        """
        Objective function for Optuna optimization
        """
        # Define search spaces for parameters
        optimization_levels = trial.suggest_int('optimization_levels', 1, 4)
        
        methods = []
        bounds_modes = []
        weights = []
        
        for level in range(optimization_levels):
            # Method selection
            method_choice = trial.suggest_categorical(f'method_{level}', 
                                                     ['Powell', 'L-BFGS-B', 'TNC'])
            methods.append(method_choice)
            
            # Bounds mode
            bounds_choice = trial.suggest_categorical(f'bounds_{level}', 
                                                     ['adaptive', 'strict', 'none'])
            bounds_modes.append(bounds_choice)
            
            # Weight (log scale for better exploration)
            weight = trial.suggest_float(f'weight_{level}', 0.05, 2.0, log=True)
            weights.append(weight)
        
        try:
            # Set anchor points if not provided
            if anchor_dst is None and anchor_src is None:
                anchor_src = src_points
                anchor_dst = dst_points

            # Detect squares
            squares = self.processor.detector.detect_squares(
                img_path=img_path,
                square_area=square_area,
                save_picture=False
            )
            
            if len(squares) < 1:
                return float('inf')  # Penalize no squares detected
            
            # Process image with current configuration
            initial_H, optimized_H, squares, initial_warp, optimized_warp = self.processor.process_image(
                img_path=img_path,
                src_points=src_points,
                dst_points=dst_points,
                anchor_points=anchor_src,
                anchor_targets=anchor_dst,
                weight=weights,
                method=methods,
                bound_mode=bounds_modes,
                output_size=output_size,
                square_size=square_size,
                square_area=square_area,
                verbose=False,
                optimization_levels=optimization_levels)
            
            # Calculate reprojection error
            optimized_error, _, _ = GeometryUtils.calculate_reprojection_error(optimized_H, rep_img, rep_dst)
            
            return optimized_error  # Возвращаем только ошибку репроекции
            
        except Exception as e:
            # Return high error for failed trials
            return float('inf')
    
    def optimize_parameters(self, img_path, src_points, dst_points, rep_img, rep_dst,
                          anchor_src=None, anchor_dst=None, square_size=114, 
                          square_area=2000, output_size=(1600, 800), verbose=True):
        """
        Find optimal parameters using Optuna
        
        Returns:
            dict: Best parameters found
            optuna.study.Study: Complete study object
        """
        if verbose:
            print(f"Starting Optuna optimization with {self.n_trials} trials...")
        
        # Create study with disabled output
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        # Wrap objective function with fixed parameters
        objective_with_args = lambda trial: self.objective(
            trial, img_path, src_points, dst_points, rep_img, rep_dst,
            anchor_src, anchor_dst, square_size, square_area, output_size
        )
        
        # Run optimization with disabled progress bar
        self.study.optimize(
            objective_with_args, 
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False  # Отключаем прогресс бар
        )
        
        return self.study.best_params, self.study
    
    def get_best_configuration(self, square_area=2000):
        """
        Convert best parameters to configuration format compatible with process_image
        
        Returns:
            dict: Configuration in the same format as manual configurations
        """
        if self.study is None:
            raise ValueError("No optimization study available. Run optimize_parameters first.")
        
        best_params = self.study.best_params
        
        config = {
            'name': f"Optuna-Best-{self.study.best_trial.number}",
            'methods': [],
            'bounds_modes': [],
            'weights': [],
            'optimization_levels': best_params['optimization_levels'],
            'square_area': square_area  # Используем переданное значение
        }
        
        for level in range(best_params['optimization_levels']):
            config['methods'].append(best_params[f'method_{level}'])
            config['bounds_modes'].append(best_params[f'bounds_{level}'])
            config['weights'].append(best_params[f'weight_{level}'])
        
        return config


def test_optimized_configuration(processor, best_config, img_path, src_points, dst_points,
                               rep_img, rep_dst, anchor_src=None, anchor_dst=None,
                               square_size=114, output_size=(1600, 800), verbose=True):
    """
    Test the best configuration found by Optuna
    
    Returns:
        dict: Complete results with error metrics
    """
    if verbose:
        print(f"Testing optimized configuration: {best_config['name']}")
    
    try:
        # Set anchor points if not provided
        if anchor_dst is None and anchor_src is None:
            anchor_src = src_points
            anchor_dst = dst_points

        # Process image with best configuration
        initial_H, optimized_H, squares, initial_warp, optimized_warp = processor.process_image(
            img_path=img_path,
            src_points=src_points,
            dst_points=dst_points,
            anchor_points=anchor_src,
            anchor_targets=anchor_dst,
            weight=best_config['weights'],
            method=best_config['methods'],
            bound_mode=best_config['bounds_modes'],
            output_size=output_size,
            square_size=square_size,
            square_area=best_config['square_area'],
            verbose=False,  # Отключаем verbose для process_image
            optimization_levels=best_config['optimization_levels'])
        
        # Calculate errors
        initial_error, _, _ = GeometryUtils.calculate_reprojection_error(initial_H, rep_img, rep_dst)
        optimized_error, _, _ = GeometryUtils.calculate_reprojection_error(optimized_H, rep_img, rep_dst)
        improvement = initial_error - optimized_error
        improvement_percent = (improvement / initial_error) * 100 if initial_error > 0 else 0
        
        result = {
            'name': best_config['name'],
            'methods': best_config['methods'],
            'bounds_modes': best_config['bounds_modes'],
            'weights': best_config['weights'],
            'optimization_levels': best_config['optimization_levels'],
            'square_area': best_config['square_area'],
            'initial_error': initial_error,
            'optimized_error': optimized_error,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'squares_count': len(squares),
            'square_size': square_size,
            'output_size': output_size,
            'initial_warp': initial_warp,
            'optimized_warp': optimized_warp,
            'success': True
        }
        
        return result
        
    except Exception as e:
        return {
            'name': best_config['name'],
            'error': str(e),
            'success': False
        }


# Backward compatibility functions
def test_multiple_methods(processor, img_path, src_points, dst_points, rep_img, rep_dst,
                         anchor_src=None, anchor_dst=None, square_size=114, square_area=2000,
                         output_size=(1600, 800), verbose=False, use_optuna=True, n_trials=50):
    """
    Enhanced version with Optuna support
    
    Args:
        use_optuna: Whether to use Optuna for automatic parameter search
        n_trials: Number of Optuna trials if use_optuna=True
    """
    if use_optuna:
        # Use Optuna for automatic parameter search
        tester = OptimizationTester(processor, n_trials=n_trials)
        best_params, study = tester.optimize_parameters(
            img_path, src_points, dst_points, rep_img, rep_dst,
            anchor_src, anchor_dst, square_size, square_area, output_size, verbose
        )
        
        best_config = tester.get_best_configuration(square_area=square_area)
        result = test_optimized_configuration(
            processor, best_config, img_path, src_points, dst_points,
            rep_img, rep_dst, anchor_src, anchor_dst, square_size, output_size, verbose
        )
        
        return [result]
    else:
        # Fall back to original manual testing (simplified)
        if verbose:
            print("Manual testing mode selected. Using balanced configuration.")
        
        balanced_config = {
            'name': 'Balanced-Powell',
            'methods': ['Powell', 'Powell', 'Powell'],
            'bounds_modes': ['adaptive', 'strict', 'adaptive'],
            'weights': [0.7, 0.4, 0.2],
            'optimization_levels': 3,
            'square_area': square_area
        }
        
        result = test_optimized_configuration(
            processor, balanced_config, img_path, src_points, dst_points,
            rep_img, rep_dst, anchor_src, anchor_dst, square_size, output_size, verbose
        )
        
        return [result]


def results_summary(results, title, verbose=True):
    """
    Prints a formatted summary of optimization results
    """
    if not verbose:
        # Краткий вывод только лучшего результата
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            return None, None, None
            
        best_result = min(successful_results, key=lambda x: x['optimized_error'])
        
        print(f"RESULTS SUMMARY: {title}")
        print("=" * 80)
        print(f"{'Configuration':<25} {'Initial':<8} {'Final':<8} {'Improvement':<12} {'% Improve':<10} {'Squares'}")
        print("-" * 80)
        print(f"{best_result['name']:<25} {best_result['initial_error']:>7.2f} {best_result['optimized_error']:>7.2f} "
              f"{best_result['improvement']:>11.2f} {best_result['improvement_percent']:>9.1f}% {best_result['squares_count']:>7}")
        print("-" * 80)
        print(f"BEST CONFIGURATION: {best_result['name']}")
        print(f"Methods: {best_result['methods']}")
        print(f"Bounds: {best_result['bounds_modes']}")
        print(f"Weights: {best_result['weights']}")
        print(f"Optimization levels: {best_result.get('optimization_levels', 3)}")
        print(f"Square area: {best_result.get('square_area', 2000)}")
        print(f"Final error: {best_result['optimized_error']:.2f} pixels")
        print(f"Improvement: {best_result['improvement_percent']:.1f}%")
        
        best_config = {
            'name': best_result['name'],
            'methods': best_result['methods'],
            'bounds_modes': best_result['bounds_modes'],
            'weights': best_result['weights'],
            'optimization_levels': best_result.get('optimization_levels', 3),
            'square_area': best_result.get('square_area', 2000),
            'optimized_error': best_result['optimized_error'],
            'improvement_percent': best_result['improvement_percent']
        }
        
        return successful_results, [], best_config
    
    # Полный вывод (оригинальный код)
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY: {title}")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        print(f"{'Configuration':<25} {'Initial':<8} {'Final':<8} {'Improvement':<12} {'% Improve':<10} {'Squares'}")
        print(f"{'-'*80}")
        
        for result in successful_results:
            print(f"{result['name']:<25} {result['initial_error']:>7.2f} {result['optimized_error']:>7.2f} "
                  f"{result['improvement']:>11.2f} {result['improvement_percent']:>9.1f}% {result['squares_count']:>7}")
    
    if failed_results:
        print(f"\nFailed configurations:")
        for result in failed_results:
            print(f"  {result['name']}: {result['error']}")
    
    best_config = None
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['optimized_error'])
        best_config = {
            'name': best_result['name'],
            'methods': best_result['methods'],
            'bounds_modes': best_result['bounds_modes'],
            'weights': best_result['weights'],
            'optimization_levels': best_result.get('optimization_levels', 3),
            'square_area': best_result.get('square_area', 2000),
            'optimized_error': best_result['optimized_error'],
            'improvement_percent': best_result['improvement_percent']
        }
        
        print(f"{'-'*80}")
        print(f"BEST CONFIGURATION: {best_result['name']}")
        print(f"Methods: {best_result['methods']}")
        print(f"Bounds: {best_result['bounds_modes']}")
        print(f"Weights: {best_result['weights']}")
        print(f"Optimization levels: {best_result.get('optimization_levels', 3)}")
        print(f"Square area: {best_result.get('square_area', 2000)}")
        print(f"Final error: {best_result['optimized_error']:.2f} pixels")
        print(f"Improvement: {best_result['improvement_percent']:.1f}%")
    
    return successful_results, failed_results, best_config


def get_best_params(results):
    """
    Extracts the best parameters from optimization results
    """
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        return None
    
    best_result = min(successful_results, key=lambda x: x['optimized_error'])
    
    best_params = {
        'name': best_result['name'],
        'methods': best_result['methods'],
        'bounds_modes': best_result['bounds_modes'],
        'weights': best_result['weights'],
        'optimization_levels': best_result.get('optimization_levels', 3),
        'square_area': best_result.get('square_area', 2000),
        'optimized_error': best_result['optimized_error'],
        'improvement_percent': best_result['improvement_percent'],
        'squares_count': best_result['squares_count']
    }
    
    return best_params