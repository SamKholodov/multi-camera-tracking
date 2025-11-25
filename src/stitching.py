# stitching.py
import cv2
import numpy as np

def simple_stitch_images(img1, img2):
    """
    Простая склейка двух изображений, которые частично перекрываются
    
    Args:
        img1: первое изображение (numpy array)
        img2: второе изображение (numpy array)
        
    Returns:
        numpy.array: склеенное изображение
    """
    # Определяем общий размер холста
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Вычисляем общий размер (максимальные координаты)
    total_width = max(w1, w2)
    total_height = max(h1, h2)
    
    # Если изображения одинакового размера и перекрываются по горизонтали
    if w1 == w2 and h1 == h2:
        # Предполагаем, что img1 слева, img2 справа с перекрытием
        overlap_region = 100  # предполагаемая область перекрытия
        
        # Создаем общий холст
        result_width = w1 + w2 - overlap_region
        result = np.zeros((h1, result_width, 3), dtype=np.uint8)
        
        # Размещаем левое изображение
        result[:, :w1] = img1
        
        # Размещаем правое изображение с перекрытием
        result[:, w1 - overlap_region:w1 - overlap_region + w2] = np.where(
            img2 > 0, img2, result[:, w1 - overlap_region:w1 - overlap_region + w2]
        )
        
        return result

def stitch_with_overlap_detection(img1, img2):
    """
    Склейка с автоматическим определением области перекрытия
    
    Args:
        img1: первое изображение (numpy array)
        img2: второе изображение (numpy array)
        
    Returns:
        numpy.array: склеенное изображение
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Создаем маски ненулевых пикселей
    mask1 = np.any(img1 > 10, axis=2)
    mask2 = np.any(img2 > 10, axis=2)
    
    # Находим область перекрытия
    overlap = np.zeros((max(h1, h2), max(w1, w2)), dtype=bool)
    overlap[:h1, :w1] = mask1
    overlap[:h2, :w2] = overlap[:h2, :w2] | mask2
    
    # Определяем границы содержания
    coords = np.where(overlap)
    if len(coords[0]) == 0:
        return simple_stitch_images(img1, img2)
    
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    
    # Создаем обрезанный результат
    result_width = x_max - x_min + 1
    result_height = y_max - y_min + 1
    result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
    
    # Переносим пиксели из обоих изображений
    for y in range(result_height):
        for x in range(result_width):
            global_y = y + y_min
            global_x = x + x_min
            
            pixel1 = np.array([0, 0, 0])
            pixel2 = np.array([0, 0, 0])
            
            # Берем пиксель из первого изображения если есть
            if global_y < h1 and global_x < w1 and np.any(img1[global_y, global_x] > 10):
                pixel1 = img1[global_y, global_x]
            
            # Берем пиксель из второго изображения если есть
            if global_y < h2 and global_x < w2 and np.any(img2[global_y, global_x] > 10):
                pixel2 = img2[global_y, global_x]
            
            # Приоритет ненулевому пикселю
            if np.any(pixel1 > 10):
                result[y, x] = pixel1
            elif np.any(pixel2 > 10):
                result[y, x] = pixel2
    
    return result

def find_optimal_seam_mask(img1, img2):
    """
    Находит оптимальную границу для смешивания используя карту разностей
    с правильной обработкой областей без перекрытия
    """
    # Создаем улучшенные маски - учитываем любые не-черные пиксели
    mask1 = np.any(img1 > 5, axis=2).astype(np.uint8)
    mask2 = np.any(img2 > 5, axis=2).astype(np.uint8)
    
    # Находим область перекрытия
    overlap = mask1 & mask2
    
    # Если нет перекрытия, создаем простую маску разделения
    if np.sum(overlap) == 0:
        seam_mask = np.zeros_like(mask1)
        # Определяем грубую границу по центрам масс
        coords1 = np.where(mask1)
        coords2 = np.where(mask2)
        
        if len(coords1[0]) > 0 and len(coords2[0]) > 0:
            center1_x = np.mean(coords1[1])
            center2_x = np.mean(coords2[1])
            boundary_x = int((center1_x + center2_x) / 2)
            seam_mask[:, :boundary_x] = 1
        else:
            # Если только одно изображение имеет содержание
            seam_mask = mask1.copy()
        
        return seam_mask
    
    # Вычисляем разницу в области перекрытия
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # Применяем размытие для сглаживания
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    
    height, width = overlap.shape
    seam_mask = np.zeros_like(overlap)
    
    # СНАЧАЛА заполняем области БЕЗ перекрытия
    # Там где есть только img1 - маска = 1
    only_img1 = (mask1 == 1) & (mask2 == 0)
    seam_mask[only_img1] = 1
    
    # Там где есть только img2 - маска = 0 (уже 0 по умолчанию)
    
    # ТЕПЕРЬ обрабатываем область перекрытия
    for y in range(height):
        row_overlap = np.where(overlap[y] > 0)[0]
        if len(row_overlap) > 0:
            x_start, x_end = row_overlap[0], row_overlap[-1]
            
            # Находим точку с минимальной разницей в области перекрытия
            if x_end - x_start > 5:  # Минимальная ширина для поиска
                differences = diff_blur[y, x_start:x_end+1]
                min_idx = np.argmin(differences)
                best_x = x_start + min_idx
                
                # В области перекрытия: слева от границы - img1, справа - img2
                seam_mask[y, x_start:best_x] = 1
                # seam_mask[y, best_x:x_end+1] = 0 (уже 0 по умолчанию)
    
    return seam_mask

def advanced_blend_images(img1, img2, alpha=0.5):
    """
    Улучшенное смешивание с обработкой черных областей
    
    Args:
        img1: первое изображение (numpy array)
        img2: второе изображение (numpy array)
        alpha: коэффициент смешивания (по умолчанию 0.5)
        
    Returns:
        numpy.array: смешанное изображение
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    
    mask1 = np.any(img1 > 0, axis=2)  
    mask2 = np.any(img2 > 0, axis=2) 
    
    overlap_mask = mask1 & mask2
    
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    result = np.zeros_like(img1_float)
    
    result[overlap_mask] = (img1_float[overlap_mask] * alpha + 
                           img2_float[overlap_mask] * alpha)
    
    only_img1 = mask1 & ~overlap_mask
    result[only_img1] = img1_float[only_img1] * alpha
    
    only_img2 = mask2 & ~overlap_mask
    result[only_img2] = img2_float[only_img2] * alpha
    
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)

def draw_points(img, points, color=(0, 255, 0), radius=6):
    """
    Рисует точки на изображении
    
    Args:
        img: изображение (numpy array)
        points: массив точек формата [[x1, y1], [x2, y2], ...]
        color: цвет точек (BGR)
        radius: радиус точек
        
    Returns:
        numpy.array: изображение с нарисованными точками
    """
    img_with_points = img.copy()
    for point in points:
        point_int = tuple(point.astype(int))
        cv2.circle(img_with_points, point_int, radius, color, -1)
    return img_with_points

def calculate_stitched_reprojection_errors(H_left, H_right, rep_img1, rep_img2, rep_dst1, rep_dst2):
    """
    Вычисляет ошибки репроекции для склеенного изображения
    
    Args:
        H_left: гомография для левого изображения
        H_right: гомография для правого изображения
        rep_img1: точки репроекции для левого изображения
        rep_img2: точки репроекции для правого изображения
        rep_dst1: целевые точки для левого изображения
        rep_dst2: целевые точки для правого изображения
        
    Returns:
        tuple: (mean_error, left_errors, right_errors, all_errors)
    """
    
    left_projected = cv2.perspectiveTransform(
        rep_img1.reshape(-1, 1, 2).astype(np.float32), 
        H_left
    ).reshape(-1, 2)
    
    right_projected = cv2.perspectiveTransform(
        rep_img2.reshape(-1, 1, 2).astype(np.float32),
        H_right
    ).reshape(-1, 2)
    
    left_errors = [np.linalg.norm(real_pt - proj_pt) 
                for real_pt, proj_pt in zip(rep_dst1, left_projected)]
    right_errors = [np.linalg.norm(real_pt - proj_pt) 
                    for real_pt, proj_pt in zip(rep_dst2, right_projected)]
    
    all_errors = left_errors + right_errors
    mean_error = np.mean(all_errors)
    
    return mean_error, left_errors, right_errors, all_errors

def draw_reprojections_on_stitched(stitched_img, H_left, H_right, 
                                 rep_img1, rep_img2, 
                                 rep_dst1_transformed, rep_dst2_transformed):
    """
    Наносит репроекционные ошибки на склеенное изображение
    
    Args:
        stitched_img: склеенное изображение
        H_left: гомография для левого изображения
        H_right: гомография для правого изображения
        rep_img1: точки репроекции для левого изображения
        rep_img2: точки репроекции для правого изображения
        rep_dst1_transformed: целевые точки для левого изображения (в координатах склеенного изображения)
        rep_dst2_transformed: целевые точки для правого изображения (в координатах склеенного изображения)
        
    Returns:
        numpy.array: склеенное изображение с нарисованными репроекционными ошибками
    """
    # Создаем копию склеенного изображения
    stitched_with_repro = stitched_img.copy()
    if len(stitched_with_repro.shape) == 2:
        stitched_with_repro = cv2.cvtColor(stitched_with_repro, cv2.COLOR_GRAY2BGR)
    
    # Преобразуем точки репроекции для левого изображения
    left_projected = cv2.perspectiveTransform(
        rep_img1.reshape(-1, 1, 2).astype(np.float32), 
        H_left
    ).reshape(-1, 2)
    
    # Преобразуем точки репроекции для правого изображения  
    right_projected = cv2.perspectiveTransform(
        rep_img2.reshape(-1, 1, 2).astype(np.float32),
        H_right
    ).reshape(-1, 2)
    
    # Рисуем репроекции для левого изображения (синий цвет)
    for i, (real_pt, proj_pt) in enumerate(zip(rep_dst1_transformed, left_projected)):
        real_pt_int = tuple(real_pt.astype(int))
        proj_pt_int = tuple(proj_pt.astype(int))
        
        # Проверяем, что точки в пределах изображения
        if (0 <= proj_pt_int[0] < stitched_with_repro.shape[1] and 
            0 <= proj_pt_int[1] < stitched_with_repro.shape[0]):
            
            error = np.linalg.norm(real_pt - proj_pt)
            
            # Рисуем реальную точку (зеленая)
            cv2.circle(stitched_with_repro, real_pt_int, 10, (0, 255, 0), -1)
            # Рисуем спроецированную точку (красная)
            cv2.circle(stitched_with_repro, proj_pt_int, 8, (0, 0, 255), -1)
            # Рисуем линию между точками (синяя)
            cv2.line(stitched_with_repro, real_pt_int, proj_pt_int, (255, 0, 0), 3)
            
            # Добавляем текст
            cv2.putText(stitched_with_repro, f'L{i+1}', 
                    (real_pt_int[0] + 15, real_pt_int[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(stitched_with_repro, f'{error:.1f}', 
                    (proj_pt_int[0] + 15, proj_pt_int[1] + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Рисуем репроекции для правого изображения (оранжевый цвет)
    for i, (real_pt, proj_pt) in enumerate(zip(rep_dst2_transformed, right_projected)):
        real_pt_int = tuple(real_pt.astype(int))
        proj_pt_int = tuple(proj_pt.astype(int))
        
        # Проверяем, что точки в пределах изображения
        if (0 <= proj_pt_int[0] < stitched_with_repro.shape[1] and 
            0 <= proj_pt_int[1] < stitched_with_repro.shape[0]):
            
            error = np.linalg.norm(real_pt - proj_pt)
            
            # Рисуем реальную точку (зеленая)
            cv2.circle(stitched_with_repro, real_pt_int, 10, (0, 255, 0), -1)
            # Рисуем спроецированную точку (красная)
            cv2.circle(stitched_with_repro, proj_pt_int, 8, (0, 0, 255), -1)
            # Рисуем линию между точками (оранжевая)
            cv2.line(stitched_with_repro, real_pt_int, proj_pt_int, (0, 165, 255), 3)
            
            # Добавляем текст
            cv2.putText(stitched_with_repro, f'R{i+1}', 
                    (real_pt_int[0] + 15, real_pt_int[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(stitched_with_repro, f'{error:.1f}', 
                    (proj_pt_int[0] + 15, proj_pt_int[1] + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Добавляем легенду
    legend_y = 30
    cv2.putText(stitched_with_repro, 'Reprojection Errors on Stitched Image', 
            (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    legend_y += 30
    cv2.putText(stitched_with_repro, 'Green: Ground truth points', 
            (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    legend_y += 25
    cv2.putText(stitched_with_repro, 'Red: Projected points', 
            (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    legend_y += 25
    cv2.putText(stitched_with_repro, 'Blue: Left image errors', 
            (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    legend_y += 25
    cv2.putText(stitched_with_repro, 'Orange: Right image errors', 
            (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    return stitched_with_repro


def stitch_with_optimal_seam(img1, img2, blend_width=20):
    """
    Stitch images using optimal seam detection with smooth blending
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    
    h, w = img1.shape[:2]
    
    # Find optimal seam mask
    seam_mask = find_optimal_seam_mask(img1, img2)
    
    # Create smooth blending mask ONLY in overlap regions
    blend_mask = np.zeros_like(seam_mask, dtype=np.float32)
    
    # Определяем область перекрытия для плавного перехода
    mask1 = np.any(img1 > 5, axis=2)
    mask2 = np.any(img2 > 5, axis=2)
    overlap_area = mask1 & mask2
    
    # Create smooth transition only in overlap area
    for y in range(h):
        if not np.any(overlap_area[y]):
            continue
            
        # Находим границы перекрытия в этой строке
        overlap_indices = np.where(overlap_area[y])[0]
        if len(overlap_indices) == 0:
            continue
            
        x_start, x_end = overlap_indices[0], overlap_indices[-1]
        
        # Находим точку перехода в этой строке
        seam_row = seam_mask[y]
        transitions = np.where(np.diff(seam_row.astype(int)) != 0)[0]
        
        if len(transitions) > 0:
            seam_x = transitions[0] + 1
            
            # Ограничиваем область смешивания только перекрытием
            blend_start = max(x_start, seam_x - blend_width)
            blend_end = min(x_end, seam_x + blend_width)
            
            # Линейный градиент
            for x in range(blend_start, blend_end):
                if x < seam_x:
                    distance = seam_x - x
                    blend_mask[y, x] = max(0, 1.0 - distance / blend_width)
                else:
                    distance = x - seam_x
                    blend_mask[y, x] = min(1.0, distance / blend_width)
    
    # Convert images to float for blending
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    # Create result
    result_float = np.zeros_like(img1_float)
    
    # 1. Области с плавным смешиванием
    blend_area = blend_mask > 0
    result_float[blend_area] = (
        img1_float[blend_area] * (1 - blend_mask[blend_area][:, np.newaxis]) +
        img2_float[blend_area] * blend_mask[blend_area][:, np.newaxis]
    )
    
    # 2. Области только с img1 (включая не-перекрывающиеся)
    only_img1_area = (seam_mask == 1) & (~blend_area)
    result_float[only_img1_area] = img1_float[only_img1_area]
    
    # 3. Области только с img2 (включая не-перекрывающиеся)
    only_img2_area = (seam_mask == 0) & (~blend_area)
    result_float[only_img2_area] = img2_float[only_img2_area]
    
    # Convert back to uint8
    result = np.clip(result_float, 0, 255).astype(np.uint8)
    
    return result

def stitch_with_multiband_blending(img1, img2, levels=5):
    """
    Advanced stitching using multi-band blending for seamless results
    
    Args:
        img1: first image (numpy array)
        img2: second image (numpy array)
        levels: number of pyramid levels for multi-band blending
        
    Returns:
        numpy.array: seamlessly blended image
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    
    # Find optimal seam mask
    seam_mask = find_optimal_seam_mask(img1, img2)
    
    # Create masks for both images
    mask1 = np.any(img1 > 10, axis=2).astype(np.float32)
    mask2 = np.any(img2 > 10, axis=2).astype(np.float32)
    
    # Create combined mask for blending
    blend_mask = seam_mask.astype(np.float32)
    
    # Create Gaussian pyramid for both images and mask
    def build_gaussian_pyramid(img, levels):
        pyramid = [img.astype(np.float32)]
        for i in range(levels-1):
            pyramid.append(cv2.GaussianBlur(pyramid[-1], (5, 5), 0))
            pyramid[-1] = cv2.resize(pyramid[-1], 
                                   (pyramid[-1].shape[1]//2, pyramid[-1].shape[0]//2))
        return pyramid
    
    def build_laplacian_pyramid(img, levels):
        gaussian_pyramid = build_gaussian_pyramid(img, levels)
        laplacian_pyramid = []
        for i in range(levels-1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.resize(gaussian_pyramid[i+1], size)
            laplacian_pyramid.append(gaussian_pyramid[i] - expanded)
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid
    
    # Build pyramids for both images
    pyramid1 = build_laplacian_pyramid(img1, levels)
    pyramid2 = build_laplacian_pyramid(img2, levels)
    
    # Build Gaussian pyramid for mask
    mask_pyramid = build_gaussian_pyramid(blend_mask, levels)
    
    # Blend pyramids
    blended_pyramid = []
    for i in range(levels):
        # For color images, blend each channel
        if len(pyramid1[i].shape) == 3:
            blended_level = np.zeros_like(pyramid1[i])
            for channel in range(3):
                mask_resized = cv2.resize(mask_pyramid[i], 
                                        (pyramid1[i].shape[1], pyramid1[i].shape[0]))
                if i < levels - 1:
                    # For Laplacian levels
                    blended_level[:,:,channel] = (
                        pyramid1[i][:,:,channel] * (1 - mask_resized) +
                        pyramid2[i][:,:,channel] * mask_resized
                    )
                else:
                    # For Gaussian level (top of pyramid)
                    blended_level[:,:,channel] = (
                        pyramid1[i][:,:,channel] * (1 - mask_resized) +
                        pyramid2[i][:,:,channel] * mask_resized
                    )
        else:
            # For grayscale
            mask_resized = cv2.resize(mask_pyramid[i], 
                                    (pyramid1[i].shape[1], pyramid1[i].shape[0]))
            if i < levels - 1:
                blended_level = (
                    pyramid1[i] * (1 - mask_resized) +
                    pyramid2[i] * mask_resized
                )
            else:
                blended_level = (
                    pyramid1[i] * (1 - mask_resized) +
                    pyramid2[i] * mask_resized
                )
        
        blended_pyramid.append(blended_level)
    
    # Reconstruct from pyramid
    result = blended_pyramid[-1]
    for i in range(levels-2, -1, -1):
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        result = cv2.resize(result, size) + blended_pyramid[i]
    
    # Ensure non-overlapping areas are preserved
    result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
    
    # Apply masks to preserve non-overlapping areas
    final_result = np.zeros_like(img1)
    
    # Areas with only img1
    only_img1 = (mask1 > 0) & (mask2 == 0)
    final_result[only_img1] = img1[only_img1]
    
    # Areas with only img2  
    only_img2 = (mask2 > 0) & (mask1 == 0)
    final_result[only_img2] = img2[only_img2]
    
    # Overlapping areas use the blended result
    overlap_area = (mask1 > 0) & (mask2 > 0)
    final_result[overlap_area] = result_uint8[overlap_area]
    
    return final_result