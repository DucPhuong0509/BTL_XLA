# sketch.py - Các hiệu ứng vẽ tay TỐI ƯU TỐC ĐỘ
import numpy as np
from scipy.ndimage import convolve


def create_hand_drawn_sketch(image: np.ndarray, diameter: int = 9, 
                           edge_strength: float = 1.0,
                           progress_callback: callable = None) -> np.ndarray:
    """
    Tạo hiệu ứng vẽ tay - PHIÊN BẢN TỐI ƯU TỐC ĐỘ
    """
    total_steps = 4
    current_step = 0
    
    def update_progress(message):
        nonlocal current_step
        if progress_callback:
            progress_callback(current_step, total_steps, message)
        current_step += 1
    
    # Bước 1: Chuyển ảnh xám
    update_progress("Đang chuẩn bị ảnh...")
    if image.ndim == 3:
        gray_image = balanced_grayscale(image)
    else:
        gray_image = image.copy()
    
    # Bước 2: Làm mịn NHANH bằng box filter
    update_progress("Đang làm mịn ảnh...")
    smoothed = fast_blur(gray_image, diameter)
    
    # Bước 3: Phát hiện biên NHANH
    update_progress("Đang phát hiện đường nét...")
    edges = fast_edge_detection(smoothed)
    
    # Bước 4: Kết hợp
    update_progress("Đang tạo hiệu ứng vẽ tay...")
    sketch = create_soft_sketch(smoothed, edges, edge_strength)
    
    return sketch


def pencil_sketch(image: np.ndarray) -> np.ndarray:
    """
    Hiệu ứng vẽ bút chì - PHIÊN BẢN TỐI ƯU
    """
    if image.ndim == 3:
        gray = high_contrast_grayscale(image)
    else:
        gray = image.copy()
    
    # Tăng contrast
    gray = adjust_brightness_contrast(gray, brightness=15, contrast=25)
    
    # Dodge blend NHANH
    inverted = 255 - gray
    blurred = fast_blur(inverted, 21)
    
    # Dodge blend
    sketch = dodge_blend(gray, blurred)
    
    # Gamma correction
    sketch = np.power(sketch / 255.0, 0.85) * 255.0
    
    # Thêm texture nhẹ
    sketch = add_pencil_texture(sketch.astype(np.uint8))
    
    return sketch.astype(np.uint8)


def fast_blur(image: np.ndarray, size: int) -> np.ndarray:
    """
    Box blur CỰC NHANH - dùng scipy.ndimage.convolve (tối ưu C)
    
    Box blur nhanh hơn Gaussian 10-20 lần!
    """
    if size % 2 == 0:
        size += 1
    
    # Tạo kernel box đơn giản
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    
    # Dùng scipy convolve (written in C, RẤT NHANH)
    blurred = convolve(image.astype(np.float32), kernel, mode='reflect')
    
    return np.clip(blurred, 0, 255).astype(np.uint8)


def fast_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Phát hiện biên Sobel NHANH - dùng scipy convolve
    """
    # Kernel Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # Dùng scipy convolve (C implementation)
    img_float = image.astype(np.float32)
    gradient_x = convolve(img_float, sobel_x, mode='reflect')
    gradient_y = convolve(img_float, sobel_y, mode='reflect')
    
    # Tính magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Chuẩn hóa bằng tanh (soft)
    edges_soft = np.tanh(gradient_magnitude / 60.0) * 255
    
    return np.clip(edges_soft, 0, 255).astype(np.uint8)


def balanced_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh xám NHANH với NumPy vectorization"""
    if image.ndim == 3:
        # Vectorized operation - RẤT NHANH
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()
    
    # Histogram equalization đơn giản
    gray = simple_histogram_equalization(gray)
    
    return np.clip(gray, 0, 255).astype(np.uint8)


def high_contrast_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển xám contrast cao NHANH"""
    if image.ndim == 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()
    
    # Tăng contrast - vectorized
    gray = gray.astype(np.float32)
    mean = np.mean(gray)
    gray = (gray - mean) * 1.8 + mean
    
    return np.clip(gray, 0, 255).astype(np.uint8)


def simple_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Histogram equalization ĐƠN GIẢN và NHANH
    """
    # Tính histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Ánh xạ - vectorized
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    return image_equalized.reshape(image.shape).astype(np.uint8)


def create_soft_sketch(smoothed: np.ndarray, edges: np.ndarray,
                      edge_strength: float) -> np.ndarray:
    """Kết hợp sketch - NHANH với vectorization"""
    # Vectorized operations
    edges_norm = edges.astype(np.float32) / 255.0
    edges_adjusted = np.power(edges_norm, 1.2) * edge_strength * 0.7
    
    sketch = smoothed.astype(np.float32) * (1.0 - edges_adjusted)
    
    # Tăng độ sáng
    sketch = adjust_brightness_contrast(sketch, brightness=20, contrast=15)
    
    # Noise nhẹ
    h, w = sketch.shape
    noise = np.random.normal(0, 2, (h, w)).astype(np.float32)
    sketch = sketch + noise
    
    return np.clip(sketch, 0, 255).astype(np.uint8)


def dodge_blend(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Dodge blend NHANH - vectorized"""
    base_float = base.astype(np.float32)
    blend_float = blend.astype(np.float32)
    
    # Vectorized operation
    result = np.divide(base_float * 255.0, 255.0 - blend_float + 1e-6)
    result = np.clip(result, 0, 255)
    
    return result


def add_pencil_texture(image: np.ndarray) -> np.ndarray:
    """Thêm texture - NHANH"""
    h, w = image.shape
    
    # Noise vectorized
    noise = np.random.normal(0, 4, (h, w)).astype(np.float32)
    textured = np.clip(image.astype(np.float32) + noise, 0, 255)
    
    # Gamma correction vectorized
    textured = np.power(textured / 255.0, 0.92) * 255
    
    return textured.astype(np.uint8)


def adjust_brightness_contrast(image: np.ndarray, brightness: float = 0, 
                               contrast: float = 0) -> np.ndarray:
    """Điều chỉnh brightness/contrast - NHANH"""
    image = image.astype(np.float32)
    
    if brightness != 0:
        image = image + brightness
    
    if contrast != 0:
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        image = factor * (image - 128) + 128
    
    return np.clip(image, 0, 255)