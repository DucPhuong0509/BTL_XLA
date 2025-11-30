# sketch.py - Các hiệu ứng vẽ tay
import numpy as np
from scipy.ndimage import convolve
from grayscale import to_grayscale
from filter import box_blur, gaussian_blur
from edge_detector import sobel_edge_detection


def create_hand_drawn_sketch(image: np.ndarray, diameter: int = 9, 
                           edge_strength: float = 1.0,
                           progress_callback: callable = None) -> np.ndarray:
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
        gray_image = to_grayscale(image)
    else:
        gray_image = image.copy()
    
    # Cân bằng histogram
    gray_image = simple_histogram_equalization(gray_image)
    
    # Bước 2: Làm mịn 
    update_progress("Đang làm mịn ảnh...")
    smoothed = box_blur(gray_image, size=diameter)
    
    # Bước 3: Phát hiện biên 
    update_progress("Đang phát hiện đường nét...")
    edges = sobel_edge_detection(smoothed, blur_sigma=0.5)
    
    # Làm mềm edges
    edges_soft = np.tanh(edges.astype(np.float32) / 60.0) * 255
    edges_soft = edges_soft.astype(np.uint8)
    
    # Bước 4: Kết hợp tạo sketch
    update_progress("Đang tạo hiệu ứng vẽ tay...")
    sketch = create_soft_sketch(smoothed, edges_soft, edge_strength)
    
    return sketch


def pencil_sketch(image: np.ndarray) -> np.ndarray:
    # Chuyển xám 
    if image.ndim == 3:
        gray = to_grayscale(image)
    else:
        gray = image.copy()
    
    # Tăng contrast
    gray = high_contrast_grayscale(gray)
    gray = adjust_brightness_contrast(gray, brightness=15, contrast=25)
    
    # Dodge blend với blur
    inverted = 255 - gray
    blurred = gaussian_blur(inverted, size=21, sigma=3.5)
    
    # Dodge blend
    sketch = dodge_blend(gray, blurred)
    
    # Gamma correction
    sketch = np.power(sketch / 255.0, 0.85) * 255.0
    
    # Thêm texture
    sketch = add_pencil_texture(sketch.astype(np.uint8))
    
    return sketch.astype(np.uint8)


def high_contrast_grayscale(gray: np.ndarray) -> np.ndarray:
    # Tăng contrast cho ảnh xám
    gray = gray.astype(np.float32)
    mean = np.mean(gray)
    gray = (gray - mean) * 1.8 + mean
    return np.clip(gray, 0, 255).astype(np.uint8)


def simple_histogram_equalization(image: np.ndarray) -> np.ndarray:
    # Cân bằng histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return image_equalized.reshape(image.shape).astype(np.uint8)


def create_soft_sketch(smoothed: np.ndarray, edges: np.ndarray,
                      edge_strength: float) -> np.ndarray:
    # Kết hợp tạo sketch mềm mại
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
    # Dodge blending mode
    base_float = base.astype(np.float32)
    blend_float = blend.astype(np.float32)
    
    result = np.divide(base_float * 255.0, 255.0 - blend_float + 1e-6)
    result = np.clip(result, 0, 255)
    
    return result


def add_pencil_texture(image: np.ndarray) -> np.ndarray:
    # Thêm texture bút chì
    h, w = image.shape
    
    noise = np.random.normal(0, 4, (h, w)).astype(np.float32)
    textured = np.clip(image.astype(np.float32) + noise, 0, 255)
    
    textured = np.power(textured / 255.0, 0.92) * 255
    
    return textured.astype(np.uint8)


def adjust_brightness_contrast(image: np.ndarray, brightness: float = 0, 
                               contrast: float = 0) -> np.ndarray:
        # Điều chỉnh brightness và contrast
    image = image.astype(np.float32)
    
    if brightness != 0:
        image = image + brightness
    
    if contrast != 0:
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        image = factor * (image - 128) + 128
    
    return np.clip(image, 0, 255)