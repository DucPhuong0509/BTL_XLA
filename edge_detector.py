# edge_detector.py - Phát hiện biên đã được tối ưu hóa
import numpy as np
from typing import Optional
from grayscale import to_grayscale
from filter import gaussian_blur, convolve2d


def sobel_edge_detection(image: np.ndarray, blur_sigma: float = 1.0, 
                        threshold: Optional[float] = None) -> np.ndarray:
    """
    Phát hiện biên sử dụng toán tử Sobel - Chuẩn và chất lượng cao.
    
    Sobel operator phát hiện biên bằng cách tính gradient theo 2 hướng:
    - Gradient X: phát hiện biên dọc (vertical edges)
    - Gradient Y: phát hiện biên ngang (horizontal edges)
    
    Args:
        image: Ảnh đầu vào (RGB hoặc grayscale)
        blur_sigma: Độ mịn trước khi phát hiện biên (0.5-2.0)
                    - Nhỏ (0.5-0.8): giữ chi tiết, nhiều biên
                    - Lớn (1.5-2.0): mịn hơn, ít biên nhiễu
        threshold: Ngưỡng cắt biên yếu (0-255, None = không cắt)
                   - Giá trị cao: chỉ giữ biên mạnh
                   - Giá trị thấp: giữ cả biên yếu
    
    Returns:
        Ảnh biên (grayscale, 0-255)
        - Giá trị cao (sáng): biên mạnh
        - Giá trị thấp (tối): không có biên
    """
    # Chuyển sang ảnh xám nếu cần
    if image.ndim == 3:
        gray = to_grayscale(image)
    else:
        gray = image.copy()
    
    # Làm mịn ảnh trước để giảm nhiễu
    if blur_sigma > 0:
        smoothed = gaussian_blur(gray, size=5, sigma=blur_sigma)
    else:
        smoothed = gray
    
    # Kernel Sobel chuẩn
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # Tính gradient theo 2 hướng
    gradient_x = convolve2d(smoothed.astype(np.float32), sobel_x)
    gradient_y = convolve2d(smoothed.astype(np.float32), sobel_y)
    
    # Tính độ lớn gradient (magnitude)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Chuẩn hóa về [0, 255]
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    
    # Áp dụng threshold nếu có
    if threshold is not None:
        gradient_magnitude[gradient_magnitude < threshold] = 0
    
    return np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
