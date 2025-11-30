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


def sobel_edge_detection_fast(image: np.ndarray, blur_sigma: float = 0.8) -> np.ndarray:
    
    return sobel_edge_detection(image, blur_sigma=blur_sigma)


def prewitt_edge_detection(image: np.ndarray, blur_sigma: float = 1.0) -> np.ndarray:
    
    if image.ndim == 3:
        gray = to_grayscale(image)
    else:
        gray = image.copy()
    
    if blur_sigma > 0:
        smoothed = gaussian_blur(gray, size=5, sigma=blur_sigma)
    else:
        smoothed = gray
    
    # Kernel Prewitt
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    
    prewitt_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float32)
    
    gradient_x = convolve2d(smoothed.astype(np.float32), prewitt_x)
    gradient_y = convolve2d(smoothed.astype(np.float32), prewitt_y)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    
    return np.clip(gradient_magnitude, 0, 255).astype(np.uint8)


def canny_edge_detection(image: np.ndarray, low_threshold: float = 50, 
                        high_threshold: float = 150, 
                        blur_sigma: float = 1.4) -> np.ndarray:
    
    if image.ndim == 3:
        gray = to_grayscale(image)
    else:
        gray = image.copy()
    
    # Bước 1: Làm mịn Gaussian
    smoothed = gaussian_blur(gray, size=5, sigma=blur_sigma)
    
    # Bước 2: Tính gradient và hướng
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    gradient_x = convolve2d(smoothed.astype(np.float32), sobel_x)
    gradient_y = convolve2d(smoothed.astype(np.float32), sobel_y)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # Bước 3: Non-maximum suppression
    edges_thin = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # Bước 4: Double thresholding và edge tracking
    edges_final = double_threshold(edges_thin, low_threshold, high_threshold)
    
    return edges_final


def non_maximum_suppression(gradient_magnitude: np.ndarray, 
                           gradient_direction: np.ndarray) -> np.ndarray:
    
    h, w = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    
    # Chuyển góc về 4 hướng chính: 0°, 45°, 90°, 135°
    angle = gradient_direction * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 255
            r = 255
            
            # Xác định 2 pixel láng giềng theo hướng gradient
            # 0° (horizontal)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            # 45° (diagonal)
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            # 90° (vertical)
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            # 135° (diagonal)
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]
            
            # Giữ pixel nếu nó lớn nhất trong 3 pixel
            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
    
    return suppressed


def double_threshold(image: np.ndarray, low_threshold: float, 
                    high_threshold: float) -> np.ndarray:

    h, w = image.shape
    
    # Phân loại pixel
    strong = 255
    weak = 75
    
    result = np.zeros_like(image, dtype=np.uint8)
    
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    # Edge tracking: kết nối weak edge với strong edge
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if result[i, j] == weak:
                # Kiểm tra 8 pixel láng giềng
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    
    return result


def laplacian_edge_detection(image: np.ndarray, blur_sigma: float = 1.0) -> np.ndarray:

    if image.ndim == 3:
        gray = to_grayscale(image)
    else:
        gray = image.copy()
    
    if blur_sigma > 0:
        smoothed = gaussian_blur(gray, size=5, sigma=blur_sigma)
    else:
        smoothed = gray
    
    # Kernel Laplacian
    laplacian = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]], dtype=np.float32)
    
    edges = convolve2d(smoothed.astype(np.float32), laplacian)
    
    # Lấy giá trị tuyệt đối
    edges = np.abs(edges)
    
    # Chuẩn hóa
    if edges.max() > 0:
        edges = (edges / edges.max()) * 255
    
    return np.clip(edges, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    
    # Đảm bảo size là số lẻ
    if size % 2 == 0:
        size += 1
    
    # Tạo kernel Gaussian
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    # Áp dụng convolution
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)