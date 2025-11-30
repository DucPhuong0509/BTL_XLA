# edge_detector.py - Phát hiện biên đã được tối ưu hóa
import numpy as np
from typing import Tuple, Optional


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
    
    Examples:
        >>> edges = sobel_edge_detection(image, blur_sigma=0.8)
        >>> edges_strong = sobel_edge_detection(image, threshold=50)
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
    gradient_x = convolve2d_fast(smoothed.astype(np.float32), sobel_x)
    gradient_y = convolve2d_fast(smoothed.astype(np.float32), sobel_y)
    
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
    """
    Phát hiện biên Sobel TỐI ƯU TỐC ĐỘ - alias cho hàm chính.
    
    Đây là alias của sobel_edge_detection() với tham số mặc định tối ưu.
    """
    return sobel_edge_detection(image, blur_sigma=blur_sigma)


def prewitt_edge_detection(image: np.ndarray, blur_sigma: float = 1.0) -> np.ndarray:
    """
    Phát hiện biên sử dụng toán tử Prewitt.
    
    Prewitt tương tự Sobel nhưng kernel đơn giản hơn (không có trọng số).
    Kết quả gần giống Sobel nhưng nhạy cảm hơn với nhiễu.
    
    Args:
        image: Ảnh đầu vào
        blur_sigma: Độ mịn trước khi phát hiện
    
    Returns:
        Ảnh biên
    """
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
    
    gradient_x = convolve2d_fast(smoothed.astype(np.float32), prewitt_x)
    gradient_y = convolve2d_fast(smoothed.astype(np.float32), prewitt_y)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    
    return np.clip(gradient_magnitude, 0, 255).astype(np.uint8)


def canny_edge_detection(image: np.ndarray, low_threshold: float = 50, 
                        high_threshold: float = 150, 
                        blur_sigma: float = 1.4) -> np.ndarray:
    """
    Phát hiện biên Canny - CHẤT LƯỢNG CAO NHẤT nhưng phức tạp.
    
    Canny là thuật toán phát hiện biên tốt nhất với 4 bước:
    1. Làm mịn Gaussian để giảm nhiễu
    2. Tính gradient và hướng
    3. Non-maximum suppression: làm mỏng biên
    4. Double thresholding: lọc biên yếu
    
    Args:
        image: Ảnh đầu vào
        low_threshold: Ngưỡng thấp (30-80)
                       - Pixel > low: biên yếu (candidate)
        high_threshold: Ngưỡng cao (100-200)
                        - Pixel > high: biên mạnh (confirmed)
        blur_sigma: Độ mịn Gaussian (1.0-2.0)
    
    Returns:
        Ảnh biên nhị phân (0 hoặc 255)
    
    Notes:
        Canny cho kết quả sắc nét nhất nhưng CHẬM nhất.
        Dùng cho ảnh cần biên rất rõ ràng.
    """
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
    
    gradient_x = convolve2d_fast(smoothed.astype(np.float32), sobel_x)
    gradient_y = convolve2d_fast(smoothed.astype(np.float32), sobel_y)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # Bước 3: Non-maximum suppression
    edges_thin = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # Bước 4: Double thresholding và edge tracking
    edges_final = double_threshold(edges_thin, low_threshold, high_threshold)
    
    return edges_final


def non_maximum_suppression(gradient_magnitude: np.ndarray, 
                           gradient_direction: np.ndarray) -> np.ndarray:
    """
    Non-maximum suppression - Làm mỏng biên.
    
    Chỉ giữ lại pixel có gradient_magnitude lớn nhất theo hướng gradient.
    Kết quả: biên mỏng 1 pixel.
    
    Args:
        gradient_magnitude: Độ lớn gradient
        gradient_direction: Hướng gradient (radian)
    
    Returns:
        Ảnh biên đã làm mỏng
    """
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
    """
    Double thresholding và edge tracking by hysteresis.
    
    - Strong edges: pixel > high_threshold → giữ lại
    - Weak edges: low_threshold < pixel < high_threshold → giữ nếu nối với strong edge
    - Non-edges: pixel < low_threshold → loại bỏ
    
    Args:
        image: Ảnh biên đã NMS
        low_threshold: Ngưỡng thấp
        high_threshold: Ngưỡng cao
    
    Returns:
        Ảnh biên nhị phân (0 hoặc 255)
    """
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
    """
    Phát hiện biên sử dụng Laplacian (đạo hàm bậc 2).
    
    Laplacian phát hiện vùng thay đổi nhanh về cường độ.
    Nhạy cảm với nhiễu hơn Sobel.
    
    Args:
        image: Ảnh đầu vào
        blur_sigma: Độ mịn trước khi phát hiện
    
    Returns:
        Ảnh biên
    """
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
    
    edges = convolve2d_fast(smoothed.astype(np.float32), laplacian)
    
    # Lấy giá trị tuyệt đối
    edges = np.abs(edges)
    
    # Chuẩn hóa
    if edges.max() > 0:
        edges = (edges / edges.max()) * 255
    
    return np.clip(edges, 0, 255).astype(np.uint8)


# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def convolve2d_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolution 2D tối ưu cho kernel 3x3.
    
    Args:
        image: Ảnh đầu vào (2D float array)
        kernel: Kernel 3x3
    
    Returns:
        Ảnh sau convolution
    """
    h, w = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    
    # Padding với chế độ reflect
    padded = np.pad(image, 1, mode='reflect')
    
    # Convolution
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)
    
    return output


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh màu sang ảnh xám sử dụng công thức luminance chuẩn ITU-R BT.601.
    
    Args:
        image: Ảnh RGB hoặc RGBA
    
    Returns:
        Ảnh xám
    """
    if image.ndim == 3 and image.shape[2] >= 3:
        # Sử dụng trọng số chuẩn: Red=0.299, Green=0.587, Blue=0.114
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()
    
    return np.clip(gray, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Làm mịn ảnh bằng Gaussian blur.
    
    Args:
        image: Ảnh xám đầu vào
        size: Kích thước kernel (số lẻ)
        sigma: Độ lệch chuẩn
    
    Returns:
        Ảnh đã làm mịn
    """
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