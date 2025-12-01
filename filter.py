# filter.py - Các bộ lọc ảnh
import numpy as np


def get_gaussian_kernel(size: int, sigma: float) -> np.ndarray:

    # Đảm bảo kích thước là số lẻ
    if size % 2 == 0:
        size += 1
    
    # Tạo lưới tọa độ 2D
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    
    # Công thức Gaussian 2D: exp(-(x² + y²) / (2σ²))
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    
    # Chuẩn hóa để tổng = 1
    kernel = kernel / np.sum(kernel)
    
    return kernel.astype(np.float32)


def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:

    # Tạo kernel Gaussian
    kernel = get_gaussian_kernel(size, sigma)
    
    if image.ndim == 3:
        # Ảnh màu: xử lý từng kênh riêng
        h, w, c = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        
        for channel in range(c):
            result[:, :, channel] = convolve2d(image[:, :, channel].astype(np.float32), kernel)
    else:
        # Ảnh xám: xử lý trực tiếp
        result = convolve2d(image.astype(np.float32), kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    # Thêm padding với chế độ reflect
    image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # Khởi tạo output
    output = np.zeros_like(image)
    
    # Thực hiện convolution
    for i in range(img_h):
        for j in range(img_w):
            region = image_padded[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)
    
    return output


def box_blur(image: np.ndarray, size: int = 5) -> np.ndarray:

    if size % 2 == 0:
        size += 1
    
    # Tạo kernel box (tất cả giá trị bằng nhau)
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    
    if image.ndim == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d(image[:, :, c].astype(np.float32), kernel)
    else:
        result = convolve2d(image.astype(np.float32), kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)
