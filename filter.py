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


def bilateral_filter(image: np.ndarray, diameter: int = 9, 
                    sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    
    # Đảm bảo diameter là số lẻ
    if diameter % 2 == 0:
        diameter += 1
    
    radius = diameter // 2
    h, w = image.shape[:2]
    
    # Tạo kernel không gian (Gaussian spatial kernel)
    space_kernel = get_gaussian_kernel(diameter, sigma_space)
    
    if image.ndim == 3:
        # Xử lý ảnh màu
        result = bilateral_filter_rgb(image, radius, space_kernel, sigma_color, sigma_space)
    else:
        # Xử lý ảnh xám
        result = bilateral_filter_gray(image, radius, space_kernel, sigma_color)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def bilateral_filter_gray(image: np.ndarray, radius: int, 
                         space_kernel: np.ndarray, sigma_color: float) -> np.ndarray:

    h, w = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    
    # Tối ưu: xử lý với step để tăng tốc
    step = 1 if min(h, w) < 500 else 2  # Với ảnh lớn, bỏ qua một số pixel
    
    for i in range(0, h, step):
        for j in range(0, w, step):
            # Xác định vùng lân cận
            i_min = max(i - radius, 0)
            i_max = min(i + radius + 1, h)
            j_min = max(j - radius, 0)
            j_max = min(j + radius + 1, w)
            
            # Lấy vùng lân cận
            region = image[i_min:i_max, j_min:j_max].astype(np.float32)
            center_pixel = float(image[i, j])
            
            # Tính kernel màu (color/intensity kernel)
            # Dựa trên độ khác biệt cường độ
            color_diff = np.abs(region - center_pixel)
            color_kernel = np.exp(-0.5 * (color_diff / sigma_color) ** 2)
            
            # Lấy phần kernel không gian tương ứng
            kernel_i_min = radius - (i - i_min)
            kernel_i_max = radius + (i_max - i)
            kernel_j_min = radius - (j - j_min)
            kernel_j_max = radius + (j_max - j)
            
            space_kernel_region = space_kernel[
                kernel_i_min:kernel_i_max,
                kernel_j_min:kernel_j_max
            ]
            
            # Kết hợp kernel không gian và màu
            kernel_combined = space_kernel_region * color_kernel
            kernel_sum = np.sum(kernel_combined)
            
            # Tính giá trị mới
            if kernel_sum > 1e-6:  # Tránh chia cho 0
                result[i, j] = np.sum(region * kernel_combined) / kernel_sum
            else:
                result[i, j] = center_pixel
    
    # Nếu đã skip pixel, cần interpolate
    if step > 1:
        result = interpolate_missing_pixels(result, step)
    
    return result


def bilateral_filter_rgb(image: np.ndarray, radius: int,
                        space_kernel: np.ndarray, sigma_color: float,
                        sigma_space: float) -> np.ndarray:

    h, w, c = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    
    # Tối ưu: xử lý với step
    step = 1 if min(h, w) < 500 else 2
    
    for i in range(0, h, step):
        for j in range(0, w, step):
            # Xác định vùng lân cận
            i_min = max(i - radius, 0)
            i_max = min(i + radius + 1, h)
            j_min = max(j - radius, 0)
            j_max = min(j + radius + 1, w)
            
            # Lấy vùng lân cận
            region = image[i_min:i_max, j_min:j_max].astype(np.float32)
            center_pixel = image[i, j].astype(np.float32)
            
            # Tính kernel màu dựa trên khoảng cách Euclidean trong không gian RGB
            color_diff = np.linalg.norm(region - center_pixel, axis=2)
            color_kernel = np.exp(-0.5 * (color_diff / sigma_color) ** 2)
            
            # Lấy phần kernel không gian tương ứng
            kernel_i_min = radius - (i - i_min)
            kernel_i_max = radius + (i_max - i)
            kernel_j_min = radius - (j - j_min)
            kernel_j_max = radius + (j_max - j)
            
            space_kernel_region = space_kernel[
                kernel_i_min:kernel_i_max,
                kernel_j_min:kernel_j_max
            ]
            
            # Kết hợp kernel
            kernel_combined = space_kernel_region * color_kernel
            kernel_sum = np.sum(kernel_combined)
            
            # Tính giá trị mới cho từng kênh màu
            if kernel_sum > 1e-6:
                for ch in range(c):
                    result[i, j, ch] = np.sum(region[:, :, ch] * kernel_combined) / kernel_sum
            else:
                result[i, j] = center_pixel
    
    # Nếu đã skip pixel, cần interpolate
    if step > 1:
        for ch in range(c):
            result[:, :, ch] = interpolate_missing_pixels(result[:, :, ch], step)
    
    return result


def interpolate_missing_pixels(image: np.ndarray, step: int) -> np.ndarray:

    h, w = image.shape
    result = image.copy()
    
    # Nội suy các hàng
    for i in range(h):
        if i % step != 0:
            i_prev = (i // step) * step
            i_next = min(i_prev + step, h - 1)
            alpha = (i - i_prev) / step
            result[i, :] = (1 - alpha) * result[i_prev, :] + alpha * result[i_next, :]
    
    # Nội suy các cột
    for j in range(w):
        if j % step != 0:
            j_prev = (j // step) * step
            j_next = min(j_prev + step, w - 1)
            alpha = (j - j_prev) / step
            result[:, j] = (1 - alpha) * result[:, j_prev] + alpha * result[:, j_next]
    
    return result


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


def median_filter(image: np.ndarray, size: int = 5) -> np.ndarray:
    
    if size % 2 == 0:
        size += 1
    
    radius = size // 2
    h, w = image.shape[:2]
    
    # Padding
    if image.ndim == 3:
        padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
        result = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                for c in range(image.shape[2]):
                    region = padded[i:i+size, j:j+size, c]
                    result[i, j, c] = np.median(region)
    else:
        padded = np.pad(image, radius, mode='reflect')
        result = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                region = padded[i:i+size, j:j+size]
                result[i, j] = np.median(region)
    
    return result.astype(np.uint8)