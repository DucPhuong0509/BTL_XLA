# grayscale.py - Chuyển đổi ảnh màu sang ảnh xám
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh màu sang ảnh xám sử dụng công thức luminance chuẩn.
    
    Công thức: Gray = 0.299*R + 0.587*G + 0.114*B
    
    Đây là công thức chuẩn ITU-R BT.601 - phù hợp với cách mắt người nhìn màu sắc.
    
    Args:
        image: Ảnh màu RGB hoặc ảnh xám
    
    Returns:
        Ảnh xám (grayscale) kiểu uint8
    
    Examples:
        >>> gray = to_grayscale(rgb_image)
        >>> print(gray.shape)  # (height, width)
    """
    # Nếu đã là ảnh xám, trả về bản sao
    if image.ndim == 2:
        return image.copy()
    
    # Nếu là ảnh màu RGB (3 kênh)
    if image.ndim == 3 and image.shape[2] == 3:
        # Sử dụng trọng số chuẩn: Red=0.299, Green=0.587, Blue=0.114
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    
    # Nếu là ảnh RGBA (4 kênh), bỏ qua kênh alpha
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    
    # Trường hợp khác (không hợp lệ)
    else:
        raise ValueError(f"Ảnh không hợp lệ. Shape: {image.shape}, cần (H,W) hoặc (H,W,3) hoặc (H,W,4)")
    
    # Đảm bảo giá trị trong khoảng [0, 255]
    return np.clip(gray, 0, 255).astype(np.uint8)


def weighted_grayscale(image: np.ndarray, weights: list = None) -> np.ndarray:
    """
    Chuyển ảnh xám với trọng số tùy chỉnh (nếu cần thử nghiệm).
    
    Args:
        image: Ảnh màu RGB
        weights: Danh sách trọng số [R, G, B], mặc định [0.299, 0.587, 0.114]
    
    Returns:
        Ảnh xám
    
    Examples:
        >>> # Tăng ảnh hưởng kênh đỏ (cho ảnh sketch đặc biệt)
        >>> gray = weighted_grayscale(image, weights=[0.5, 0.3, 0.2])
    """
    # Nếu không có weights, dùng chuẩn
    if weights is None:
        weights = [0.299, 0.587, 0.114]
    
    # Kiểm tra weights hợp lệ
    if len(weights) != 3:
        raise ValueError("weights phải có đúng 3 giá trị [R, G, B]")
    
    # Nếu đã là ảnh xám
    if image.ndim == 2:
        return image.copy()
    
    # Chuyển đổi với weights
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = np.dot(image[..., :3], weights)
    else:
        raise ValueError(f"Ảnh không hợp lệ. Shape: {image.shape}")
    
    return np.clip(gray, 0, 255).astype(np.uint8)