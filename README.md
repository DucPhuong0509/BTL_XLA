# Chuyển Ảnh Thành Hình Vẽ Tay (Colored Pencil Sketch Effect)

Phần mềm chuyển đổi ảnh màu thành hình vẽ tay kiểu bút chì màu cực kỳ chân thực, hoàn toàn tự viết thuật toán xử lý ảnh (không dùng OpenCV).

### Tính năng nổi bật
- Hiệu ứng colored pencil sketch đẹp nhất có thể đạt được chỉ bằng NumPy + Pillow  
- Bilateral filter tự viết (bảo toàn biên cực tốt)  
- Pencil shading texture tự tạo từ dodge/blend kỹ thuật cổ điển  
- Giao diện Tkinter đơn giản, đẹp, có slider điều chỉnh thông số realtime  
- Xem trước ảnh gốc và kết quả cạnh nhau  
- Lưu kết quả PNG/JPG  
- Tốc độ nhanh (ảnh 2000×2000 xử lý dưới 10 giây trên máy thường)

### Cấu trúc thư mục
photo-to-sketch/
│
├── app.py              # File chính, giao diện Tkinter
├── sketch.py           # Hàm tạo hiệu ứng vẽ tay chính (gọi các module khác)
├── filter.py           # Gaussian blur + Bilateral filter tự viết
├── grayscale.py        # Chuyển ảnh màu → ảnh xám chuẩn ITU-R
├── edge_detector.py    # Sobel + Non-maximum suppression (hiện chưa dùng trong hiệu ứng chính nhưng giữ lại để mở rộng)
├── README.md           
└── requirements.txt


### Yêu cầu hệ thống

- Python 3.7 trở lên
- Chỉ cần 2 thư viện:
  - numpy
  - Pillow

### Cài đặt

```bash
# Tạo môi trường ảo (khuyến khích)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install numpy pillow

# Hoặc dùng file requirements.txt có sẵn
pip install -r requirements.txt

#Cách chạy
python app.py