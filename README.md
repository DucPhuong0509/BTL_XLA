# Chuyển Ảnh Thành Hình Vẽ Tay (Colored Pencil Sketch Effect)

Phần mềm chuyển đổi ảnh thành hình vẽ tay kiểu bút chì cực kỳ chân thực.

### Tính năng nổi bật
- Hiệu ứng colored pencil sketch.
- Giao diện Tkinter đơn giản, đẹp, có slider điều chỉnh thông số realtime  
- Xem trước ảnh gốc và kết quả cạnh nhau  
- Lưu kết quả PNG/JPG  
- Tốc độ nhanh.

### Cấu trúc thư mục
photo-to-sketch/
│
├── app.py              # File chính, giao diện Tkinter
├── sketch.py           # Hàm tạo hiệu ứng vẽ tay chính
├── filter.py           # Gaussian blur + Bilateral filter
├── grayscale.py        # Chuyển ảnh màu thành ảnh xám 
├── edge_detector.py    # Phát hiện biên
├── README.md           
└── requirements.txt


### Yêu cầu hệ thống

- Python 3.7 trở lên
- Chỉ cần 3 thư viện:
  - numpy
  - Pillow
  - scipy

### Cài đặt

```bash
# Tạo môi trường ảo (khuyến khích)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install numpy pillow scipy

# Hoặc dùng file requirements.txt có sẵn
pip install -r requirements.txt

#Cách chạy

python app.py
