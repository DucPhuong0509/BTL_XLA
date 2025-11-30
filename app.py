# app.py - Phần mềm chuyển ảnh thành hình vẽ tay
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import threading
import time


class SketchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sketch Artist - Chuyển ảnh thành tranh vẽ tay")
        self.geometry("1300x800")
        self.minsize(1100, 700)
        self.configure(bg="#2c3e50")
        
        # Biến lưu ảnh
        self.original_image = None
        self.sketch_image = None
        self.is_processing = False
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10, "bold"), padding=6)
        self.style.configure("TScale", background="#34495e")
        self.style.configure("Custom.Horizontal.TProgressbar", 
                           thickness=12, background="#3498db")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self, bg="#1a252f", height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="SKETCH ARTIST", 
                              font=("Arial", 24, "bold"), 
                              fg="white", bg="#1a252f")
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        subtitle_label = tk.Label(header_frame, 
                                 text="Biến ảnh thường thành tác phẩm nghệ thuật",
                                 font=("Arial", 11), 
                                 fg="#bdc3c7", bg="#1a252f")
        subtitle_label.pack(side=tk.LEFT, padx=10, pady=20)
        
        # Main content area
        main_container = tk.Frame(self, bg="#2c3e50")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        control_panel = tk.Frame(main_container, bg="#34495e", width=300, 
                                relief="raised", bd=1)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        control_panel.pack_propagate(False)
        
        # Control panel content
        control_title = tk.Label(control_panel, text="ĐIỀU KHIỂN", 
                               font=("Arial", 12, "bold"), 
                               fg="white", bg="#1a252f")
        control_title.pack(fill=tk.X, padx=10, pady=15)
        
        # Action buttons
        btn_style = {"font": ("Arial", 10, "bold"), "width": 20, "height": 2}
        
        tk.Button(control_panel, text="TẢI ẢNH", command=self.load_image,
                 bg="#27ae60", fg="white", **btn_style).pack(pady=8)
        
        tk.Button(control_panel, text="TẠO TRANH VẼ", command=self.process_image,
                 bg="#2980b9", fg="white", **btn_style).pack(pady=8)
        
        tk.Button(control_panel, text="LƯU TRANH", command=self.save_image,
                 bg="#e67e22", fg="white", **btn_style).pack(pady=8)
        
        # Separator
        sep1 = tk.Frame(control_panel, height=2, bg="#1a252f")
        sep1.pack(fill=tk.X, padx=10, pady=20)
        
        # Sketch style selection
        style_frame = tk.Frame(control_panel, bg="#34495e")
        style_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(style_frame, text="PHONG CÁCH VẼ:", 
                font=("Arial", 10, "bold"), 
                fg="white", bg="#34495e").pack(anchor="w")
        
        self.sketch_style = tk.StringVar(value="hand_drawn")
        
        style_options = [
            ("Vẽ tay thường - Mềm mại", "hand_drawn"),
            ("Vẽ bút chì - Tương phản cao", "pencil")
        ]
        
        for text, value in style_options:
            rb = tk.Radiobutton(style_frame, text=text, variable=self.sketch_style,
                               value=value, font=("Arial", 9),
                               fg="white", bg="#34495e", selectcolor="#2c3e50",
                               activebackground="#34495e", anchor="w")
            rb.pack(fill=tk.X, pady=3)
        
        # Style descriptions
        desc_frame = tk.Frame(control_panel, bg="#2c3e50", relief="sunken", bd=1)
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.style_desc = tk.Label(desc_frame, 
                                  text="Vẽ tay: Đường nét mềm mại, tự nhiên\nBút chì: Đậm nét, tương phản cao", 
                                  font=("Arial", 8), fg="#bdc3c7", bg="#2c3e50",
                                  justify=tk.LEFT)
        self.style_desc.pack(padx=5, pady=5)
        
        # Parameters
        params_frame = tk.Frame(control_panel, bg="#34495e")
        params_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Edge strength
        edge_frame = tk.Frame(params_frame, bg="#34495e")
        edge_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(edge_frame, text="ĐỘ RÕ NÉT:", 
                font=("Arial", 9, "bold"), 
                fg="white", bg="#34495e").pack(anchor="w")
        
        self.edge_strength = tk.IntVar(value=80)
        edge_scale = ttk.Scale(edge_frame, from_=30, to=150, 
                              variable=self.edge_strength,
                              orient=tk.HORIZONTAL)
        edge_scale.pack(fill=tk.X, pady=5)
        
        edge_value_frame = tk.Frame(edge_frame, bg="#34495e")
        edge_value_frame.pack(fill=tk.X)
        
        tk.Label(edge_value_frame, text="Mờ", font=("Arial", 8),
                fg="#bdc3c7", bg="#34495e").pack(side=tk.LEFT)
        self.edge_label = tk.Label(edge_value_frame, text="80%", 
                                  font=("Arial", 9, "bold"), fg="white", bg="#34495e")
        self.edge_label.pack(side=tk.RIGHT)
        tk.Label(edge_value_frame, text="Rõ", font=("Arial", 8),
                fg="#bdc3c7", bg="#34495e").pack(side=tk.RIGHT)
        
        # Smoothness
        smooth_frame = tk.Frame(params_frame, bg="#34495e")
        smooth_frame.pack(fill=tk.X, pady=8)
        
        tk.Label(smooth_frame, text="ĐỘ MỊN:", 
                font=("Arial", 9, "bold"), 
                fg="white", bg="#34495e").pack(anchor="w")
        
        self.smoothness = tk.IntVar(value=8)
        smooth_scale = ttk.Scale(smooth_frame, from_=3, to=15, 
                                variable=self.smoothness,
                                orient=tk.HORIZONTAL)
        smooth_scale.pack(fill=tk.X, pady=5)
        
        smooth_value_frame = tk.Frame(smooth_frame, bg="#34495e")
        smooth_value_frame.pack(fill=tk.X)
        
        tk.Label(smooth_value_frame, text="Chi tiết", font=("Arial", 8),
                fg="#bdc3c7", bg="#34495e").pack(side=tk.LEFT)
        self.smooth_label = tk.Label(smooth_value_frame, text="8", 
                                    font=("Arial", 9, "bold"), fg="white", bg="#34495e")
        self.smooth_label.pack(side=tk.RIGHT)
        tk.Label(smooth_value_frame, text="Mịn", font=("Arial", 8),
                fg="#bdc3c7", bg="#34495e").pack(side=tk.RIGHT)
        
        # Progress section
        progress_frame = tk.Frame(control_panel, bg="#34495e")
        progress_frame.pack(fill=tk.X, padx=15, pady=20)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                           variable=self.progress_var,
                                           maximum=100,
                                           style="Custom.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_text = tk.Label(progress_frame, text="SẴN SÀNG", 
                                     font=("Arial", 9, "bold"),
                                     fg="#2ecc71", bg="#34495e")
        self.progress_text.pack()
        
        # Right panel - Image display
        display_panel = tk.Frame(main_container, bg="#2c3e50")
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display frames
        display_container = tk.Frame(display_panel, bg="#2c3e50")
        display_container.pack(fill=tk.BOTH, expand=True)
        
        # Original image frame
        orig_frame = tk.LabelFrame(display_container, text="  ẢNH GỐC  ", 
                                  font=("Arial", 11, "bold"),
                                  fg="#ecf0f1", bg="#2c3e50", 
                                  labelanchor="n", bd=2, relief="groove")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.canvas_original = tk.Canvas(orig_frame, bg="#34495e", 
                                        highlightthickness=0)
        self.canvas_original.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sketch image frame
        sketch_frame = tk.LabelFrame(display_container, text="  TRANH VẼ TAY  ", 
                                    font=("Arial", 11, "bold"),
                                    fg="#ecf0f1", bg="#2c3e50", 
                                    labelanchor="n", bd=2, relief="groove")
        sketch_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.canvas_sketch = tk.Canvas(sketch_frame, bg="#34495e", 
                                      highlightthickness=0)
        self.canvas_sketch.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind scale events
        edge_scale.configure(command=self.update_edge_label)
        smooth_scale.configure(command=self.update_smooth_label)
        self.sketch_style.trace('w', self.update_style_description)
        
        # Display initial placeholders
        self.after(100, self.display_placeholders)
    
    def update_edge_label(self, value=None):
        """Cập nhật label độ rõ nét"""
        value = self.edge_strength.get()
        self.edge_label.config(text=f"{int(value)}%")
    
    def update_smooth_label(self, value=None):
        """Cập nhật label độ mịn"""
        value = self.smoothness.get()
        self.smooth_label.config(text=str(int(value)))
    
    def update_style_description(self, *args):
        """Cập nhật mô tả phong cách khi thay đổi"""
        style = self.sketch_style.get()
        if style == "pencil":
            desc = "Bút chì: Đường nét đậm, tương phản cao\nPhù hợp cho ảnh chân dung, kiến trúc"
        else:
            desc = "Vẽ tay: Đường nét mềm mại, tự nhiên\nPhù hợp cho phong cảnh, nghệ thuật"
        self.style_desc.config(text=desc)
    
    def update_progress(self, value: int, text: str = ""):
        """Cập nhật thanh tiến trình"""
        self.progress_var.set(value)
        if text:
            self.progress_text.config(text=text)
        
        # Đổi màu theo tiến độ
        if value < 30:
            color = "#e74c3c"
        elif value < 70:
            color = "#f39c12"
        else:
            color = "#2ecc71"
        
        self.progress_text.config(fg=color)
        self.update_idletasks()
    
    def display_placeholders(self):
        """Hiển thị placeholder ban đầu"""
        self.display_placeholder(self.canvas_original, 
                               "ẢNH GỐC\n\nNhấn 'TẢI ẢNH' để chọn\nmột bức ảnh đẹp", 
                               "#3498db")
        
        self.display_placeholder(self.canvas_sketch, 
                               "TRANH VẼ\n\nẢnh vẽ tay sẽ xuất hiện\nsau khi xử lý", 
                               "#9b59b6")
    
    def display_placeholder(self, canvas, text, color):
        """Hiển thị placeholder với text và màu"""
        canvas.delete("all")
        
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 10 or height <= 10:
            return
        
        # Vẽ vòng tròn nền
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        
        canvas.create_oval(center_x - radius, center_y - radius,
                          center_x + radius, center_y + radius,
                          fill=color, outline="", width=0)
        
        # Vẽ text
        canvas.create_text(center_x, center_y + radius + 50,
                          text=text, fill="white", 
                          font=("Arial", 12, "bold"),
                          justify=tk.CENTER)
    
    def load_image(self):
        """Tải ảnh từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh để biến thành tranh vẽ",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.update_progress(20, "ĐANG TẢI ẢNH...")
                self.original_image = Image.open(file_path).convert("RGB")
                self.sketch_image = None
                self.update_progress(100, "TẢI ẢNH THÀNH CÔNG")
                self.display_images()
                time.sleep(1)
                self.update_progress(0, "SẴN SÀNG")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh:\n{str(e)}")
                self.update_progress(0, "LỖI KHI TẢI ẢNH")
    
    def display_images(self):
        """Hiển thị ảnh gốc và ảnh sketch"""
        # Hiển thị ảnh gốc
        if self.original_image:
            self.display_image(self.canvas_original, self.original_image)
        else:
            self.display_placeholder(self.canvas_original,
                                   "ẢNH GỐC\n\nChưa có ảnh nào được tải",
                                   "#3498db")
        
        # Hiển thị ảnh sketch
        if self.sketch_image:
            self.display_image(self.canvas_sketch, self.sketch_image)
        else:
            self.display_placeholder(self.canvas_sketch,
                                   "TRANH VẼ\n\nNhấn 'TẠO TRANH VẼ'\nđể bắt đầu xử lý",
                                   "#9b59b6")
    
    def display_image(self, canvas, image):
        """Hiển thị ảnh trên canvas với scaling phù hợp"""
        canvas.delete("all")
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Tính tỷ lệ scaling
        img_width, img_height = image.size
        ratio = min((canvas_width - 40) / img_width, 
                   (canvas_height - 40) / img_height)
        
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize ảnh
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized_image)
        
        # Hiển thị ảnh
        canvas.image = photo
        canvas.create_image(canvas_width // 2, canvas_height // 2, 
                          image=photo, anchor=tk.CENTER)
    
    def process_image_thread(self):
        """Xử lý ảnh trong thread riêng"""
        try:
            self.update_progress(10, "BẮT ĐẦU TẠO TRANH VẼ...")
            
            # Chuyển PIL Image sang numpy array
            img_array = np.array(self.original_image)
            self.update_progress(25, "ĐANG XỬ LÝ ẢNH...")
            
            # Progress callback
            def progress_callback(step, total_steps, message):
                progress = 25 + (step / total_steps) * 60
                self.update_progress(progress, message)
            
            # Import và xử lý ảnh
            from sketch import create_hand_drawn_sketch, pencil_sketch
            
            if self.sketch_style.get() == "pencil":
                self.update_progress(50, "ĐANG VẼ BÚT CHÌ...")
                result_array = pencil_sketch(img_array)
            else:
                result_array = create_hand_drawn_sketch(
                    img_array,
                    diameter=self.smoothness.get(),
                    edge_strength=self.edge_strength.get() / 100.0,
                    progress_callback=progress_callback
                )
            
            self.update_progress(90, "ĐANG HOÀN THIỆN...")
            
            # Chuyển kết quả sang PIL Image
            self.sketch_image = Image.fromarray(result_array)
            
            self.update_progress(100, "TẠO TRANH THÀNH CÔNG")
            
            # Hiển thị kết quả
            self.after(0, self.display_images)
            self.after(1500, lambda: self.update_progress(0, "SẴN SÀNG"))
            
        except Exception as e:
            error_msg = f"Lỗi khi tạo tranh vẽ:\n{str(e)}"
            self.after(0, lambda: messagebox.showerror("Lỗi", error_msg))
            self.after(0, lambda: self.update_progress(0, "LỖI KHI XỬ LÝ"))
        finally:
            self.is_processing = False
    
    def process_image(self):
        """Bắt đầu xử lý ảnh"""
        if self.original_image is None:
            messagebox.showwarning("Cảnh báo", 
                "Vui lòng tải ảnh trước khi tạo tranh vẽ!")
            return
        
        if self.is_processing:
            messagebox.showwarning("Cảnh báo", 
                "Đang xử lý ảnh, vui lòng chờ...")
            return
        
        # Hiển thị thông báo đang xử lý
        self.display_placeholder(self.canvas_sketch,
                               "ĐANG XỬ LÝ...\n\nVui lòng chờ trong giây lát",
                               "#f39c12")
        
        self.is_processing = True
        
        # Chạy xử lý trong thread riêng
        thread = threading.Thread(target=self.process_image_thread)
        thread.daemon = True
        thread.start()
    
    def save_image(self):
        """Lưu ảnh sketch"""
        if self.sketch_image is None:
            messagebox.showwarning("Cảnh báo", 
                "Chưa có tranh vẽ nào để lưu!\nHãy tạo tranh vẽ trước.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Lưu tranh vẽ của bạn",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"), 
                ("JPEG files", "*.jpg"), 
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.update_progress(50, "ĐANG LƯU TRANH...")
                self.sketch_image.save(file_path, quality=95)
                self.update_progress(100, "LƯU TRANH THÀNH CÔNG")
                messagebox.showinfo("Thành công", 
                    f"Đã lưu tranh vẽ thành công!\n\n{file_path}")
                time.sleep(1)
                self.update_progress(0, "SẴN SÀNG")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu tranh:\n{str(e)}")
                self.update_progress(0, "LỖI KHI LƯU")


if __name__ == "__main__":
    app = SketchApp()
    app.mainloop()