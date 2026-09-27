# 🛒 SKU Recognition System — Nhận Diện Sản Phẩm Trên Kệ Hàng

Hệ thống **SKU Recognition** là một ứng dụng Computer Vision dùng để **phát hiện, nhận diện và thống kê sản phẩm trên kệ hàng từ hình ảnh**.

Hệ thống được xây dựng theo pipeline gồm hai giai đoạn chính:

**RetinaNet → Object Detection → Crop sản phẩm → MobileNetV3 → Feature Embedding → FAISS → SKU Recognition → Thống kê**

Backend được triển khai bằng **FastAPI**, kết hợp với giao diện Web cho phép người dùng tải ảnh lên, phân tích và trực quan hóa kết quả.

---

## Đường link các checkpoints : https://drive.google.com/drive/folders/1jGdX0p8tIiO8To0tW5-vhyTF8PwNtbjW?usp=drive_link

## ✨ 1. Tính năng chính

* 📤 Upload ảnh kệ hàng.
* 🔍 Phát hiện các sản phẩm bằng **RetinaNet**.
* 📦 Tự động crop từng vùng sản phẩm được phát hiện.
* 🧠 Trích xuất đặc trưng hình ảnh bằng **MobileNetV3**.
* 🔎 Nhận diện SKU bằng **FAISS** và Cosine Similarity.
* 🗳️ Majority Voting với Top-K kết quả để tăng độ ổn định khi nhận diện.
* 📊 Thống kê số lượng sản phẩm theo từng SKU.
* 🖼️ Hiển thị bounding box trực tiếp trên ảnh.
* ⚙️ Cho phép điều chỉnh ngưỡng confidence/recognition.
* 🚀 Cung cấp REST API thông qua FastAPI.
* 🧪 Có các script hỗ trợ debug và kiểm tra chất lượng Gallery.

---

# 🏗️ 2. Kiến trúc hệ thống

Pipeline xử lý của hệ thống:

```text
                    Input Image
                         │
                         ▼
                ┌─────────────────┐
                │    RetinaNet    │
                │ Object Detection│
                └────────┬────────┘
                         │
                  Bounding Boxes
                         │
                         ▼
                 Crop từng sản phẩm
                         │
                         ▼
                ┌─────────────────┐
                │   MobileNetV3   │
                │ Feature Extractor│
                └────────┬────────┘
                         │
                   Feature Vector
                         │
                         ▼
                ┌─────────────────┐
                │      FAISS      │
                │ Similarity Search│
                └────────┬────────┘
                         │
                    Top-K Results
                         │
                         ▼
                Majority Voting
                         │
                         ▼
                  SKU Recognition
                         │
                         ▼
              ┌─────────────────────┐
              │ Detection + Label   │
              │ + Statistics        │
              └─────────────────────┘
```

### Các thành phần chính

| Thành phần              | Vai trò                              |
| ----------------------- | ------------------------------------ |
| **RetinaNet**           | Phát hiện vị trí sản phẩm            |
| **MobileNetV3**         | Trích xuất vector đặc trưng          |
| **FAISS**               | Tìm kiếm các sản phẩm tương đồng     |
| **Cosine Similarity**   | Đo mức độ tương đồng giữa các vector |
| **Majority Voting**     | Chọn SKU dựa trên các kết quả Top-K  |
| **FastAPI**             | Cung cấp Backend API                 |
| **HTML/CSS/JavaScript** | Xây dựng giao diện Web               |

---

# 📁 3. Cấu trúc thư mục

```text
SKU_project/
│
├── backend/
│   ├── main.py
│   │
│   └── api/
│       ├── config.py
│       └── search.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── src/
│   ├── debug_recognition.py
│   ├── audit_gallery.py
│   └── clean_gallery.py
│
├── checkpoints/
│   ├── retinanet/
│   └── mobilenetv3/
│
├── embeddings/
│   └── gallery_embeddings.npz
│
├── faiss_index/
│   └── shape_gallery.index
│
├── data/
│   └── ...
│
└── README.md
```

> Nếu tên thư mục frontend trong project hiện tại là `fontend`, nên đổi thành `frontend` để tránh nhầm lẫn và tuân theo cách đặt tên phổ biến.

---

# ⚙️ 4. Cài đặt

## 4.1. Yêu cầu

Khuyến nghị sử dụng:

* Python 3.10+
* Windows / Linux
* Virtual Environment hoặc Conda
* GPU NVIDIA nếu muốn tăng tốc inference

## 4.2. Cài đặt thư viện

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt trên Windows:

```powershell
.venv\Scripts\activate
```

Cài đặt các thư viện:

```bash
pip install torch torchvision
pip install numpy opencv-python
pip install fastapi uvicorn
pip install faiss-cpu
```

Nếu hệ thống sử dụng GPU và môi trường FAISS tương thích:

```bash
pip install faiss-gpu
```

---

# 🚀 5. Khởi chạy hệ thống

## 5.1. Chạy Backend

Mở Terminal tại thư mục gốc của project:

```text
E:/SKU_project
```

Sau đó chạy:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Nếu server khởi động thành công, API sẽ hoạt động tại:

```text
http://localhost:8000
```

Swagger API Documentation:

```text
http://localhost:8000/docs
```

---

# 🖥️ 6. Frontend

Mở file:

```text
frontend/index.html
```

bằng trình duyệt như:

* Google Chrome
* Microsoft Edge
* Cốc Cốc

Frontend sẽ gửi request phân tích ảnh tới:

```text
http://localhost:8000/analyze
```

### Quy trình sử dụng

```text
Upload ảnh
    ↓
Nhấn Analyze
    ↓
Frontend gửi ảnh tới FastAPI
    ↓
RetinaNet phát hiện sản phẩm
    ↓
Crop từng bounding box
    ↓
MobileNetV3 tạo embedding
    ↓
FAISS tìm sản phẩm tương đồng
    ↓
Majority Voting xác định SKU
    ↓
Trả kết quả về Frontend
    ↓
Hiển thị bounding box + SKU + thống kê
```

---

# 🔎 7. Recognition bằng FAISS

Sau khi RetinaNet phát hiện sản phẩm, mỗi bounding box sẽ được crop ra thành một ảnh riêng.

Ví dụ:

```text
Input Image
     │
     ├── Product 1
     ├── Product 2
     ├── Product 3
     └── Product 4
```

Mỗi crop được đưa qua MobileNetV3 để tạo một vector đặc trưng:

```text
Product Crop
     ↓
MobileNetV3
     ↓
Feature Vector
     ↓
FAISS Search
```

FAISS sau đó tìm những vector trong Gallery có độ tương đồng cao nhất.

---

# 🗳️ 8. Majority Voting

Hệ thống không nhất thiết lấy ngay kết quả Top-1.

Thay vào đó, hệ thống lấy **Top-K kết quả gần nhất** và sử dụng Majority Voting để giảm ảnh hưởng của những mẫu Gallery bị nhiễu.

Ví dụ:

```text
Top-5 FAISS results:

SKU_001
SKU_001
SKU_003
SKU_001
SKU_002
```

Kết quả:

```text
Predicted SKU = SKU_001
```

Với cấu hình mặc định:

```python
TOP_K = 5
```

---

# ⚙️ 9. Cấu hình Recognition

Các thông số chính được đặt trong:

```text
backend/api/config.py
```

Ví dụ:

```python
RECOGNIZE_THRESHOLD = 0.85
TOP_K = 5
```

### `RECOGNIZE_THRESHOLD`

Ngưỡng similarity tối thiểu để chấp nhận kết quả nhận diện.

```text
Similarity >= 0.85
        ↓
   Accept SKU

Similarity < 0.85
        ↓
   Reject / Unknown
```

Ngưỡng này có thể được điều chỉnh tùy thuộc vào chất lượng Gallery và đặc trưng của dữ liệu.

### `TOP_K`

Số lượng kết quả gần nhất được lấy từ FAISS để thực hiện Majority Voting.

Ví dụ:

```python
TOP_K = 5
```

có nghĩa là hệ thống lấy 5 kết quả gần nhất.

---

# 🧪 10. Công cụ Debug và quản lý Gallery

Thư mục:

```text
src/
```

chứa các script hỗ trợ kiểm tra và cải thiện hệ thống Recognition.

---

## 10.1. Debug Recognition

File:

```text
src/debug_recognition.py
```

Dùng để kiểm tra tại sao một sản phẩm bị nhận diện sai hoặc một sản phẩm bị phân thành nhiều SKU khác nhau.

### Debug một ảnh

```bash
python src/debug_recognition.py --image data/anh_cua_ban.jpg
```

### Debug toàn bộ thư mục

```bash
python src/debug_recognition.py --image_dir data/
```

Script sẽ hiển thị các kết quả Top-K để quan sát mức độ tương đồng giữa sản phẩm cần nhận diện và Gallery.

Ví dụ:

```text
Query Product
      │
      ├── SKU_001 : 0.91
      ├── SKU_003 : 0.89
      ├── SKU_001 : 0.88
      ├── SKU_002 : 0.84
      └── SKU_001 : 0.82
```

Qua đó có thể xác định nguyên nhân nhận diện sai hoặc Gallery có dữ liệu gây nhiễu.

---

# 🔬 10.2. Audit Gallery

File:

```text
src/audit_gallery.py
```

Dùng để kiểm tra chất lượng Gallery Embeddings.

Chạy:

```bash
python src/audit_gallery.py
```

Script hỗ trợ phát hiện các vấn đề như:

* Embedding bất thường.
* Mẫu dữ liệu có khả năng là outlier.
* Các mẫu thuộc những SKU khác nhau nhưng có độ tương đồng quá cao.
* Gallery có dữ liệu gây nhầm lẫn giữa các SKU.

---

# 🧹 10.3. Clean Gallery

File:

```text
src/clean_gallery.py
```

Dùng để loại bỏ những mẫu không phù hợp khỏi Gallery.

Chạy:

```bash
python src/clean_gallery.py
```

Sau khi xử lý, hệ thống có thể tạo:

```text
embeddings/
├── gallery_embeddings.npz
└── clean_gallery_embeddings.npz

faiss_index/
└── shape_gallery.index
```

Gallery sạch được sử dụng để build lại FAISS Index, giúp quá trình tìm kiếm ổn định hơn.

---

# 🖼️ 11. Gallery Embeddings

Gallery là tập ảnh mẫu dùng làm cơ sở để nhận diện SKU.

Mỗi SKU nên có nhiều ảnh đại diện, ví dụ:

```text
SKU_001/
├── image_01.jpg
├── image_02.jpg
├── image_03.jpg
├── image_04.jpg
└── image_05.jpg
```

Các ảnh này được đưa qua MobileNetV3:

```text
Gallery Image
      ↓
MobileNetV3
      ↓
Embedding Vector
      ↓
gallery_embeddings.npz
      ↓
FAISS Index
```

---

# 📌 12. Thêm SKU mới

Khi thêm một sản phẩm mới, nên chuẩn bị nhiều ảnh crop chất lượng tốt.

Khuyến nghị tối thiểu:

```text
4–5 ảnh / SKU
```

Nên bao quát sự thay đổi về:

* Góc nhìn.
* Khoảng cách.
* Ánh sáng.
* Vị trí trên kệ.
* Kích thước sản phẩm.
* Một số biến đổi thực tế khác.

Ví dụ:

```text
SKU_001
│
├── front.jpg
├── left.jpg
├── right.jpg
├── bright.jpg
└── shelf.jpg
```

Số lượng ảnh nhiều hơn và đa dạng hơn có thể giúp Gallery đại diện tốt hơn cho cùng một SKU, nhưng chất lượng và tính phù hợp của ảnh vẫn rất quan trọng.

---

# 🔄 13. Cập nhật FAISS Index

Sau khi cập nhật Gallery Embeddings, cần build lại FAISS Index để hệ thống nhận diện được dữ liệu mới.

Quy trình:

```text
Thêm ảnh SKU mới
       ↓
Generate Embeddings
       ↓
Update Gallery Embeddings
       ↓
Clean / Audit Gallery
       ↓
Build FAISS Index
       ↓
Restart Backend
       ↓
Test Recognition
```

---

# 📊 14. Kết quả đầu ra

Sau khi phân tích ảnh, hệ thống trả về thông tin của từng sản phẩm được phát hiện.

Ví dụ:

```text
Input:
Shelf Image
```

Kết quả:

```text
┌─────────────────────────────────┐
│ Product Detection               │
│                                 │
│ SKU_001    × 3                  │
│ SKU_002    × 2                  │
│ SKU_005    × 1                  │
│                                 │
│ Total Products: 6               │
└─────────────────────────────────┘
```

Trên ảnh kết quả, mỗi sản phẩm được hiển thị bằng bounding box kèm nhãn SKU.

---

# 🧩 15. API

Endpoint chính:

```http
POST /analyze
```

### Input

Ảnh sản phẩm/kệ hàng được upload lên API.

### Processing

```text
Image
 ↓
RetinaNet
 ↓
Bounding Boxes
 ↓
MobileNetV3
 ↓
FAISS
 ↓
Majority Voting
```

### Output

Thông tin nhận diện và thống kê sản phẩm, bao gồm các bounding box, SKU được nhận diện và số lượng theo từng SKU.

---

# 🎯 16. Công nghệ sử dụng

### Computer Vision

* PyTorch
* Torchvision
* RetinaNet
* MobileNetV3
* OpenCV

### Similarity Search

* FAISS
* Cosine Similarity
* Embedding-based Retrieval
* Majority Voting

### Backend

* Python
* FastAPI
* Uvicorn

### Frontend

* HTML
* CSS
* JavaScript

### Data

* SKU/Product Images
* Gallery Embeddings
* FAISS Index

---

# 📌 17. Tổng quan hệ thống

Toàn bộ hệ thống có thể được mô tả ngắn gọn như sau:

```text
                 ┌──────────────┐
                 │   User       │
                 └──────┬───────┘
                        │
                    Upload Image
                        │
                        ▼
              ┌───────────────────┐
              │     FastAPI       │
              │      Backend      │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │     RetinaNet     │
              │  Object Detection │
              └─────────┬─────────┘
                        │
                  Product Crops
                        │
                        ▼
              ┌───────────────────┐
              │    MobileNetV3    │
              │ Feature Extraction│
              └─────────┬─────────┘
                        │
                   Embeddings
                        │
                        ▼
              ┌───────────────────┐
              │       FAISS       │
              │ Similarity Search │
              └─────────┬─────────┘
                        │
                     Top-K
                        │
                        ▼
              ┌───────────────────┐
              │ Majority Voting   │
              │   SKU Recognition │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Statistics + UI   │
              │ Bounding Boxes    │
              └───────────────────┘
```

---

## 🚀 18. Mục tiêu của dự án

Dự án hướng tới việc xây dựng một pipeline Computer Vision hoàn chỉnh cho bài toán **Product Detection & SKU Recognition**, kết hợp:

**Object Detection + Feature Extraction + Vector Similarity Search + Backend API + Web Interface**

Qua đó hệ thống có thể được mở rộng cho các bài toán thực tế như:

* Quản lý hàng hóa trên kệ.
* Kiểm kê sản phẩm tự động.
* Nhận diện SKU.
* Phân tích hình ảnh kệ hàng.
* Hỗ trợ quản lý tồn kho.
* Xây dựng hệ thống Retail Computer Vision.

