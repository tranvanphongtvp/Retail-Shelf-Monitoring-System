# 🛒 Shelf Product Recognition — Nhận Diện Sản Phẩm Trên Kệ Hàng

Hệ thống **Shelf Product Recognition** là một ứng dụng Computer Vision được xây dựng để **phát hiện, nhận diện và thống kê các sản phẩm (SKU) trên kệ hàng từ hình ảnh**.

Hệ thống sử dụng pipeline gồm hai giai đoạn chính:

**RetinaNet → Object Detection → Crop sản phẩm → MobileNetV3 → Feature Embedding → FAISS → SKU Recognition → Statistics**

Backend được triển khai bằng **FastAPI**, kết hợp với giao diện Web cho phép người dùng upload ảnh, phân tích và trực quan hóa kết quả nhận diện.

---

# 📌 1. Tổng quan hệ thống

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
                ┌────────────────────────┐
                │ Bounding Boxes + Labels│
                │ + Product Statistics   │
                └────────────────────────┘
```

---

# 🧠 2. Kiến trúc và công nghệ

| Thành phần         | Công nghệ                   | Chức năng                              |
| ------------------ | --------------------------- | -------------------------------------- |
| Object Detection   | **RetinaNet**               | Phát hiện vị trí sản phẩm trên kệ      |
| Feature Extraction | **MobileNetV3**             | Trích xuất đặc trưng hình ảnh sản phẩm |
| Similarity Search  | **FAISS**                   | Tìm kiếm sản phẩm tương đồng           |
| Similarity Metric  | **Cosine Similarity**       | Đánh giá độ tương đồng giữa embeddings |
| Recognition        | **Majority Voting**         | Xác định SKU từ Top-K kết quả          |
| Backend            | **FastAPI**                 | Cung cấp API phân tích ảnh             |
| Server             | **Uvicorn**                 | Chạy FastAPI                           |
| Frontend           | **HTML / CSS / JavaScript** | Giao diện người dùng                   |
| Image Processing   | **OpenCV**                  | Xử lý và crop ảnh                      |
| Deep Learning      | **PyTorch / Torchvision**   | Xây dựng và inference model            |

---

# 📊 3. Datasets

Project sử dụng **hai dataset cho hai giai đoạn khác nhau** trong pipeline.

## 3.1. Dataset cho Object Detection — SKU110K

**SKU110K** được sử dụng cho giai đoạn **phát hiện sản phẩm trên kệ hàng**.

Dataset chứa các hình ảnh kệ hàng với mật độ sản phẩm cao và bounding box cho các sản phẩm. Đây là dữ liệu phù hợp để huấn luyện RetinaNet học cách xác định vị trí sản phẩm.

### Mục đích sử dụng

```text
SKU110K
   ↓
Train RetinaNet
   ↓
Object Detection
   ↓
Bounding Boxes
```

### Dataset

📦 **SKU110K Fixed — Kaggle**

👉 [tại đây](https://www.kaggle.com/datasets/lordrovks/sku110k-fixed)

Dataset này được sử dụng cho phần **Detection** của hệ thống.

---

# 🛍️ 3.2. Dataset cho Product Recognition — SHAPE

Đối với giai đoạn **nhận diện SKU**, project sử dụng **SHAPE — SHelf mAnagement Product datasEt**, dataset sản phẩm được cung cấp bởi nhóm tác giả của bài báo:

**Shelf Management: A Deep Learning-Based System for Shelf Visual Monitoring**

Pipeline trong bài báo sử dụng **MobileNetV3 + FAISS** cho bài toán product recognition, tương tự hướng triển khai của project này.

### Mục đích sử dụng

```text
SHAPE Dataset
      ↓
Train MobileNetV3
      ↓
Feature Extraction
      ↓
Product Embeddings
      ↓
FAISS Gallery
      ↓
SKU Recognition
```

### Dataset SHAPE của tác giả

📦 **SHAPE — SHelf mAnagement Product datasEt**

👉 [tại đây](https://figshare.com/articles/dataset/SHAPE_-_SHelf_mAnagement_Product_datasEt/24100704)

Dataset được chia thành:

```text
training_set/
    ├── category_1/
    │      ├── EAN_1/
    │      ├── EAN_2/
    │      └── ...
    │
    ├── category_2/
    │      └── ...
    │
    └── ...

test_set/
    ├── category_1/
    ├── category_2/
    └── ...
```

Trong đó các thư mục EAN được sử dụng làm nhãn cho từng sản phẩm.

---

# 🔗 4. Checkpoints

Các checkpoint của project được lưu trữ trên Google Drive.

📦 **Model Checkpoints**

👉 [tại đây](https://drive.google.com/drive/folders/1jGdX0p8tIiO8To0tW5-vhyTF8PwNtbjW?usp=drive_link)

Các checkpoint có thể bao gồm:

```text
checkpoints/
│
├── RetinaNet
│
└── MobileNetV3
```

Sau khi tải checkpoint, đặt chúng vào thư mục tương ứng trong project.

---

# ✨ 5. Các tính năng chính

* 📤 Upload ảnh kệ hàng.
* 🔍 Phát hiện sản phẩm bằng **RetinaNet**.
* 📦 Tự động crop từng sản phẩm từ bounding box.
* 🧠 Trích xuất đặc trưng bằng **MobileNetV3**.
* 🔎 Nhận diện sản phẩm bằng **FAISS**.
* 📐 Sử dụng **Cosine Similarity** để tìm sản phẩm tương đồng.
* 🗳️ Majority Voting dựa trên Top-K kết quả.
* 📊 Thống kê số lượng từng SKU.
* 🖼️ Hiển thị bounding box và nhãn trực tiếp trên ảnh.
* ⚙️ Điều chỉnh ngưỡng recognition.
* 🚀 REST API với FastAPI.
* 🧪 Các công cụ debug và kiểm tra Gallery.

---

# 📁 6. Cấu trúc thư mục

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

---

# ⚙️ 7. Cài đặt môi trường

## 7.1. Yêu cầu

Khuyến nghị:

* Python 3.10+
* Windows / Linux
* Virtual Environment hoặc Conda
* GPU NVIDIA nếu muốn tăng tốc inference

---

## 7.2. Tạo Virtual Environment

```bash
python -m venv .venv
```

Kích hoạt trên Windows:

```powershell
.venv\Scripts\activate
```

---

## 7.3. Cài đặt thư viện

```bash
pip install torch torchvision
pip install numpy opencv-python
pip install fastapi uvicorn
pip install faiss-cpu
```

Nếu sử dụng GPU và môi trường hỗ trợ:

```bash
pip install faiss-gpu
```

---

# 🚀 8. Khởi chạy Backend

Mở Terminal tại thư mục gốc:

```text
E:/SKU_project
```

Sau đó chạy:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ hoạt động tại:

```text
http://localhost:8000
```

Swagger API Documentation:

```text
http://localhost:8000/docs
```

---

# 🖥️ 9. Khởi chạy Frontend

Mở:

```text
frontend/index.html
```

bằng:

* Google Chrome
* Microsoft Edge
* Cốc Cốc

Frontend sẽ gửi ảnh tới API:

```text
http://localhost:8000/analyze
```

---

# 🔄 10. Quy trình xử lý

```text
Upload Image
      ↓
Nhấn Analyze
      ↓
Frontend gửi ảnh tới FastAPI
      ↓
RetinaNet phát hiện sản phẩm
      ↓
Bounding Boxes
      ↓
Crop từng sản phẩm
      ↓
MobileNetV3 tạo Feature Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Results
      ↓
Majority Voting
      ↓
SKU Recognition
      ↓
Thống kê sản phẩm
      ↓
Hiển thị Bounding Boxes + Labels
```

---

# 🔎 11. Object Detection với RetinaNet

RetinaNet chịu trách nhiệm xác định vị trí các sản phẩm trong ảnh kệ hàng.

Input:

```text
Shelf Image
```

Output:

```text
Bounding Boxes
```

Ví dụ:

```text
Product 1 → [x1, y1, x2, y2]
Product 2 → [x1, y1, x2, y2]
Product 3 → [x1, y1, x2, y2]
```

Các bounding box sau đó được sử dụng để crop từng sản phẩm.

---

# 🧠 12. Product Recognition với MobileNetV3

Sau khi phát hiện sản phẩm, từng bounding box được crop:

```text
Shelf Image
     ↓
RetinaNet
     ↓
Bounding Box
     ↓
Product Crop
```

Product Crop được đưa vào MobileNetV3:

```text
Product Crop
      ↓
MobileNetV3
      ↓
Feature Extractor
      ↓
Feature Vector
```

Feature vector đại diện cho đặc trưng hình ảnh của sản phẩm.

---

# 🔍 13. Similarity Search với FAISS

Các feature vector của Gallery được lưu trước:

```text
Gallery Images
      ↓
MobileNetV3
      ↓
Embeddings
      ↓
gallery_embeddings.npz
      ↓
FAISS Index
```

Khi có một sản phẩm mới:

```text
Query Product
      ↓
MobileNetV3
      ↓
Query Embedding
      ↓
FAISS
      ↓
Top-K Similar Products
```

FAISS giúp tìm kiếm nhanh những sản phẩm có vector đặc trưng gần với sản phẩm đầu vào.

---

# 🗳️ 14. Majority Voting

Thay vì chỉ sử dụng kết quả Top-1, hệ thống lấy **Top-K kết quả gần nhất** và thực hiện Majority Voting.

Ví dụ:

```text
Top-5 Results:

SKU_001
SKU_001
SKU_003
SKU_001
SKU_002
```

Kết quả cuối cùng:

```text
Predicted SKU = SKU_001
```

Cấu hình mặc định:

```python
TOP_K = 5
```

---

# ⚙️ 15. Cấu hình Recognition

Các thông số recognition được đặt trong:

```text
backend/api/config.py
```

Ví dụ:

```python
RECOGNIZE_THRESHOLD = 0.85
TOP_K = 5
```

## `RECOGNIZE_THRESHOLD`

Ngưỡng similarity tối thiểu để chấp nhận kết quả.

```text
Similarity >= 0.85
        ↓
    Accept SKU
```

```text
Similarity < 0.85
        ↓
    Reject / Unknown
```

Giá trị threshold có thể được điều chỉnh dựa trên chất lượng Gallery và kết quả thực tế.

---

# 🧪 16. Debug Recognition

File:

```text
src/debug_recognition.py
```

Dùng để kiểm tra các trường hợp nhận diện sai.

### Debug một ảnh

```bash
python src/debug_recognition.py --image data/anh_cua_ban.jpg
```

### Debug một thư mục

```bash
python src/debug_recognition.py --image_dir data/
```

Script sẽ hiển thị Top-K kết quả:

```text
Query Product
      │
      ├── SKU_001 : 0.91
      ├── SKU_003 : 0.89
      ├── SKU_001 : 0.88
      ├── SKU_002 : 0.84
      └── SKU_001 : 0.82
```

Điều này giúp kiểm tra:

* Model đang nhận diện sản phẩm nào.
* Các SKU nào dễ nhầm lẫn.
* Similarity giữa query và Gallery.
* Gallery có chứa mẫu gây nhiễu hay không.

---

# 🔬 17. Audit Gallery

File:

```text
src/audit_gallery.py
```

Chạy:

```bash
python src/audit_gallery.py
```

Script được sử dụng để kiểm tra chất lượng Gallery Embeddings.

Có thể hỗ trợ phát hiện:

* Embedding bất thường.
* Outlier.
* Các SKU khác nhau nhưng có embedding quá giống nhau.
* Các mẫu Gallery có khả năng gây nhầm lẫn.

---

# 🧹 18. Clean Gallery

File:

```text
src/clean_gallery.py
```

Chạy:

```bash
python src/clean_gallery.py
```

Script được sử dụng để làm sạch Gallery và build lại FAISS Index.

Ví dụ:

```text
embeddings/
├── gallery_embeddings.npz
└── clean_gallery_embeddings.npz

faiss_index/
└── shape_gallery.index
```

Sau khi Gallery được làm sạch, FAISS Index được cập nhật để sử dụng tập dữ liệu tốt hơn cho quá trình recognition.

---

# 🖼️ 19. Gallery Embeddings

Gallery là tập ảnh tham chiếu được sử dụng để nhận diện SKU.

Ví dụ:

```text
SKU_001/
├── image_01.jpg
├── image_02.jpg
├── image_03.jpg
├── image_04.jpg
└── image_05.jpg
```

Quy trình tạo Gallery:

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

# 📌 20. Thêm SKU mới

Khi thêm một SKU mới, nên chuẩn bị nhiều ảnh crop chất lượng tốt.

Khuyến nghị tối thiểu:

```text
4–5 ảnh / SKU
```

Nên có sự đa dạng về:

* Góc nhìn.
* Khoảng cách.
* Ánh sáng.
* Vị trí sản phẩm.
* Kích thước sản phẩm.
* Điều kiện chụp thực tế.

Ví dụ:

```text
SKU_001/
├── front.jpg
├── left.jpg
├── right.jpg
├── bright.jpg
└── shelf.jpg
```

Gallery càng đa dạng và phù hợp với dữ liệu thực tế thì khả năng retrieval có thể càng ổn định.

---

# 🔄 21. Cập nhật FAISS Index

Sau khi thêm hoặc thay đổi Gallery:

```text
Thêm ảnh SKU mới
       ↓
Generate Embeddings
       ↓
Update Gallery
       ↓
Audit / Clean Gallery
       ↓
Build FAISS Index
       ↓
Restart Backend
       ↓
Test Recognition
```

---

# 📊 22. Kết quả đầu ra

Sau khi phân tích ảnh, hệ thống hiển thị bounding box và SKU tương ứng.

Ví dụ:

```text
┌─────────────────────────────────┐
│ Product Statistics              │
│                                 │
│ SKU_001    × 3                  │
│ SKU_002    × 2                  │
│ SKU_005    × 1                  │
│                                 │
│ Total Products: 6               │
└─────────────────────────────────┘
```

Trên ảnh kết quả:

```text
┌─────────────────────────────┐
│ SKU_001                     │
│ ┌───────────────┐           │
│ │               │           │
│ │   PRODUCT     │           │
│ │               │           │
│ └───────────────┘           │
│                             │
│ SKU_002                     │
│ ┌───────────────┐           │
│ │   PRODUCT     │           │
│ └───────────────┘           │
└─────────────────────────────┘
```

---

# 🧩 23. API

Endpoint chính:

```http
POST /analyze
```

### Input

Ảnh kệ hàng được upload lên Backend.

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

API trả về:

* Bounding boxes.
* SKU được nhận diện.
* Similarity score.
* Danh sách sản phẩm.
* Thống kê số lượng từng SKU.
* Tổng số sản phẩm được phát hiện.

---

# 📦 24. Datasets & Resources

## Object Detection

**SKU110K Fixed — Kaggle**

👉 [tại đây](https://www.kaggle.com/datasets/lordrovks/sku110k-fixed)

Dùng cho:

```text
RetinaNet
    ↓
Object Detection
```

---

## Product Recognition

**SHAPE — SHelf mAnagement Product datasEt**

Dataset được cung cấp bởi nhóm tác giả của bài báo **Shelf Management: A Deep Learning-Based System for Shelf Visual Monitoring**.

👉 [tại đây](https://figshare.com/articles/dataset/SHAPE_-_SHelf_mAnagement_Product_datasEt/24100704)

Dùng cho:

```text
MobileNetV3
    ↓
Feature Extraction
    ↓
Product Recognition
```

---

## Model Checkpoints

👉 [tại đây](https://drive.google.com/drive/folders/1jGdX0p8tIiO8To0tW5-vhyTF8PwNtbjW?usp=drive_link)

---

# 🛠️ 25. Công nghệ sử dụng

### Computer Vision

* PyTorch
* Torchvision
* RetinaNet
* MobileNetV3
* OpenCV

### Similarity Search

* FAISS
* Cosine Similarity
* Feature Embedding
* Majority Voting

### Backend

* Python
* FastAPI
* Uvicorn

### Frontend

* HTML
* CSS
* JavaScript

### Dataset

* SKU110K
* SHAPE

---

# 🎯 26. Mục tiêu dự án

Dự án hướng tới việc xây dựng một pipeline Computer Vision hoàn chỉnh cho bài toán:

**Product Detection & SKU Recognition**

kết hợp:

```text
Object Detection
       +
Feature Extraction
       +
Vector Similarity Search
       +
SKU Recognition
       +
Backend API
       +
Web Interface
```

Hệ thống có thể được mở rộng cho các bài toán thực tế như:

* 🛒 Quản lý hàng hóa trên kệ.
* 📦 Kiểm kê sản phẩm tự động.
* 🔎 Nhận diện SKU.
* 📊 Phân tích hình ảnh kệ hàng.
* 📋 Hỗ trợ quản lý tồn kho.
* 🏪 Retail Computer Vision.
* 📐 Shelf monitoring.
* 🤖 Automated inventory analysis.

👉 [tại đây](https://github.com/rokopi-byte/shelf_management)
