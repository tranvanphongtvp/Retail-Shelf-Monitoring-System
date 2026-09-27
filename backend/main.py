"""
main.py

Backend FastAPI cho pipeline: ảnh vào -> detect (RetinaNet + EM-Merger) ->
crop từng box -> embedding + FAISS search (nhãn/EAN) -> thống kê số lượng
sản phẩm theo từng nhãn trong ảnh.

Không bao gồm shelf row detection / product localization (row/column/subrow)
-- chỉ tập trung đúng phần: ảnh vào, detect, embedding, trả nhãn, thống kê.
Nếu sau này cần thêm localization, có thể ghép thêm ShelfRowDetector từ
pipeline.py vào endpoint này.

Chạy server (từ thư mục backend):
    uvicorn main:app --host 0.0.0.0 --port 8000

Test web: mở fontend/index.html (Live Server hoặc tương đương), upload ảnh kệ hàng.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.config import cfg
from api.detection import ProductDetector
from api.search import ProductRecognizer
from api.inventory import summarize_counts


# ============================== RESPONSE SCHEMA ==============================

class ProductDetection(BaseModel):
    box: list  # [x1, y1, x2, y2]
    det_score: float
    EAN: str
    reliability: float


class AnalyzeResponse(BaseModel):
    total_detected: int          # tổng số box detect được (trước lọc reliability)
    total_recognized: int        # số box nhận diện được nhãn (đạt ngưỡng reliability)
    products: list[ProductDetection]
    counts: dict                 # {EAN: số lượng} -- thống kê chính


# ============================== APP ==============================

app = FastAPI(title="Shelf Product Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model MỘT LẦN lúc khởi động server, không load lại mỗi request
detector: ProductDetector = None
recognizer: ProductRecognizer = None


@app.on_event("startup")
def load_models():
    global detector, recognizer
    print(f"Đang load model lên {cfg.device} ...")
    detector = ProductDetector(cfg)
    recognizer = ProductRecognizer(cfg)
    print("Đã load xong model, server sẵn sàng nhận request.")


@app.get("/health")
def health():
    return {"status": "ok", "device": cfg.device}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if detector is None or recognizer is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng, thử lại sau.")

    contents = await file.read()
    np_arr = np.frombuffer(contents, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh, kiểm tra lại định dạng file.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    boxes, scores = detector.detect(image_rgb)
    total_detected = len(boxes)

    products = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_w), min(y2, img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_rgb[y1:y2, x1:x2]
        label, reliability = recognizer.recognize(crop)

        if reliability > cfg.RECOGNIZE_THRESHOLD:
            products.append(ProductDetection(
                box=[x1, y1, x2, y2],
                det_score=float(score),
                EAN=label,
                reliability=reliability,
            ))

    counts = summarize_counts(products)

    return AnalyzeResponse(
        total_detected=total_detected,
        total_recognized=len(products),
        products=products,
        counts=counts,
    )