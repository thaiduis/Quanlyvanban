# Quản lý văn bản (FastAPI + MySQL + OCR)

Ứng dụng quản lý văn bản, hỗ trợ upload PDF/hình ảnh, OCR (pytesseract), tìm kiếm nâng cao bằng MySQL FULLTEXT + fallback LIKE, giao diện HTML cơ bản.

## Tính năng
- Upload tài liệu (PDF/JPG/PNG), tự động OCR và trích xuất text
- Lưu trữ file, metadata, tags, nội dung text vào MySQL
- Tìm kiếm nâng cao theo `title`, `content`, `tags` (FULLTEXT + LIKE)
- Xem/ tải xuống/ xóa tài liệu
- Giao diện web (Jinja2) + API JSON

## Yêu cầu hệ thống (Windows)
- Python 3.10+
- MySQL 8.x (bật InnoDB, hỗ trợ FULLTEXT)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Poppler for Windows](https://blog.alivate.com.au/poppler-windows/)

Sau khi cài đặt, cập nhật `.env`:
- `TESSERACT_CMD` trỏ tới `tesseract.exe`
- `POPPLER_BIN` trỏ tới thư mục `bin` của Poppler

## Cài đặt
```bash
python -m venv .venv
. .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Sửa `.env` cho đúng MySQL và đường dẫn Tesseract/Poppler.

Tạo database MySQL:
```sql
CREATE DATABASE docmgr CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
```

## Chạy ứng dụng
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Ghi chú OCR
- PDF có lớp text: đọc text bằng `pdfplumber`
- PDF scan/ảnh: chuyển trang PDF -> ảnh (pdf2image + Poppler) rồi OCR
- Ảnh: OCR trực tiếp bằng pytesseract

## Cấu trúc thư mục
```
app/
  main.py
  database.py
  models.py
  schemas.py
  ocr.py
  routers/
    documents.py
templates/
  base.html
  index.html
  detail.html
static/
  styles.css
uploads/
```

## Bảo mật & mở rộng
- Thêm xác thực (JWT/OAuth2) khi triển khai thực tế
- Dùng hàng đợi cho OCR nếu tải lớn (Celery/RQ)
- Lưu file lên S3/MinIO nếu cần
