FROM python:3.10-slim

WORKDIR /app

# BƯỚC 1: Cài PyTorch trước. 
# Sau lần build này, tầng này sẽ bị "đóng băng". 
# Dù bạn sửa code hay sửa thư viện khác, Docker cũng KHÔNG tải lại PyTorch nữa.
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# BƯỚC 2: Cài các thư viện nhẹ
# BƯỚC 2: Cài các thư viện nhẹ
COPY requirements.txt .
# Nâng cấp pip và cài đặt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# BƯỚC 3: Copy code
COPY . .

CMD ["python", "extractors/medical_extractor.py"]