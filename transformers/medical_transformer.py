import pandas as pd
import boto3
import io
import hashlib
from sqlalchemy import create_engine

# Cấu hình kết nối (Sử dụng tên container nội bộ của Docker)
MINIO_URL = "http://minio:9000"
DB_URI = "postgresql://healthcare_admin:SecurePass2026!@postgres:5432/medical_features"

def transform_medical_data():
    print("1. Đang kết nối MinIO và kéo dữ liệu thô...")
    s3_client = boto3.client(
        's3', endpoint_url=MINIO_URL,
        aws_access_key_id='admin', aws_secret_access_key='SuperSecret123!'
    )
    
    # Lấy file từ RAM (không cần lưu xuống ổ cứng)
    obj = s3_client.get_object(Bucket='raw-data', Key='medical_transcriptions_raw.csv')
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    print("2. Bắt đầu làm sạch và Ẩn danh hóa (Anonymization)...")
    # A. Mã hóa ID bệnh nhân để bảo mật y tế (Biến P001 thành chuỗi băm ngẫu nhiên)
    df['patient_id_hashed'] = df['patient_id'].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:12]
    )
    df = df.drop(columns=['patient_id']) # Cực kỳ quan trọng: Xóa ID gốc
    
    # B. Làm sạch văn bản (Chữ thường, xóa ký tự đặc biệt)
    df['clean_transcription'] = df['transcription'].str.lower().str.strip()
    # Chỉ giữ lại chữ cái, số và dấu cách (hỗ trợ cả tiếng Việt nếu có)
    df['clean_transcription'] = df['clean_transcription'].str.replace(r'[^\w\s]', '', regex=True)
    
    print("3. Đang đẩy dữ liệu sạch vào cơ sở dữ liệu PostgreSQL...")
    engine = create_engine(DB_URI)
    
    # Tạo bảng 'cleaned_records' và đổ dữ liệu vào
    df.to_sql('cleaned_records', engine, if_exists='replace', index=False)
    
    print(f"Thành công! Đã xử lý {len(df)} hồ sơ và lưu vào bảng 'cleaned_records'.")

if __name__ == "__main__":
    transform_medical_data()