import pandas as pd
import boto3
import io

# Thông tin kết nối MinIO (Cố ý hardcode để lát nữa Semgrep sẽ bắt lỗi bảo mật)
MINIO_URL = "http://minio:9000"
ACCESS_KEY = "admin"
SECRET_KEY = "SuperSecret123!"

def extract_medical_records():
    print("Bắt đầu trích xuất dữ liệu hồ sơ bệnh án...")
    
    # 1. Giả lập tập dữ liệu thô (Medical Transcriptions)
    mock_data = {
        "patient_id": ["P001", "P002", "P003", "P004"],
        "transcription": [
            "Bệnh nhân nam, 45 tuổi, tiền sử cao huyết áp. Than phiền đau ngực trái.",
            "Bệnh nhân nữ, 28 tuổi, không có tiền sử bệnh lý. Bị dị ứng với Penicillin.",
            "Kết quả MRI cho thấy có khối u nhỏ ở thùy trán. Cần theo dõi thêm.",
            "Nhịp tim đập nhanh, SPO2 98%. Đã kê đơn thuốc Paracetamol."
        ],
        "label": ["Cardiovascular", "Allergy", "Neurology", "General"]
    }
    df = pd.DataFrame(mock_data)
    
    # 2. Kết nối tới MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    
    # 3. Đảm bảo Bucket tồn tại
    bucket_name = "raw-data"
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except:
        print(f"Bucket '{bucket_name}' chưa có, đang tạo mới...")
        s3_client.create_bucket(Bucket=bucket_name)

    # 4. Đẩy file CSV thẳng lên MinIO từ RAM (Không lưu ra ổ cứng)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    file_name = "medical_transcriptions_raw.csv"
    s3_client.upload_fileobj(csv_buffer, bucket_name, file_name)
    
    print(f"Thành công! Đã tải file {file_name} lên MinIO (Bucket: {bucket_name}).")

if __name__ == "__main__":
    extract_medical_records()