import os
import subprocess
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import pandas as pd
from sqlalchemy import create_engine

# Cấu hình Logging chuyên nghiệp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tắt cảnh báo Git Python
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# 1. KIẾN TRÚC MÔ HÌNH (MODEL ARCHITECTURE)
class MedicalNeuralClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(MedicalNeuralClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.softmax(out)

# 2. HÀM QUẢN LÝ DVC (DVC HANDLER)
def dvc_versioning(file_path):
    logger.info(f"Đang thực hiện DVC versioning cưỡng chế cho: {file_path}")
    try:
        # 1. Khởi tạo cưỡng chế (-f để ghi đè nếu đã có folder lỗi)
        subprocess.run(["dvc", "init", "--no-scm", "-f"], check=True)
        
        # 2. Cấu hình Remote (Lưu ý: dùng tên service 'minio' của Docker)
        subprocess.run(["dvc", "remote", "add", "-d", "myremote", "s3://dvc-storage", "-f"], check=True)
        subprocess.run(["dvc", "remote", "modify", "myremote", "endpointurl", "http://minio:9000"], check=True)
        
        # 3. Thêm file vào DVC
        subprocess.run(["dvc", "add", file_path], check=True)
        
        # 4. Đẩy lên MinIO
        # DVC sẽ tự lấy Access Key từ biến môi trường chúng ta truyền vào Airflow
        subprocess.run(["dvc", "push"], check=True)
        
        logger.info("✅ DVC: Đẩy mô hình lên MinIO thành công!")
    except Exception as e:
        logger.error(f"❌ Lỗi DVC thực tế: {e}")
        # Quan trọng: Raise lỗi để Airflow đánh dấu Task là Fail nếu DVC thất bại
        raise e

# 3. LUỒNG HUẤN LUYỆN CHÍNH (MAIN TRAINING PIPELINE)
def run_training():
    # Cấu hình môi trường MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Healthcare_Standard_PyTorch")

    # A. Lấy dữ liệu
    logger.info("Đang kết nối Database...")
    engine = create_engine("postgresql://healthcare_admin:SecurePass2026!@postgres:5432/medical_features")
    try:
        df = pd.read_sql("SELECT * FROM cleaned_records", engine)
    except Exception:
        logger.warning("Không có dữ liệu trong DB, tạo dữ liệu mẫu để demo...")
        df = pd.DataFrame({'id': range(100)})

    # B. Tiền xử lý (Giả lập Tokenization)
    vocab_size = 5000
    num_classes = 5
    X = torch.randint(0, vocab_size, (len(df), 30)) # 30 tokens mỗi bệnh án
    y = torch.randint(0, num_classes, (len(df),))

    # C. Khởi tạo Model & Optimizer
    model = MedicalNeuralClassifier(vocab_size, 64, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.NLLLoss()

    # D. Training Loop với MLflow Tracking
    with mlflow.start_run(run_name="Standard_Training_Run"):
        logger.info("Bắt đầu vòng lặp huấn luyện...")
        model.train()
        
        epochs = 15
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.005)
        mlflow.log_param("model_type", "LSTM_Classifier")

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            mlflow.log_metric("loss", loss.item(), step=epoch)
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

        # E. Lưu trữ vật lý
        model_name = "medical_model.pth"
        torch.save(model.state_dict(), model_name)
        logger.info(f"Đã lưu mô hình tại: {model_name}")

        # F. DVC Versioning
        dvc_versioning(model_name)

        # G. Log mô hình vào MLflow Registry
        mlflow.pytorch.log_model(model, "medical_nlp_model")
        logger.info("Đã đăng ký mô hình vào MLflow Registry.")

if __name__ == "__main__":
    run_training()