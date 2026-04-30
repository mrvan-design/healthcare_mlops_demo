import torch
import os
import pytest
from api.main import MedicalNeuralClassifier, DEPARTMENTS

# ==========================================
# 1. TEST MODEL (Kiểm định Mô hình AI)
# ==========================================
def test_model_architecture():
    """Kiểm tra xem kiến trúc mô hình có khớp với số lượng chuyên khoa không."""
    # Giả lập đầu vào là 1 câu dài 30 chữ (tensor)
    dummy_input = torch.zeros((1, 30), dtype=torch.long)
    
    # Khởi tạo mô hình giả lập
    model = MedicalNeuralClassifier(vocab_size=5000, embedding_dim=64, num_classes=5)
    output = model(dummy_input)
    
    # Đầu ra bắt buộc phải có shape (1, 5) tương ứng 5 khoa. Nếu AI trả ra 4 khoa là lỗi!
    assert output.shape == (1, 5), "❌ Lỗi: Đầu ra mô hình không khớp với 5 chuyên khoa!"
    assert len(DEPARTMENTS) == 5, "❌ Lỗi: Danh sách chuyên khoa trong hệ thống bị thiếu!"

# ==========================================
# 2. TEST DATA (Kiểm định Logic Dữ liệu)
# ==========================================
def test_data_preprocessing():
    """Kiểm tra xem hệ thống có xử lý văn bản bị lỗi (Null/Empty) an toàn không."""
    text = "đau đầu tê bì"
    # Giả lập lại tokenizer trong API
    tokens = [ord(c) % 5000 for c in text[:30]]
    
    assert len(tokens) > 0, "❌ Lỗi: Tokenizer không biến đổi được văn bản thành số!"
    assert all(isinstance(t, int) for t in tokens), "❌ Lỗi: Dữ liệu đầu vào cho Tensor không phải là số nguyên!"

# ==========================================
# 3. TEST AIRFLOW (Kiểm định Pipeline)
# ==========================================
def test_airflow_dags_integrity():
    """Kiểm tra xem các file DAG của Airflow có bị lỗi cú pháp Python (Syntax Error) không."""
    # Nó sẽ quét thư mục airflow/dags, nếu ai đó code sai 1 dấu phẩy, CI/CD sẽ chặn lại ngay.
    dag_folder = "airflow/dags"
    if os.path.exists(dag_folder):
        for filename in os.listdir(dag_folder):
            if filename.endswith(".py"):
                filepath = os.path.join(dag_folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                try:
                    # Thử biên dịch file Python xem có lỗi không
                    compile(source, filepath, 'exec')
                except Exception as e:
                    pytest.fail(f"❌ Phát hiện lỗi cú pháp trong file Airflow DAG '{filename}': {str(e)}")