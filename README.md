# 🏥 Hệ thống Trí tuệ Y tế (Healthcare Intelligence System) - MLOps Pipeline

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-13ADC7?style=for-the-badge&logo=data-version-control&logoColor=white)

## 📖 Giới thiệu (Overview)

Dự án này là một hệ thống phân loại chuyên khoa y tế tự động dựa trên mô tả triệu chứng của bệnh nhân. Tuy nhiên, thay vì chỉ tập trung vào việc xây dựng một mô hình AI đơn lẻ, dự án này được thiết kế như một **hệ thống MLOps hoàn chỉnh (End-to-End)**.

Mục tiêu cốt lõi không chỉ là dự đoán chính xác, mà là giải quyết các bài toán vận hành hệ thống AI trong thực tế y tế:
- **Tự động hóa luồng triển khai (CI/CD)**.
- **Minh bạch hóa quyết định của AI (Explainable AI - XAI)**.
- **Quản lý phiên bản dữ liệu và mô hình độc lập (Decoupled Architecture)**.

## ✨ Tính năng nổi bật (Key Features)

### 1. Hybrid AI & Explainability (XAI)
Thay vì sử dụng AI như một "hộp đen" (Black Box), hệ thống kết hợp giữa Deep Learning (LSTM) và Clinical Knowledge Base (Hệ chuyên gia):
- **Phân bổ xác suất mềm (Soft Classification):** Cung cấp biểu đồ "Chẩn đoán phân biệt" (Differential Diagnosis) cho tất cả các chuyên khoa.
- **Trích xuất thực thể (Entity Extraction):** Minh chứng quyết định của AI bằng cách làm nổi bật các đặc trưng y khoa quan trọng (VD: *Cranial Feature, Cardiac Signal*).
- **Knowledge Override:** Tự động điều chỉnh trọng số an toàn dựa trên luật y khoa khi phát hiện các triệu chứng đặc thù (VD: gãy xương).

### 2. Kiến trúc MLOps vững chắc (Robust Architecture)
- **Tách biệt hoàn toàn (Decoupled):** Giao diện (UI) và Lõi suy luận (API) hoạt động trên hai container riêng biệt.
- **Data & Model Versioning:** Sử dụng `DVC` kết nối với AWS S3 (giả lập) để quản lý phiên bản file `.pth` (Model weights) tách biệt với Source Code (Git).
- **Continuous Integration/Continuous Testing (CI/CT):** GitHub Actions tự động kiểm thử 3 lớp (API Health, Model Integrity, Data Pipeline) mỗi khi có thay đổi code.

---

## 🏗️ Kiến trúc hệ thống (System Architecture)

Dự án áp dụng mô hình Microservices, được đóng gói toàn bộ bằng Docker Compose:

1. **`api` (FastAPI + PyTorch):** Đóng vai trò là Inference Engine. Tiếp nhận văn bản, mã hóa, chạy qua mạng LSTM, kết hợp cơ sở tri thức y khoa và trả về JSON phân tích chi tiết.
2. **`ui` (Streamlit):** Bảng điều khiển (Dashboard) dành cho bác sĩ. Giao tiếp với API qua cổng an toàn (X-API-KEY protected) và trực quan hóa kết quả (Charts, Progress bars, Tags).

*(Bạn có thể chèn một ảnh sơ đồ luồng dữ liệu vào đây nếu có)*

---

## 🚀 Hướng dẫn Cài đặt và Vận hành (Getting Started)

### Yêu cầu hệ thống (Prerequisites)
- Docker & Docker Compose
- Git
- Python 3.10+ (nếu muốn chạy môi trường ảo cục bộ)

### Khởi chạy hệ thống bằng 1 lệnh duy nhất (One-click Run)
```bash
# 1. Clone repository
git clone https://github.com/mrvan-design/healthcare_mlops_demo.git
cd healthcare_mlops_demo

# 2. Xây dựng và khởi chạy các container (API & UI)
docker-compose up -d --build

Sau khi hệ thống báo Started, bạn có thể truy cập:

Giao diện Bác sĩ (Streamlit): http://localhost:8501

Tài liệu API (Swagger UI): http://localhost:8000/docs

🧪 Demo: Trải nghiệm tính năng XAI
Để thấy rõ sức mạnh của hệ thống Hybrid, hãy nhập thử các kịch bản sau vào giao diện:

Kịch bản 1: Có từ khóa đặc hiệu (Knowledge Override)

Nhập: "Bệnh nhân bị gãy chân do tai nạn giao thông."

Kết quả: Hệ thống nhận diện thực thể "Tổn thương cấu trúc" và đẩy xác suất khoa Chỉnh hình lên mức rất cao (>85%).

Kịch bản 2: Không có từ khóa đặc hiệu (Deep Learning Baseline)

Nhập: "Người bệnh có cảm giác mệt mỏi kéo dài, lả đi không rõ nguyên nhân."

Kết quả: Không có thực thể được bắt. Hệ thống chuyển sang dùng lớp Hidden State của mạng LSTM để phân tích ngữ cảnh, dự đoán khoa Nội tổng quát kèm dòng cảnh báo "Dựa trên phân tích Vector đặc trưng".

Kịch bản 3: Chẩn đoán phân biệt (Complex Case)

Nhập: "Bệnh nhân đau đầu dữ dội, kèm theo đau nhức các khớp tay."

Kết quả: Giao diện hiển thị biểu đồ phân bổ xác suất cho cả khoa Thần kinh và khoa Chỉnh hình, giúp bác sĩ có cái nhìn toàn cảnh.

🛡️ Hệ thống CI/CD (Continuous Integration)
Dự án được tích hợp luồng kiểm thử tự động trên GitHub Actions. Mỗi lệnh git push sẽ kích hoạt Pipeline thực hiện:

Giả lập môi trường Ubuntu.

Cài đặt các Dependency theo cơ chế ghim phiên bản (Pinning) để tránh hiện tượng Dependency Hell.

Chạy pytest đa luồng:

test_api.py: Kiểm tra API Health và Status Codes.

test_mlops.py: Kiểm định Tensor output của mô hình và tính toàn vẹn của dữ liệu đầu vào.

👤 Thông tin Tác giả
Nguyễn Văn Tài
Linkedn : 

Đại học Công nghiệp TP.HCM (IUH)

Chuyên ngành: Hệ thống thông tin (IS)

Định hướng: Data & MLOps Engineering

Dự án này là minh chứng cho việc áp dụng tư duy kỹ thuật hệ thống (Systems Engineering) vào lĩnh vực Trí tuệ Nhân tạo trong Y tế.
