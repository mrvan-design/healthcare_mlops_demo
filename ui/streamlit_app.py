import streamlit as st
import requests

# 1. Cấu hình trang
st.set_page_config(page_title="AI Medical Assistant", page_icon="🏥", layout="centered")

# 2. Giao diện chính
st.title("🏥 Hệ thống Hỗ trợ Chẩn đoán Y tế AI")
st.markdown("""
Hệ thống sử dụng mô hình **Deep Learning (LSTM)** để phân loại bệnh án tự động vào các khoa chuyên môn.
""")

st.divider()

# 3. Khu vực nhập liệu
transcription = st.text_area(
    "Nhập bệnh lý hoặc mô tả triệu chứng của bệnh nhân:",
    placeholder="Ví dụ: Bệnh nhân đau thắt ngực, khó thở, có tiền sử hở van tim...",
    height=200
)


if st.button("🚀 Bắt đầu Phân tích AI"):
    if transcription.strip() == "":
        st.warning("Vui lòng nhập mô tả bệnh lý để AI có thể làm việc!")
    else:
        with st.spinner('Đang truy vấn mô hình Deep Learning...'):
            try:
                # 1. Cấu hình bảo mật & gọi API
                headers = {"X-API-KEY": "Healthcare_Secret_2026"}
                response = requests.post(
                    "http://api:8000/predict", 
                    json={"transcription": transcription},
                    headers=headers,
                    timeout=10 # Tránh treo UI nếu API lỗi
                )

                if response.status_code == 200:
                    result = response.json()
                    
                    # 1. Hiển thị kết quả dự đoán chính
                    st.success(f"### Dự đoán: Khoa {result['department']}")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        # Thanh tiến trình độ tin cậy
                        conf_value = float(result['confidence'].replace('%', '')) / 100
                        st.progress(conf_value)
                    with col2:
                        st.metric("Độ tin cậy", result['confidence'])

                    # 2. PHẦN GIẢI THÍCH (XAI) - Lấy từ trong result['analysis']
                    st.markdown("---")
                    st.subheader("🔍 Phân tích thực thể lâm sàng (Clinical Entities)")
                    
                    analysis_data = result.get('analysis', {})
                    entities = analysis_data.get('detected_entities', [])
                    
                    if entities:
                        st.write("**Các chỉ số bệnh lý được hệ thống nhận diện:**")
                        cols = st.columns(len(entities) if len(entities) > 0 else 1)
                        for i, entity in enumerate(entities):
                            st.info(f"🧬 {entity}")
                    else:
                        st.warning("⚠️ Không tìm thấy thực thể từ điển. Kết quả dựa trên phân tích Vector đặc trưng (Deep Features).")

                    # ===== THÊM ĐOẠN NÀY: BIỂU ĐỒ CHẨN ĐOÁN PHÂN BIỆT =====
                    st.markdown("---")
                    st.subheader("📊 Chẩn đoán phân biệt (Differential Diagnosis)")
                    st.caption("Khả năng phân bổ vào các chuyên khoa khác dựa trên đặc trưng phụ:")
                    
                    all_probs = result.get('all_probabilities', {})
                    for dept, prob_str in all_probs.items():
                        prob_val = float(prob_str.replace('%', '')) / 100
                        # Tạo 3 cột: Tên khoa | Thanh tiến trình | Phần trăm
                        col_dept, col_bar, col_pct = st.columns([3, 6, 2])
                        with col_dept:
                            st.write(f"**{dept}**")
                        with col_bar:
                            # Đổi màu thanh top 1 cho nổi bật
                            if dept == result['department']:
                                st.progress(prob_val)
                            else:
                                st.progress(prob_val) # Streamlit bản mới tự xử lý thanh ngắn dài rất đẹp
                        with col_pct:
                            st.write(f"{prob_str}")
                    # =======================================================

                    with st.expander("🔬 Xem logic ra quyết định của AI"):
                        st.write(f"**Phương pháp:** {analysis_data.get('logic', 'N/A')}")
                        st.write(f"**Cơ sở quyết định:** {analysis_data.get('decision_basis', 'N/A')}")
                        st.caption("Mô hình LSTM đã trích xuất các vector đặc trưng từ chuỗi văn bản để xác định trọng số phân loại.")
                
                elif response.status_code == 403:
                    st.error("Lỗi bảo mật: API Key không hợp lệ!")
                else:
                    st.error(f"Lỗi hệ thống: API trả về mã {response.status_code}")

            except Exception as e:
                st.error(f"Không thể kết nối với Backend: {str(e)}")

# Sidebar nâng cấp để nhìn giống hệ thống quản lý hơn
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")
st.sidebar.write("🟢 **API State:** Online")
st.sidebar.write("🆔 **Model ID:** `MED-LSTM-V1` (Production)")
st.sidebar.write("🔒 **Security:** X-API-KEY Protected")