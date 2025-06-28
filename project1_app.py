import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import gensim
import pickle

# ====== Load model & data ======
@st.cache_resource
def load_models():
    # Đường dẫn file model/vectorizer đã lưu (đã sửa lại cho đồng bộ với file pickle)
    with open('tfidf_balanced.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('sentiment_model.pkl', 'rb') as f:
        sentiment_model = pickle.load(f)
    with open('lda_model.pkl', 'rb') as f:
        lda_model = pickle.load(f)
    with open('dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('agglo_model.pkl', 'rb') as f:
        agglo_model = pickle.load(f)
    with open('dbscan_model.pkl', 'rb') as f:
        dbscan_model = pickle.load(f)
    with open('company_data.pkl', 'rb') as f:
        company_data = pickle.load(f)
    return (tfidf_vectorizer, sentiment_model, lda_model, dictionary, 
            kmeans_model, agglo_model, dbscan_model, company_data)

(tfidf_vectorizer, sentiment_model, lda_model, dictionary, 
 kmeans_model, agglo_model, dbscan_model, company_data) = load_models()

# ====== Sidebar with 2 selectboxes ======
st.sidebar.title("MENU")

main_menu = st.sidebar.selectbox(
    "Project 01",
    ["Business Objective", "Sentiment Analysis", "Information Clustering"]
)

# Submenu options mapping
submenu_options = {
    "Business Objective": ["Business Objective"],
    "Sentiment Analysis": ["Business Objective", "Build Project", "New Prediction"],
    "Information Clustering": ["Business Objective", "Build Project", "New Prediction"]
}

submenu = st.sidebar.selectbox(
    "Chi tiết",
    submenu_options[main_menu]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Thành viên thực hiện")
st.sidebar.write("Nguyễn Lê Châu Uyên")
st.sidebar.write("uyenngchau@gmail.com")
st.sidebar.write("Nguyễn Trọng Hiến")
st.sidebar.write("tronghien97lx@gmail.com")

# ====== Main content logic ======

# --- 1. Business Objective chung ---
if main_menu == "Business Objective":
    st.image("cover_image.jpg", use_container_width=True)
    st.title("Sentiment Analysis & Information Clustering")
    st.subheader("Business Objective")
    st.markdown("""
    - **Mục tiêu dự án:** 
        + Phân loại đánh giá người dùng thành 3 cảm xúc: tích cực, trung lập, tiêu cực
        + So sánh độ chính xác giữa các mô hình học máy
        + Phân cụm đánh giá để khám phá chủ đề trong dữ liệu
        + Hỗ trợ doanh nghiệp nắm bắt về nhân sự, cải thiện tuyển dụng
    """)

# --- 2. Sentiment Analysis ---
elif main_menu == "Sentiment Analysis":
    if submenu == "Business Objective":
        st.image("sentiment_cover.jpg", use_container_width=True)
        st.title("Sentiment Analysis - Business Objective")
        st.subheader("Mục tiêu phân tích cảm xúc")
        st.markdown("""
        - **Phát hiện cảm xúc tích cực, tiêu cực, trung lập trong review của nhân viên về công ty.**
        - **Giúp doanh nghiệp hiểu rõ hơn về điểm mạnh/yếu trong trải nghiệm nhân viên.**
        """)
    elif submenu == "Build Project":
        st.title("Build Project: Sentiment Analysis")
        st.subheader("Số lượng nhãn cảm xúc")
        st.image('sentiment_label.jpg')
        st.markdown("""
        - **Nhận xét:** 
            + Positive có 6,028 đánh giá, chiếm tỷ lệ 73.76%, cho thấy hầu như người tham gia đánh giá phản hồi tích cực về các công ty IT
            + Neutral có 1,639 đánh giá, chiếm tỷ lệ 19.48%, cho thấy một bộ phận người tham gia đánh giá phản hồi trung lập
            + Negative có 570 đánh giá, chiếm tỷ lệ 6.77%, cho thấy đánh giá không hài lòng chiếm tỷ lệ khá thấp
        - **Đánh giá:** 
            + Dữ liệu mất cân bằng, nhóm Positive chiếm ưu thế rõ rệt → cần xử lý cân bằng khi huấn luyện mô hình
        """)    

        st.subheader("**Các mô hình đã huấn luyện:** Logistic Regression, Naive Bayes, SVM")
        st.write("Bảng kết quả")
        st.image("sentiment_result.jpg", use_container_width=True)
        st.markdown("""
        - **Nhận xét:** 
            + Tất cả mô hình có Macro F1 > 80% → khả năng phân loại giữa 3 nhóm cảm xúc được cải thiện rõ rệt
            + Linear SVM vượt trội nhất ở mọi chỉ số (Accuracy, Macro F1, Weighted F1), kế đến là Logistic Regression cũng cho kết quả khá tốt
        """)
    elif submenu == "New Prediction":
        st.title("Dự đoán cảm xúc review mới")
        text_input = st.text_area("Nhập nội dung review:", "")
        if st.button("Phân tích cảm xúc"):
            # Tiền xử lý giống pipeline gốc
            def preprocess_for_sentiment(text):
                # ... chuẩn hóa, tách từ, loại stopwords như pipeline gốc ...
                # (Bạn nên copy đúng hàm tiền xử lý đã dùng khi train model)
                return " ".join(text.split())  # placeholder

            X_new = tfidf_vectorizer.transform([preprocess_for_sentiment(text_input)])
            pred = sentiment_model.predict(X_new)[0]
            st.success(f"Kết quả cảm xúc: **{pred.upper()}**")

# --- 3. Information Clustering ---
elif main_menu == "Information Clustering":
    if submenu == "Business Objective":
        st.image("clustering_cover.jpg", use_container_width=True)  # Đổi thành ảnh phù hợp nếu có
        st.title("Information Clustering - Business Objective")
        st.subheader("Mục tiêu phân cụm chủ đề")
        st.markdown("""
        - **Phân nhóm các review/công ty thành các chủ đề/cụm nội dung nổi bật.**
        - **Khám phá các khía cạnh quan trọng như lương thưởng, môi trường làm việc, đào tạo, chính sách,...**
        - **Hỗ trợ doanh nghiệp xác định điểm mạnh/yếu theo từng nhóm chủ đề cụ thể.**
        """)
    elif submenu == "Build Project":
        st.subheader("Dùng LDA để xác định số cụm = 4")
        st.markdown("""
        - **Tên các cụm:** 
            + 0: Lương & lợi ích
            + 1: Cơ hội đào tạo & phát triển
            + 2: Văn hóa & Chính sách công ty
            + 3: Văn phòng & môi trường làm việc
        """)
        st.subheader("Trực quan từ khóa của mỗi cụm")
        st.image("clustering_wordcloud_0.jpg")
        st.image("clustering_wordcloud_1.jpg")
        st.image("clustering_wordcloud_2.jpg")
        st.image("clustering_wordcloud_3.jpg")
        st.subheader("**Các mô hình đã huấn luyện:** KMeans, Agglomerative Clustering, BSCAN")
        st.write("Bảng kết quả")
        st.image("clustering_result.jpg", use_container_width=True)
        st.image("clustering_visual.jpg", use_container_width=True)
        st.markdown("""
        - **Nhận xét:** 
            + Thuật toán KMeans và Agglomerative cho giá trị Silhouette ổn
            + Thuật toán DBSCAN không phù hợp để phân cụm đánh giá người dùng trong bài toán này
        """)
    elif submenu == "New Prediction":
        st.title("Phân tích công ty theo review")
        company_name = st.text_input("Nhập tên công ty:")
        if st.button("Phân tích công ty"):
            # Tìm id công ty
            company_row = company_data[company_data['Company Name'].str.lower() == company_name.strip().lower()]
            if not company_row.empty:
                st.write(f"**ID công ty:** {company_row['id'].values[0]}")
            else:
                st.error("Không tìm thấy công ty!")