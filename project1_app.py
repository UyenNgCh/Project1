import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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

# ====== Sidebar ======
st.sidebar.title("PROJECT MENU")
menu = st.sidebar.selectbox(
    "Chọn trang",
    [
        "Business Objective",
        "Build Project - Sentiment Analysis",
        "Build Project - Clustering",
        "New Prediction - Sentiment",
        "New Prediction - Clustering"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Thành viên thực hiện")
st.sidebar.write("Nguyễn Lê Châu Uyên")
st.sidebar.write("uyenngchau@gmail.com")
st.sidebar.write("Nguyễn Trọng Hiến")
st.sidebar.write("tronghien97lx@gmail.com")

# ====== Business Objective ======
if menu == "Business Objective":
    st.title("PHÂN TÍCH DỮ LIỆU REVIEW CÔNG TY")
    st.subheader("Business Objective")
    st.image("cover_image.jpg", use_column_width=True)  # đổi thành ảnh của bạn
    st.markdown("""
    - **Mục tiêu:** Ứng dụng NLP để phân tích cảm xúc và chủ đề nổi bật trong các review về công ty.
    - **Yêu cầu:** 
        + Phân tích sentiment của review
        + Phân cụm chủ đề các review & công ty
        + Đưa ra gợi ý cải tiến cho doanh nghiệp
    """)

# ====== Build Project - Sentiment Analysis ======
elif menu == "Build Project - Sentiment Analysis":
    st.title("Build Project: Sentiment Analysis")
    st.write("**Các mô hình đã thử nghiệm:** Logistic Regression, Naive Bayes, SVM,...")
    # Load bảng kết quả đã lưu hoặc tính lại
    results_df = pd.read_csv('sentiment_results.csv')
    st.dataframe(results_df)
    st.image('sentiment_confusion_matrix.png', caption="Confusion Matrix")
    st.image('sentiment_wordcloud.png', caption="Word Cloud Positive Review")
    st.image('sentiment_wordcloud_neg.png', caption="Word Cloud Negative Review")
    st.write("**Nhận xét:** Mô hình Logistic Regression cho kết quả tốt nhất với F1-score cao nhất...")

# ====== Build Project - Clustering ======
elif menu == "Build Project - Clustering":
    st.title("Build Project: Clustering")
    st.write("**Các thuật toán:** LDA, KMeans, Agglomerative, DBSCAN (có FastText Embedding)")
    cluster_df = pd.read_csv('clustering_results.csv')
    st.dataframe(cluster_df)
    st.image('lda_wordcloud_0.png', caption="Wordcloud Topic 0")
    st.image('lda_wordcloud_1.png', caption="Wordcloud Topic 1")
    st.image('clustering_pca.png', caption="PCA Visualize Clusters")
    st.write("**Nhận xét:** KMeans và Agglomerative clustering cho kết quả phân cụm ổn định, sát với số topic LDA...")

# ====== New Prediction - Sentiment ======
elif menu == "New Prediction - Sentiment":
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

# ====== New Prediction - Clustering ======
elif menu == "New Prediction - Clustering":
    st.title("Phân tích công ty theo review")
    company_name = st.text_input("Nhập tên công ty:")
    if st.button("Phân tích công ty"):
        # Tìm id công ty
        company_row = company_data[company_data['Company Name'].str.lower() == company_name.strip().lower()]
        if not company_row.empty:
            st.write(f"**ID công ty:** {company_row['id'].values[0]}")
            # Thống kê sentiment
            st.write("### Thống kê sentiment")
            pos = company_row['positive_pct'].values[0]
            neu = company_row['neutral_pct'].values[0]
            neg = company_row['negative_pct'].values[0]
            st.write(f"- Tích cực: {pos:.1%}")
            st.write(f"- Trung lập: {neu:.1%}")
            st.write(f"- Tiêu cực: {neg:.1%}")
            # Thông tin topic/cụm, từ khóa đặc trưng
            st.write("### Công ty thuộc cụm:")
            cluster = company_row['cluster_KMeans'].values[0]
            topic = company_row['lda_topic'].values[0]
            st.write(f"- Cluster: {cluster}")
            st.write(f"- Topic: {topic}")
            # Wordcloud (giả sử đã lưu sẵn ảnh theo id công ty)
            img_path = f"wordcloud_company_{company_row['id'].values[0]}.png"
            st.image(img_path, caption="Wordcloud các từ khóa nổi bật")
            # Gợi ý cải tiến
            st.write("### Gợi ý cải tiến")
            st.write(company_row['suggestion'].values[0])
        else:
            st.error("Không tìm thấy công ty!")