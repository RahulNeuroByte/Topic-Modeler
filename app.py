# app.py



import os
import pickle
import streamlit as st
import pandas as pd
import streamlit as st
from datetime import datetime
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from utils import read_uploaded_file, predict_cluster_kmeans, load_pickle
import streamlit.components.v1 as components


# ---------- Paths ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# ---------- Helpers ----------
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()

@st.cache_resource
def load_lda_model(name):
    return LdaModel.load(os.path.join(models_dir, f"tuned_lda_model_{name}.model"))

@st.cache_resource
def load_dictionary(name):
    return Dictionary.load(os.path.join(models_dir, f"dictionary_{name}.dict"))

@st.cache_resource
def load_pipeline(name):
    return load_pickle(os.path.join(models_dir, f"pipeline_{name}.pkl"))
# tuned_kmeans_model_20.pkl
@st.cache_resource
def load_kmeans_model(name):
    return load_pickle(os.path.join(models_dir, f"tuned_kmeans_model_{name}.pkl"))



# ---------- Config ----------
st.set_page_config(page_title="Topic Modeling App", layout="wide")

st.set_page_config(page_title="Document Clustering for Topic Modeling", page_icon="📄")

st.title("Document Clustering & Topic Modeling App")
st.markdown("Built by [RahulNeuroByte](https://github.com/RahulNeuroByte) • Powered by LDA & KMeans.")



# ---------- Sidebar ----------
# Live Date-Time (Interactive Look)

# Get current date and time
current_datetime = datetime.now().strftime("%A, %d %B %Y | %I:%M %p")

# Sidebar styled date-time display
st.sidebar.markdown(
    f"""
    <div style="text-align:center; padding: 8px; background-color: #e6e6fa;
                border-radius: 10px; border: 1px solid #ccc; margin-bottom: 15px;">
        <h5 style='margin: 0; color: #4b0082;'> {current_datetime}</h5>
    </div>
    """,
    unsafe_allow_html=True
)

model_name = st.sidebar.selectbox("Select Model", ["20_newsgroups", "mini_newsgroups", "merged"])
show_html = st.sidebar.checkbox("Show Full pyLDAvis (if available)")



with st.sidebar:
    
    
    # Title and Settings
    st.title("Document Analyzer")
    st.markdown("---")
    st.markdown("###  Settings")
    num_topics = st.slider("Number of Topics (LDA)", 2, 15, 5)
    num_clusters = st.slider("Number of Clusters (KMeans)", 2, 10, 5)
    st.markdown("---")

    # Topic Modeling Options
    st.title("Topic Modeling Dashboard")
    st.markdown("### Quick Insights")
    st.markdown("---")
    st.markdown("### Customize View")
    show_wordcloud = st.checkbox("Show WordCloud", value=True)
    show_pie = st.checkbox("Show Pie Chart", value=True)
    show_table = st.checkbox("Show Topic Table", value=True)
    st.markdown("---")

    # 🥚 Secret Easter Egg
    if st.button("👀 Secret?"):
        messages = [
            "You're curious! That's the first step to being a great Data Scientist 😎",
            "You discovered the hidden egg! 🥚",
            "Stay hungry, stay foolish – Steve Jobs 💡",
            "You’re going to build something amazing 🚀",
        ]
        st.balloons()
        st.success(random.choice(messages))

# ----- Python version -------
import streamlit as st
import sys
st.sidebar.markdown(f"🐍 **Python version:** `{sys.version}`")



# ---------- Input ----------
uploaded_file = st.file_uploader("Upload File (txt, csv, pdf, json)", type=["txt", "csv", "pdf", "json"])
analyze_file = st.button("Analyze Uploaded File")


with st.expander("ℹ️ How It Works"):
    st.write("""
    - Upload your document dataset (.pdf .csv or .txt).
    - Choose between LDA or KMeans.
    - See clusters/topics & download results.
    """)

##  ------------ Input text or paste the text -------------
st.markdown("### Paste Custom Text")
user_text = st.text_area("Enter text below:", height=200)
analyze_text = st.button("Analyze Text Input")

# ---------- Run Analysis ----------
texts, source = [], ""
if analyze_text and user_text.strip():
    texts = [user_text.strip()]
    source = "text"
elif analyze_file and uploaded_file:
    texts = read_uploaded_file(uploaded_file)
    source = "file"

# ---------- Prediction ----------
if source and texts and texts[0].strip():
    raw_text = texts[0]
    st.subheader(" Raw Input")
    st.write(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)

    try:
        pipeline = load_pipeline(model_name.replace("_newsgroups", ""))
        lda_model = load_lda_model(model_name)
        dictionary = load_dictionary(model_name.replace("_newsgroups", ""))
        kmeans_model = load_kmeans_model(model_name.replace("_newsgroups", ""))

        cleaned = pipeline.named_steps["preprocessor"].transform([raw_text])[0]
        vectorized = pipeline.transform([raw_text])
        bow = dictionary.doc2bow(cleaned.split())
        topics = lda_model.get_document_topics(bow)
        topics_sorted = sorted(topics, key=lambda x: x[1], reverse=True)
        topic_num, topic_score = topics_sorted[0] if topics_sorted else (-1, 0)
        topic_words = lda_model.show_topic(topic_num, topn=10) if topic_num >= 0 else []

        cluster = predict_cluster_kmeans([raw_text], pipeline, kmeans_model)[0]

        # Display Predictions
        st.subheader("Cleaned Text")
        st.write(cleaned)

        st.subheader("Topic Prediction")
        if topic_num >= 0:
            st.markdown(f"**Topic:** #{topic_num}  |  **Score:** `{topic_score:.4f}`")
            st.markdown("Top Keywords: " + ", ".join([f"`{w}`" for w, _ in topic_words]))
        else:
            st.warning("No topic assigned.")

        st.subheader(" KMeans Cluster")
        st.markdown(f"Predicted Cluster: `{cluster}`")

        # ---------- Visualizations ----------
        st.subheader("Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            if topic_words:
                fig_pie, ax = plt.subplots(figsize=(4, 4))
                words = [w for w, _ in topic_words]
                weights = [p for _, p in topic_words]
                ax.pie(weights, labels=words, autopct="%1.1f%%", startangle=140)
                ax.set_title(f"Topic #{topic_num} - Pie Chart")
                st.pyplot(fig_pie)
                st.download_button(" Download Pie Chart", fig_to_bytes(fig_pie), file_name="pie_chart.png", mime="image/png")

        with col2:
            if topic_words:
                wc = WordCloud(width=400, height=250, background_color='white').generate_from_frequencies(dict(topic_words))
                fig_wc, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)
                st.download_button(" Download WordCloud", fig_to_bytes(fig_wc), file_name="wordcloud.png", mime="image/png")

        if topics:
            topic_ids = [t[0] for t in topics]
            topic_scores = [t[1] for t in topics]
            fig_dist, ax = plt.subplots(figsize=(6, 3))
            ax.bar(topic_ids, topic_scores, color="skyblue")
            ax.set_xlabel("Topic ID")
            ax.set_ylabel("Score")
            ax.set_title("Document Topic Distribution")
            st.pyplot(fig_dist)
            st.download_button("Download Distribution", fig_to_bytes(fig_dist), file_name="distribution.png", mime="image/png")

    except Exception as e:
        st.error(f"Error during prediction: {e}")


# ---------- PyLDAvis HTML ----------
if show_html:
    st.subheader("Full pyLDAvis Visualization")
    html_path = os.path.join(results_dir, f"lda_topics_visual_{model_name}.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600, scrolling=True)
    else:
        st.warning("pyLDAvis HTML file not found. Run `pyLDAvis_generator.py` to generate it.")


# ----- Footer -------

st.markdown("""
<style>
footer {visibility: hidden;}
footer:after {
    content: '✨ Built with ❤️ by RahulNeuroByte';
    visibility: visible;
    display: block;
    text-align: center;
    color: grey;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)



# ------- My Info on the footer --------- 
st.markdown("""
<style>
/* Hide default Streamlit footer */
footer {visibility: hidden;}

/* Custom footer styling */
.footer {
    background: linear-gradient(to right, #6a0dad, #3f51b5); /* Purple to blue gradient */
    padding: 10px 0 5px;
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    z-index: 9999;
    color: white;
    font-family: 'Segoe UI', sans-serif;
    text-align: center;
}

.footer .top-text {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 6px;
}

.footer-icons a {
    display: inline-block;
    margin: 0 10px;
    color: white;
    transition: transform 0.2s ease, color 0.3s;
}

.footer-icons a:hover {
    color: #ffd700;
    transform: scale(1.2);
}

.footer-icons svg {
    width: 24px;
    height: 24px;
    fill: white;
}
</style>

<div class="footer">
    <div class="top-text"> Built with ❤️ by Rahul</div>
    <div class="footer-icons">
        <a href="https://www.linkedin.com/in/rahul-kumar-dubey-4a4971256" target="_blank" title="LinkedIn">
            <svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>LinkedIn</title><path d="M20.447 20.452H16.89V15.63c0-1.15-.02-2.63-1.604-2.63-1.607 0-1.853 1.254-1.853 2.544v4.908H9.883V9h3.41v1.56h.05c.476-.9 1.635-1.848 3.362-1.848 3.595 0 4.258 2.366 4.258 5.444v6.296zM5.337 7.433c-1.1 0-1.99-.89-1.99-1.987 0-1.098.89-1.988 1.99-1.988 1.097 0 1.987.89 1.987 1.988 0 1.097-.89 1.987-1.987 1.987zM6.872 20.452H3.8V9h3.072v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.723v20.555C0 23.228.792 24 1.771 24h20.451C23.2 24 24 23.228 24 22.278V1.723C24 .774 23.2 0 22.225 0z"/></svg>
        </a>
        <a href="https://x.com/rahuldubey0129?t=j4IEIfiTFQ7eUN6-WeXCzQ&s=09" target="_blank" title="Twitter">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M23.954 4.569c-.885.389-1.83.654-2.825.775 1.014-.611 1.794-1.574 2.163-2.723-.949.568-2.003.982-3.127 1.201-.897-.954-2.178-1.55-3.594-1.55-2.717 0-4.917 2.208-4.917 4.917 0 .39.045.765.127 1.124-4.083-.205-7.699-2.159-10.125-5.134-.422.724-.664 1.561-.664 2.475 0 1.708.87 3.215 2.188 4.099-.807-.026-1.566-.247-2.228-.616v.061c0 2.385 1.693 4.374 3.946 4.827-.413.112-.849.171-1.296.171-.317 0-.626-.03-.928-.086.631 1.953 2.445 3.376 4.6 3.417-1.68 1.318-3.808 2.105-6.102 2.105-.396 0-.787-.023-1.17-.067 2.179 1.397 4.768 2.213 7.557 2.213 9.054 0 14-7.496 14-13.986 0-.21-.005-.423-.014-.634.961-.689 1.8-1.56 2.46-2.548l-.047-.02z"/></svg>
        </a>
        <a href="https://www.instagram.com/iamrahulzz.01" target="_blank" title="Instagram">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 1.366.062 2.633.337 3.608 1.312.975.975 1.25 2.242 1.312 3.608.058 1.266.07 1.646.07 4.847s-.012 3.584-.07 4.85c-.062 1.366-.337 2.633-1.312 3.608-.975.975-2.242 1.25-3.608 1.312-1.266.058-1.646.07-4.85.07s-3.584-.012-4.85-.07c-1.366-.062-2.633-.337-3.608-1.312-.975-.975-1.25-2.242-1.312-3.608C2.175 15.647 2.163 15.267 2.163 12s.012-3.584.07-4.85c.062-1.366.337-2.633 1.312-3.608.975-.975 2.242-1.25 3.608-1.312 1.266-.058 1.646-.07 4.85-.07M12 0C8.741 0 8.332.013 7.052.072 5.773.131 4.675.415 3.72 1.37c-.955.955-1.239 2.053-1.298 3.332C2.013 5.925 2 6.334 2 9.593v4.813c0 3.259.013 3.668.072 4.948.059 1.279.343 2.377 1.298 3.332.955.955 2.053 1.239 3.332 1.298 1.279.059 1.688.072 4.948.072s3.668-.013 4.948-.072c1.279-.059 2.377-.343 3.332-1.298.955-.955 1.239-2.053 1.298-3.332.059-1.279.072-1.688.072-4.948V9.593c0-3.259-.013-3.668-.072-4.948-.059-1.279-.343-2.377-1.298-3.332C20.052.415 18.954.131 17.675.072 16.395.013 15.987 0 12.728 0h-.456z"/><circle cx="12" cy="12" r="3.5"/><circle cx="18.406" cy="5.594" r="1.44"/></svg>
        </a>
        <a href="mailto:rahuldubey9119@gmail.com" title="Email">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path d="M502.3 190.8L327.4 338c-15.6 13.2-39.2 13.2-54.8 0L9.7 190.8C3.9 186.2 0 179.4 0 172V112c0-26.5 21.5-48 48-48h416c26.5 0 48 21.5 48 48v60c0 7.4-3.9 14.2-9.7 18.8zM0 240v160c0 26.5 21.5 48 48 48h416c26.5 0 48-21.5 48-48V240L336.5 386.3c-28.2 23.8-69.8 23.8-98 0L0 240z"/></svg>
        </a>
        <a href="https://github.com/RahulNeuroByte" target="_blank" title="GitHub">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577v-2.165c-3.338.724-4.033-1.415-4.033-1.415-.546-1.387-1.333-1.757-1.333-1.757-1.089-.744.084-.729.084-.729 1.205.085 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.775.418-1.305.762-1.605-2.665-.3-5.467-1.335-5.467-5.931 0-1.31.469-2.381 1.235-3.221-.123-.303-.535-1.521.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.403 1.02.004 2.047.137 3.006.403 2.289-1.553 3.295-1.23 3.295-1.23.653 1.655.242 2.873.119 3.176.77.84 1.232 1.911 1.232 3.221 0 4.609-2.807 5.625-5.479 5.921.43.372.823 1.104.823 2.222v3.293c0 .322.218.694.825.576C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)












