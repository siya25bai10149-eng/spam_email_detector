import streamlit as st
import pandas as pd
import numpy as np
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="wide")

# ── Data ──────────────────────────────────────────────────────────────────────
SPAM = [
    "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
    "FREE VIAGRA! Buy now and get 80% off! Limited time offer!!!",
    "URGENT: Your account will be suspended. Confirm your details now.",
    "Make money fast! Work from home and earn $5000 per week guaranteed!",
    "GET RICH QUICK! No experience needed. Unlimited earning potential!!!",
    "Meet singles in your area tonight! Free registration. Act now!",
    "Lose 30 pounds in 30 days with this miracle pill! Doctors hate this!",
    "Nigerian prince needs your help transferring $10 million. Huge reward!",
    "Alert: Virus detected on your computer. Call this number immediately!",
    "Your PayPal account is limited. Verify now to restore full access.",
]
HAM = [
    "Hi, can we reschedule tomorrow's meeting to 3pm? I have a conflict.",
    "Please find attached the quarterly report for your review.",
    "Thanks for your help with the project. I really appreciate it.",
    "Reminder: Your dentist appointment is on Friday at 2:30pm.",
    "Can you review this pull request when you get a chance?",
    "The conference call is scheduled for 10am EST. Dial-in details attached.",
    "Don't forget to submit your timesheet before end of day Friday.",
    "Your order has been confirmed. Thank you for shopping with us.",
    "Your monthly bank statement is ready to view online.",
    "Great job on the presentation today! The client was very impressed.",
]

def get_data():
    texts  = SPAM + HAM
    labels = ["spam"] * len(SPAM) + ["ham"] * len(HAM)
    return pd.DataFrame({"text": texts, "label": labels})

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

# ── Train ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def train(model_name):
    df = get_data()
    df["clean"] = df["text"].apply(preprocess)
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(df["clean"])
    y = (df["label"] == "spam").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = {"Naive Bayes": MultinomialNB(), "Logistic Regression": LogisticRegression(max_iter=1000),
           "Linear SVM": LinearSVC(max_iter=2000)}[model_name]
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    metrics = {
        "Accuracy":  accuracy_score(y_te, yp),
        "Precision": precision_score(y_te, yp, zero_division=0),
        "Recall":    recall_score(y_te, yp, zero_division=0),
        "F1":        f1_score(y_te, yp, zero_division=0),
    }
    return clf, vec, metrics, confusion_matrix(y_te, yp), df

def predict(text, clf, vec):
    v = vec.transform([preprocess(text)])
    pred = clf.predict(v)[0]
    if hasattr(clf, "predict_proba"):
        conf = clf.predict_proba(v)[0][pred]
    else:
        conf = 1 / (1 + np.exp(-abs(clf.decision_function(v)[0])))
    return ("spam" if pred else "ham"), float(conf)

# ── Sidebar ───────────────────────────────────────────────────────────────────
model_choice = st.sidebar.selectbox("Model", ["Naive Bayes", "Logistic Regression", "Linear SVM"])
st.sidebar.markdown("---\nClassifies emails as **Spam** or **Ham** using TF-IDF + ML.")

clf, vec, metrics, cm, df = train(model_choice)

st.title("📧 Spam Email Detector")
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Performance", "🗂️ Dataset"])

# ── Tab 1: Predict ────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 2])
    with col1:
        if st.button("🚨 Spam Example"): st.session_state.email = SPAM[0]
        if st.button("✅ Ham Example"):  st.session_state.email = HAM[0]
        email = st.text_area("Email text", value=st.session_state.get("email", ""),
                              height=200, placeholder="Paste email body here…")
        run = st.button("🔎 Analyse", use_container_width=True)

    with col2:
        if run and email.strip():
            label, conf = predict(email, clf, vec)
            color = "red" if label == "spam" else "green"
            icon  = "🚨 SPAM" if label == "spam" else "✅ NOT SPAM"
            st.markdown(f"<h2 style='color:{color}'>{icon}</h2>", unsafe_allow_html=True)
            st.progress(conf, text=f"Confidence: {conf:.1%}")
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Words", len(email.split()));      c2.metric("Characters", len(email))
            c1.metric("Exclamations", email.count("!")); c2.metric("Caps", sum(c.isupper() for c in email))
        elif run:
            st.warning("Please enter email text.")
        else:
            st.info("Enter text and click **Analyse**.")

    st.markdown("---")
    st.subheader("Batch Prediction")
    up = st.file_uploader("Upload CSV with `text` column", type=["csv"])
    if up:
        bdf = pd.read_csv(up)
        if "text" not in bdf.columns:
            st.error("CSV must have a `text` column.")
        else:
            results = [predict(str(t), clf, vec) for t in bdf["text"].fillna("")]
            bdf["prediction"], bdf["confidence(%)"] = zip(*results)
            bdf["confidence(%)"] = bdf["confidence(%)"].apply(lambda x: round(x * 100, 1))
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(bdf))
            c2.metric("Spam",  sum(bdf["prediction"] == "spam"))
            c3.metric("Ham",   sum(bdf["prediction"] == "ham"))
            st.dataframe(bdf, use_container_width=True)
            st.download_button("⬇️ Download", bdf.to_csv(index=False).encode(), "results.csv")

# ── Tab 2: Performance ────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Metrics — {model_choice}")
    for col, (k, v) in zip(st.columns(4), metrics.items()):
        col.metric(k, f"{v:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"])
        ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
        st.pyplot(fig, use_container_width=True)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(metrics.keys(), metrics.values(), color=["#e63946","#f4a261","#2dc653","#4895ef"])
        ax2.set_ylim(0, 1.2); ax2.set_title("Model Metrics")
        ax2.spines[["top","right"]].set_visible(False)
        st.pyplot(fig2, use_container_width=True)

# ── Tab 3: Dataset ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Training Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(df)); c2.metric("Spam", len(SPAM)); c3.metric("Ham", len(HAM))
    filt = st.radio("Filter", ["All", "Spam", "Ham"], horizontal=True)
    show = df if filt == "All" else df[df["label"] == filt.lower()]
    st.dataframe(show[["text","label"]].reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ Download Dataset", df.to_csv(index=False).encode(), "dataset.csv")
