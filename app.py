# app.py
# Streamlit Frontend for Recommendation System (Task 1: CNN Recommender + Task 2: Anomaly Detection)
# ---------------------------------------------------------------
# Files (if present) auto-load:
# - Cleaned CSVs: events_cleaned.csv, category_tree_cleaned.csv, item_properties_cleaned.csv
# - Task 1 artifacts: cnn_model.h5, tokenizer.json, labelencoder.pkl
# - Task 2 artifacts: cnn_ae.h5, scaler.pkl
# Fallbacks are used if artifacts are missing.

import os
import io
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

# Optional: TensorFlow only when models exist
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Recommendation System",
    page_icon="🧠",
    layout="wide",
)

# -------------------------
# Minimal styling
# -------------------------
st.markdown("""
<style>
/* clean headers + cards */
.reportview-container .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
h1, h2, h3 {color:#0b3d91;}
div[data-testid="stMetricValue"] {font-size: 1.4rem;}
.sidebar .sidebar-content {background: #f7f9fc;}
.section-card {background:#ffffff;border:1px solid #e7ecf3;border-radius:16px;padding:18px;margin-bottom:16px;box-shadow: 0 2px 10px rgba(0,0,0,0.03);}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------

REQUIRED_EVENTS_COLS = {"timestamp", "visitorid", "event", "itemid"}
REQUIRED_PROPS_COLS  = {"timestamp", "itemid", "property", "value"}
REQUIRED_CAT_COLS    = {"categoryid", "parentid"}

def clean_numeric_str(val):
    # Handles ugly values like "n277.200", "1116713 960601 n277.200"
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Keep only numbers, dot and minus; drop words/letters
    # Split by whitespace, keep numeric-like tokens
    toks = []
    for t in s.split():
        t = t.replace(",", ".")
        # remove leading 'n'
        if t.startswith("n"):
            t = t[1:]
        try:
            float(t)
            toks.append(t)
        except:
            continue
    if not toks:
        return np.nan
    # If multiple numeric tokens, choose the last (often the real numeric value)
    try:
        return float(toks[-1])
    except:
        return np.nan

def ensure_datetime_ms(series):
    # Accepts mixed timestamp: ms int, seconds, or ISO strings
    s = series.copy()
    # Try numeric to ms
    def _to_dt(x):
        if pd.isna(x):
            return pd.NaT
        try:
            xv = float(x)
            # Heuristic: if very large -> ms
            if xv > 1e12:
                xv = int(xv)
                return pd.to_datetime(xv, unit="ms", errors="coerce")
            elif xv > 1e10:
                # Some datasets put microseconds
                return pd.to_datetime(int(xv), unit="us", errors="coerce")
            else:
                return pd.to_datetime(int(xv), unit="s", errors="coerce")
        except:
            return pd.to_datetime(x, errors="coerce")
    return s.apply(_to_dt)

@st.cache_data(show_spinner=False)
def load_csv_default_or_upload(default_path, uploaded_file, expected_cols=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        return None, f"Missing: {default_path}"
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if expected_cols:
        missing = set([c.lower() for c in expected_cols]) - set(df.columns)
        if missing:
            return None, f"Missing columns in {default_path or 'uploaded file'}: {missing}"
    return df, None

def build_user_features(events_df):
    feats = []
    for uid, g in events_df.groupby("visitorid"):
        total = len(g)
        views = (g["event"] == "view").sum()
        adds = (g["event"] == "addtocart").sum()
        buys = (g["event"] == "transaction").sum()
        feats.append({
            "visitorid": uid,
            "total_events": total,
            "views": views,
            "adds": adds,
            "buys": buys,
            "add_rate": (adds/total) if total > 0 else 0.0,
            "conv_rate": (buys/total) if total > 0 else 0.0
        })
    return pd.DataFrame(feats).set_index("visitorid")

def latest_item_property(item_props, prop_name):
    # Get latest value per item for a given property
    sub = item_props[item_props["property"] == prop_name].sort_values("timestamp")
    return sub.groupby("itemid")["value"].last().to_dict()

def prepare_task1_sequences(events, item_props, target_prop, tokenizer=None, max_words=5000, max_len=50):
    # Build mapping from item->label (latest)
    item_latest_prop = latest_item_property(item_props, target_prop)

    X_texts, y_labels = [], []
    # For each add-to-cart event, collect user's prior viewed items' target_prop values as tokens
    add_df = events[events["event"] == "addtocart"]
    for _, row in add_df.iterrows():
        hist = events[(events["visitorid"] == row["visitorid"]) &
                      (events["timestamp"] < row["timestamp"]) &
                      (events["event"] == "view")]
        tokens = [str(item_latest_prop.get(i, "")) for i in hist["itemid"].values]
        label = item_latest_prop.get(row["itemid"])
        if label and tokens:
            X_texts.append(" ".join(tokens))
            y_labels.append(label)

    if len(X_texts) == 0:
        return None, None, None, None, "No training samples could be formed."

    # Fit tokenizer if absent
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_texts)
    X_seq = tokenizer.texts_to_sequences(X_texts)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding="post")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    return X_pad, y_enc, tokenizer, le, None

def recommend_content_based(events, item_props, visitorid, topn=10, exclude_already_viewed=True):
    # Simple fallback: recommend items whose properties (text) are most similar to the user's viewed items
    # Build item "text" from latest property values (excluding noisy properties)
    noisy = {"available", "categoryid"}
    props_pivot = (
        item_props[~item_props["property"].isin(noisy)]
        .sort_values("timestamp")
        .groupby(["itemid", "property"])["value"].last().unstack(fill_value="")
    )
    props_pivot = props_pivot.fillna("")
    props_pivot["__text__"] = props_pivot.astype(str).apply(lambda r: " ".join(r.values.tolist()), axis=1)

    vect = TfidfVectorizer(max_features=5000)
    item_tfidf = vect.fit_transform(props_pivot["__text__"].values)

    # User profile = mean of TF-IDF of items the user viewed
    uviews = events[(events["visitorid"] == visitorid) & (events["event"] == "view")]
    viewed_ids = [str(i) for i in uviews["itemid"].tolist()]
    if len(viewed_ids) == 0:
        # cold start -> top popular items by views
        pop = events[events["event"]=="view"]["itemid"].value_counts().index[:topn].astype(str).tolist()
        return pop, "Cold-start: no views; showing popular items."

    sub = props_pivot.loc[props_pivot.index.astype(str).isin(viewed_ids)]
    if sub.empty:
        pop = events[events["event"]=="view"]["itemid"].value_counts().index[:topn].astype(str).tolist()
        return pop, "No overlap with properties; showing popular items."

    sub_tfidf = vect.transform(sub["__text__"].values)
    user_vec = sub_tfidf.mean(axis=0)

    sims = cosine_similarity(user_vec, item_tfidf).ravel()
    props_pivot = props_pivot.assign(similarity=sims)

    if exclude_already_viewed:
        props_pivot = props_pivot.loc[~props_pivot.index.astype(str).isin(viewed_ids)]
    recs = props_pivot.sort_values("similarity", ascending=False).index.astype(str).tolist()[:topn]
    return recs, "Content-based recommendations."

# -------------------------
# Sidebar: data + artifacts
# -------------------------
st.sidebar.header("📦 Data & Artifacts")
use_cleaned = st.sidebar.checkbox("Use cleaned CSV filenames", value=True)
events_file = st.sidebar.file_uploader("events.csv (or cleaned)", type=["csv"])
cat_file    = st.sidebar.file_uploader("category_tree.csv (or cleaned)", type=["csv"])
props_file  = st.sidebar.file_uploader("item_properties.csv (or cleaned)", type=["csv"])

default_events = "events_cleaned.csv" if use_cleaned else "events.csv"
default_cat    = "category_tree_cleaned.csv" if use_cleaned else "category_tree.csv"
default_props  = "item_properties_cleaned_n.csv" if use_cleaned else "item_properties.csv"

events, e_err = load_csv_default_or_upload(default_events, events_file, REQUIRED_EVENTS_COLS)
category_tree, c_err = load_csv_default_or_upload(default_cat, cat_file, REQUIRED_CAT_COLS)
item_props, p_err = load_csv_default_or_upload(default_props, props_file, REQUIRED_PROPS_COLS)

if any([e_err, c_err, p_err]):
    st.error(f"Data loading error:\n- {e_err}\n- {c_err}\n- {p_err}")
    st.stop()

# Parse timestamps robustly
events["timestamp"] = ensure_datetime_ms(events["timestamp"])
item_props["timestamp"] = ensure_datetime_ms(item_props["timestamp"])
# Basic cleaning
events = events.dropna(subset=["timestamp", "visitorid", "event", "itemid"])
item_props = item_props.dropna(subset=["timestamp", "itemid", "property", "value"])
item_props["value"] = item_props["value"].apply(clean_numeric_str).fillna(item_props["value"])

st.title("🧠 Recommendation System Dashboard")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["📈 EDA", "🛒 Recommendations (Task 1)", "🚨 Anomaly Detection (Task 2)"])

# -------------------------
# TAB 1: EDA
# -------------------------
with tab1:
    st.subheader("Data Overview")
    colA, colB, colC = st.columns(3)
    colA.metric("Events", f"{len(events):,}")
    colB.metric("Item Properties", f"{len(item_props):,}")
    colC.metric("Categories", f"{len(category_tree):,}")

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Event Distribution")
        ev_counts = events["event"].value_counts()
        fig = px.bar(ev_counts, title="Events by Type")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Daily Views")
        ed = events.copy()
        ed["date"] = ed["timestamp"].dt.date
        daily_views = ed[ed["event"] == "view"].groupby("date").size().reset_index(name="views")
        if not daily_views.empty:
            fig2 = px.line(daily_views, x="date", y="views", title="Daily Views Over Time")
            fig2.update_yaxes(tickformat=",")  # no scientific notation
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No view events found.")

        st.markdown("#### Top Properties (excluding noisy)")
        noisy = {"available", "categoryid"}
        top_props = (
            item_props[~item_props["property"].isin(noisy)]
            ["property"].value_counts().head(20)
        )
        if not top_props.empty:
            fig3 = px.bar(top_props, title="Top Properties")
            fig3.update_yaxes(tickformat=",")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No non-noisy properties found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Correlation Heatmap (Numeric properties only)")
        # Create wide numeric features per item
        props_num = item_props.copy()
        # try to coerce value
        props_num["value_num"] = pd.to_numeric(props_num["value"], errors="coerce")
        num_wide = (
            props_num.dropna(subset=["value_num"])
            .sort_values("timestamp")
            .groupby(["itemid", "property"])["value_num"].last().unstack()
        )
        if num_wide is not None and num_wide.shape[1] > 1:
            corr = num_wide.corr().fillna(0)
            fig4 = px.imshow(corr, title="Property Correlations", color_continuous_scale="RdBu", origin="lower")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Not enough numeric properties to compute correlation.")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# TAB 2: Recommendations (Task 1)
# -------------------------
with tab2:
    st.subheader("Task 1: Property-based Recommendation (CNN if available)")

    # Target property: most frequent
    target_prop = item_props["property"].value_counts().index[0]
    st.write(f"**Target property:** `{target_prop}`")

    # Artifacts
    model_path = st.text_input("CNN model path (h5)", value="cnn_model.h5")
    tokenizer_path = st.text_input("Tokenizer path (json)", value="tokenizer.json")
    labelenc_path = st.text_input("LabelEncoder path (pkl)", value="labelencoder.pkl")

    cnn_model = None
    tokenizer = None
    label_enc = None
    MAX_LEN = 50

    if TF_AVAILABLE and os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(labelenc_path):
        try:
            cnn_model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, "r") as f:
                tok_json = json.load(f)
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tok_json)
            with open(labelenc_path, "rb") as f:
                label_enc = pickle.load(f)
            st.success("Loaded CNN recommender artifacts.")
        except Exception as e:
            st.warning(f"Could not load CNN recommender artifacts. Falling back. Reason: {e}")
    else:
        st.info("CNN artifacts not provided or TensorFlow unavailable. Using content-based fallback.")

    visitor_ids = events["visitorid"].unique().tolist()
    chosen_user = st.selectbox("Select a visitor", visitor_ids[:5000] if len(visitor_ids)>5000 else visitor_ids)

    topN = st.slider("Number of recommendations", 5, 30, 10)

    run_btn = st.button("Recommend")

    if run_btn:
        if cnn_model is not None and tokenizer is not None and label_enc is not None:
            # Build a single-sample sequence from user's history
            # Use the same logic used for training: user's viewed items -> tokens from target_prop
            item_latest_prop = latest_item_property(item_props, target_prop)
            hist = events[(events["visitorid"] == chosen_user) & (events["event"] == "view")] \
                    .sort_values("timestamp")
            tokens = [str(item_latest_prop.get(i, "")) for i in hist["itemid"].values if str(item_latest_prop.get(i,""))!=""]
            if len(tokens)==0:
                st.info("No valid tokens from user's history; falling back to content-based.")
                recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
                st.success(msg)
                st.write("Recommendations (itemid):", recs)
            else:
                text = " ".join(tokens)
                seq = tokenizer.texts_to_sequences([text])
                pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
                probs = cnn_model.predict(pad, verbose=0)[0]
                top_k_idx = np.argsort(probs)[::-1][:3]
                predicted_props = label_enc.inverse_transform(top_k_idx)
                st.write("Top predicted property values:", [str(p) for p in predicted_props])

                # Recommend items whose target_prop matches the top predicted values and are closest to user profile
                item_latest = latest_item_property(item_props, target_prop)
                target_items = [iid for iid, val in item_latest.items() if str(val) in set(map(str, predicted_props))]
                if len(target_items)==0:
                    recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
                    st.success("No direct matches. " + msg)
                    st.write("Recommendations (itemid):", recs)
                else:
                    # rank by popularity among those target items
                    pop = (
                        events[(events["event"]=="view") & (events["itemid"].astype(str).isin(pd.Series(target_items).astype(str)))]
                        ["itemid"].value_counts().index.astype(str).tolist()[:topN]
                    )
                    if len(pop)==0:
                        recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
                        st.success("Sparse segment. " + msg)
                        st.write("Recommendations (itemid):", recs)
                    else:
                        st.success("CNN-driven property match recommendations:")
                        st.write("Recommendations (itemid):", pop)
        else:
            recs, msg = recommend_content_based(events, item_props, chosen_user, topn=topN)
            st.success(msg)
            st.write("Recommendations (itemid):", recs)

# -------------------------
# TAB 3: Anomaly Detection (Task 2)
# -------------------------
with tab3:
    st.subheader("Task 2: Abnormal User Detection")

    # Try to load DL artifacts; fallback to Isolation Forest
    ae_path = st.text_input("CNN Autoencoder (h5)", value="cnn_ae.h5")
    scaler_path = st.text_input("Scaler (pkl)", value="scaler.pkl")

    use_dl = False
    ae_model = None
    scaler = None

    if TF_AVAILABLE and os.path.exists(ae_path) and os.path.exists(scaler_path):
        try:
            ae_model = tf.keras.models.load_model(ae_path)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            use_dl = True
            st.success("Loaded CNN Autoencoder + Scaler.")
        except Exception as e:
            st.warning(f"Could not load DL anomaly detector. Falling back to Isolation Forest. Reason: {e}")
    else:
        st.info("DL artifacts not provided. Using Isolation Forest fallback.")

    user_feats = build_user_features(events)

    if use_dl and (scaler is not None) and (ae_model is not None):
        X_scaled = scaler.transform(user_feats.values)
        X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        recon = ae_model.predict(X_cnn, verbose=0)
        mse = np.mean((X_cnn - recon)**2, axis=(1,2))
        threshold = np.percentile(mse, 98)
        user_feats["recon_error"] = mse
        user_feats["outlier"] = (mse > threshold).astype(int)
        st.metric("Detected outliers", int(user_feats["outlier"].sum()))
        # Plots
        figH = px.histogram(mse, nbins=50, title="Reconstruction Error Distribution (No scientific notation)")
        figH.update_layout(xaxis_tickformat=",", yaxis_tickformat=",")
        figH.add_vline(x=float(threshold), line_color="red", line_dash="dash")
        st.plotly_chart(figH, use_container_width=True)

        figS = go.Figure()
        figS.add_trace(go.Scattergl(y=mse, mode="markers",
                                    marker=dict(color=(mse>threshold), colorscale="RdBu"),
                                    name="Users"))
        figS.add_hline(y=float(threshold), line_color="red", line_dash="dash")
        figS.update_layout(
            title="User Reconstruction Error (Outlier Detection)",
            xaxis_title="User Index",
            yaxis_title="Reconstruction Error (MSE)",
            yaxis_tickformat=",",
            xaxis_tickformat=","
        )
        st.plotly_chart(figS, use_container_width=True)

        # Inspect a user
        u_sel = st.selectbox("Inspect user", user_feats.index.astype(str))
        if u_sel:
            st.write(user_feats.loc[u_sel])

    else:
        # Isolation Forest fallback
        iso = IsolationForest(contamination=0.02, random_state=42)
        preds = iso.fit_predict(user_feats)
        user_feats["outlier"] = (preds == -1).astype(int)
        st.metric("Detected outliers (IForest)", int(user_feats["outlier"].sum()))
        # Simple score: anomaly score from decision_function (lower -> more anomalous)
        scores = -iso.score_samples(user_feats.drop(columns=["outlier"]))
        user_feats["anomaly_score"] = scores

        figH = px.histogram(scores, nbins=50, title="Isolation Forest Anomaly Scores")
        figH.update_layout(xaxis_tickformat=",", yaxis_tickformat=",")
        st.plotly_chart(figH, use_container_width=True)

        # Scatter
        figS = go.Figure()
        figS.add_trace(go.Scattergl(y=scores, mode="markers",
                                    marker=dict(color=user_feats["outlier"], colorscale="RdBu"),
                                    name="Users"))
        figS.update_layout(
            title="Anomaly Scores by User",
            xaxis_title="User Index",
            yaxis_title="Anomaly Score",
            yaxis_tickformat=",",
            xaxis_tickformat=","
        )
        st.plotly_chart(figS, use_container_width=True)

        u_sel = st.selectbox("Inspect user", user_feats.index.astype(str))
        if u_sel:
            st.write(user_feats.loc[u_sel])

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("© Recommendation System • Streamlit UI • CNN Recommender + Autoencoder/IForests • Cleaned CSV Compatible")
