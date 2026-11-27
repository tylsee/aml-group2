"""LeafLens Streamlit interface for the final four-model ensemble."""
from pathlib import Path
import base64
import hashlib
import html
import io
from PIL import Image, ImageOps
import streamlit as st
from ensemble import paths, check_files, load_ensemble, predict, overlay

BASE=Path(__file__).resolve().parent
st.set_page_config(page_title="LeafLens | Plant Disease Identification",layout="wide",initial_sidebar_state="collapsed")
st.markdown("<style>"+(BASE/"assets/style.css").read_text()+"</style>",unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def cached_model(stamps):
    return load_ensemble()

def label(x):
    return html.escape(x.replace("_"," ").capitalize())

def image_html(im,caption):
    im=ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((980,980))
    buf=io.BytesIO()
    im.save(buf,format="JPEG",quality=88)
    data=base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(f'<div class="photo-wrap"><div class="photo-chip">{caption}</div><img src="data:image/jpeg;base64,{data}" alt="Leaf photograph"></div>',unsafe_allow_html=True)

try:
    check_files(False)
    ready=True
except ValueError as exc:
    ready=False
    missing=str(exc)

st.markdown('<div class="brand-bar"><div><div class="brand-name">❧ LeafLens</div><div class="brand-subtitle">PLANT &amp; DISEASE IDENTIFICATION</div></div><div class="top-status">'+("MODEL FILES READY" if ready else "MODEL NOT CONNECTED")+'</div></div>',unsafe_allow_html=True)
st.markdown('<div class="stepper"><span>① Upload image</span><span>② AI analysis</span><span>③ View result</span></div>',unsafe_allow_html=True)
if not ready:
    st.warning(missing+". Download the four release assets into gui/LeafLens/models/ and restart. See README.md.")
left,right=st.columns(2)
with left:
    st.markdown("**Leaf photograph**")
    upload=st.file_uploader("Upload a leaf photograph",type=["jpg","jpeg","png","webp"])
    image=None
    if upload:
        data=upload.getvalue()
        key=hashlib.sha256(data).hexdigest()
        if st.session_state.get("image_key")!=key:
            st.session_state.image_key=key
            st.session_state.result=None
        try:
            image=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception as exc:
            st.error("Invalid image: "+str(exc))
    else:
        st.session_state.result=None
    result=st.session_state.get("result")
    view=st.radio("Image view",["Normal View","AI View"],horizontal=True)
    if image is None:
        st.markdown('<div class="photo-empty"><h3>Upload a leaf photograph</h3><p>Choose an image to identify its crop and disease.</p></div>',unsafe_allow_html=True)
    elif view=="AI View" and result:
        shown,signal=overlay(image,result["cam"])
        image_html(shown,"AI VIEW · DISEASE ATTRIBUTION")
        st.caption("Warmer colours indicate greater model sensitivity, not confirmed lesions." if signal else "No usable attribution signal; image shown without overlay.")
    else:
        image_html(image,"YOUR IMAGE")
    if st.button("ANALYSE IMAGE",type="primary",disabled=image is None or not ready,use_container_width=True):
        st.session_state.result=None
        try:
            with st.spinner("Loading verified final ensemble and computing disease attribution…"):
                stamps=tuple((a,p.stat().st_mtime_ns,p.stat().st_size) for a,p in paths().items())
                st.session_state.result=predict(cached_model(stamps),image)
            st.rerun()
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
with right:
    st.markdown("**Identification**")
    result=st.session_state.get("result")
    if result is None:
        st.markdown('<div class="result-shell"><div class="result-eyebrow">ANALYSIS RESULTS</div><h2 class="main-prediction">Ready to identify your leaf</h2><p>Upload a leaf photograph, then select Analyse Image.</p></div>',unsafe_allow_html=True)
    else:
        c,d=label(result["crop"]),label(result["disease"])
        cs,ds=result["crop_score"],result["disease_score"]
        rows=""
        for i,(name,score) in enumerate(result["top3"],1):
            rows+=f'<div class="rank-row"><span>{i}</span><div class="rank-main">{label(name)}<div class="rank-track"><div class="rank-fill" style="width:{score*100:.3f}%"></div></div></div><strong class="rank-score">{score:.1%}</strong></div>'
        status="Seen in training" if result["seen"] else "Unseen in training"
        st.markdown(f'<div class="result-shell"><div class="result-eyebrow">ANALYSIS RESULTS · FINAL FOUR-MODEL ENSEMBLE</div><h1 class="main-prediction">{c} / {d}</h1><div class="prediction-grid"><div class="prediction-item"><span>Crop</span><strong>{c}</strong><small>Model score {cs:.1%}</small></div><div class="prediction-item"><span>Disease</span><strong>{d}</strong><small>Model score {ds:.1%}</small></div></div><section class="rank-section"><h3>Top 3 disease predictions</h3>{rows}<p class="score-note">Normalised model scores, not calibrated diagnostic probabilities.</p></section><section class="composition-section"><h3>Predicted composition status</h3><strong>{status}</strong><p>Relative to the 8,000 training images; this does not establish correctness or test-set membership.</p></section></div>',unsafe_allow_html=True)
st.markdown('<p class="app-footer">LeafLens · AI-assisted identification, not a confirmed diagnosis.</p>',unsafe_allow_html=True)
