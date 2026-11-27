"""LeafLens — simple localhost GUI for plant crop–disease classification.

Requires frozen PyTorch checkpoints and training label metadata for
actual predictions. Training photographs are not required.
Run: python -m streamlit run app.py
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from leaflens.attention import gradcam_for_prediction, model_view
from leaflens.heatmap import vivid_gradcam
from leaflens.data import class_catalogue, load_training_manifest
from leaflens.model import load_p2_checkpoint, load_prototypes, predict_p2
from leaflens.dual_model import (load_pair, predict_pair, load_ensemble, predict_ensemble,
                                 checkpoint_paths_ready, FINAL_ARMS)

BASE = Path(__file__).resolve().parent
ASSETS = BASE / 'assets'
st.set_page_config(page_title='LeafLens | Plant Disease Identification', page_icon='🍃',
                   layout='wide', initial_sidebar_state='collapsed')
st.markdown(f'<style>{(ASSETS / "style.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)

CONFIG = json.loads((BASE/'config.example.json').read_text(encoding='utf-8'))
if (BASE/'config.json').exists():
    try:
        CONFIG.update(json.loads((BASE/'config.json').read_text(encoding='utf-8')))
    except (OSError,ValueError) as exc:
        st.error(f'Could not read config.json: {exc}')
CONFIG.setdefault('inference_head','P2')


def abs_path(s: str) -> Path:
    p=Path(s).expanduser()
    return p if p.is_absolute() else BASE / p


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def image_data(im: Image.Image, max_side: int = 980) -> str:
    im=ImageOps.exif_transpose(im).convert('RGB')
    im.thumbnail((max_side,max_side))
    buffer=io.BytesIO()
    im.save(buffer, 'JPEG', quality=88, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def render_header(connected: bool):
    status='MODEL READY' if connected else 'MODEL NOT CONNECTED'
    st.markdown(f'''<div class="brand-bar">
        <div class="brand-lockup"><span class="brand-icon">❧</span>
        <div><div class="brand-name">LeafLens</div><div class="brand-subtitle">PLANT &amp; DISEASE IDENTIFICATION</div></div></div>
        <div class="top-status"><i class="{'ready' if connected else 'offline'}"></i>{status}</div>
    </div>''', unsafe_allow_html=True)


def render_steps(stage: int):
    labels=['Upload image', 'AI analysis', 'View result']
    rows=[]
    for i,label in enumerate(labels):
        done=(stage>i)
        active=(stage==i)
        cls='complete' if done else ('active' if active else 'waiting')
        sym='✓' if done else str(i+1)
        rows.append(f'<div class="step {cls}"><span class="step-circle">{sym}</span><span class="step-title">{label}</span></div>')
    st.markdown('<div class="stepper"><div class="stepper-track"></div>' + ''.join(rows) + '</div>',unsafe_allow_html=True)


def render_photo(image: Image.Image | None, view: str, cam):
    if image is None:
        st.markdown('''<div class="photo-empty"><div class="upload-glyph">✧</div>
            <h2>Upload a leaf photograph</h2><p>Choose an image to identify its crop and disease.</p></div>''',unsafe_allow_html=True)
        return
    shown=image
    if st.session_state.prediction is not None and view=='AI View':
        shown=model_view(image, size=st.session_state.prediction['size'],
                         resize_short=st.session_state.prediction['resize_short'])
    badge='YOUR IMAGE' if st.session_state.prediction is None or view=='Normal View' else 'AI VIEW · DISEASE ATTRIBUTION'
    has_signal=True
    if st.session_state.prediction is not None and view=='AI View':
        if cam is not None:
            shown,has_signal=vivid_gradcam(shown,cam)
        else:
            has_signal=False
    src=image_data(shown)
    st.markdown(f'<div class="photo-wrap"><div class="photo-chip">{badge}</div><img src="{src}" alt="Uploaded plant photograph"></div>',unsafe_allow_html=True)
    if st.session_state.prediction is not None and view=='AI View':
        if has_signal:
            st.markdown('''<div class="heat-note"><div class="heat-legend"><span class="heat-key"></span><span>Low → high</span></div><div><b>Disease attribution</b><p>Warmer colours indicate greater model sensitivity, not confirmed lesions.</p></div></div>''',unsafe_allow_html=True)
        else:
            st.warning('The model produced no usable attribution signal for this prediction. Showing the image without an overlay.')


def display_label(label: str) -> str:
    return str(label).replace('_', ' ').strip().capitalize()


def composition_status(records, crop, disease):
    if any(r.crop == crop and r.disease == disease for r in records):
        return 'Seen in training', 'This predicted pairing occurs in the supplied training metadata.'
    if len(records) == 8000:
        return 'Unseen in training', 'This predicted pairing is absent from the supplied 8,000 training records. This does not establish prediction correctness or official-test membership.'
    return 'Not found in metadata', 'The pairing is absent from the supplied records; the full training set has not been confirmed.'


def score_text(score):
    return f'{float(score):.1%}'


def render_result(result: dict | None):
    if result is None:
        st.markdown('<div class="result-shell idle"><div class="result-eyebrow">ANALYSIS RESULTS</div><h2>Ready to identify your leaf</h2><p>Upload a leaf photograph, then select Analyse Image to identify the crop and disease.</p><div class="idle-decoration"></div></div>', unsafe_allow_html=True)
        return
    crop_label, disease_label = display_label(result['crop']), display_label(result['disease'])
    crop_score = result.get('crop_score')
    crop_score_html = f'<small class="score-caption">Model score <b>{score_text(crop_score)}</b></small>' if crop_score is not None else ''
    disease_score = result.get('disease_score')
    disease_score_html = f'<small class="score-caption">Model score <b>{score_text(disease_score)}</b></small>' if disease_score is not None else ''
    rows=[]
    for rank, item in enumerate(result.get('disease_top3', []), 1):
        label=esc(display_label(item['label']))
        score=score_text(item['score'])
        width=max(0., min(100., float(item['score'])*100.))
        tag='<span class="top-choice">Top prediction</span>' if rank == 1 else ''
        rows.append(f'<div class="rank-row"><span class="rank-number">{rank}</span><div class="rank-main"><div class="rank-label">{label}{tag}</div><div class="rank-track"><div class="rank-fill" style="width:{width:.3f}%"></div></div></div><strong class="rank-score">{score}</strong></div>')
    ranking = ('<section class="rank-section"><h3>Top 3 disease predictions</h3><p>Ranked across all 175 disease classes.</p>' + ''.join(rows) + '<p class="score-note">Scores are normalised model outputs, not calibrated diagnostic probabilities.</p></section>') if rows else ''
    status, note = result.get('composition_status', ('', ''))
    composition = (f'<section class="composition-section"><h3>Predicted composition status</h3><strong>{esc(status)}</strong><p>{esc(note)}</p></section>') if status else ''
    # No blank lines or indentation within the rendered HTML block.
    result_html = (
        '<div class="result-shell"><div class="result-eyebrow">ANALYSIS RESULTS <span class="result-status">✓ Complete</span></div>'
        '<div class="result-kicker">Predicted crop–disease combination</div>'
        f'<h1 class="main-prediction">{esc(crop_label)}<span class="prediction-separator"> / </span>{esc(disease_label)}</h1>'
        '<div class="prediction-grid">'
        f'<div class="prediction-item"><span>Crop</span><strong>{esc(crop_label)}</strong>{crop_score_html}</div>'
        f'<div class="prediction-item"><span>Disease</span><strong>{esc(disease_label)}</strong>{disease_score_html}</div>'
        '</div>' + ranking + composition + '</div>'
    )
    st.markdown(result_html, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cached_model(path: str, timestamp: int):
    return load_p2_checkpoint(path)


@st.cache_data(show_spinner=False)
def cached_manifest(csv_path: str, stamp: int, root: str):
    return load_training_manifest(csv_path, root, require_images=False)


@st.cache_resource(show_spinner=False)
def cached_pair(crop_path, crop_stamp, disease_path, disease_stamp):
    return load_pair(crop_path, disease_path)


def predict_dual(image, config, notify=None):
    crop_path=abs_path(config['crop_checkpoint_path'])
    disease_path=abs_path(config['disease_checkpoint_path'])
    csv_path=abs_path(config['training_csv'])
    root=abs_path(config['training_image_root'])
    for p in (crop_path,disease_path,csv_path):
        if not p.is_file():
            raise FileNotFoundError('Required file not found: '+str(p))
    if notify: notify(15,'Loading ConvNeXt and CLIP checkpoints…')
    records=cached_manifest(str(csv_path),csv_path.stat().st_mtime_ns,str(root))
    crops,diseases=class_catalogue(records)
    if (len(crops),len(diseases))!=(55,175):
        raise ValueError('Expected original training CSV with 55 crop and 175 disease IDs.')
    pair=cached_pair(str(crop_path),crop_path.stat().st_mtime_ns,str(disease_path),disease_path.stat().st_mtime_ns)
    if notify: notify(42,'Predicting crop and disease; computing CLIP attribution…')
    result=predict_pair(pair,image,crops,diseases)
    if notify: notify(88,'Checking predicted composition against training metadata…')
    result.update(head='ConvNeXt / CLIP P2 · raw', size=224, resize_short=256,
                  composition_status=composition_status(records,result['crop'],result['disease']))
    return result


@st.cache_resource(show_spinner=False)
def cached_ensemble(entries):
    return load_ensemble({arm:path for arm,path,stamp in entries})


def final_paths(config):
    return {arm:abs_path(config['ensemble_checkpoints'][arm]) for arm in FINAL_ARMS}


def predict_final(image, config, notify=None):
    paths=final_paths(config)
    csv_path=abs_path(config['training_csv'])
    root=abs_path(config['training_image_root'])
    if not csv_path.is_file():
        raise FileNotFoundError('Required training CSV: '+str(csv_path))
    for arm,p in paths.items():
        if not p.is_file():
            raise FileNotFoundError('Missing final Stage-2 checkpoint: '+str(p))
        if p.stat().st_size != FINAL_ARMS[arm][3]:
            raise ValueError('Incomplete or unexpected release checkpoint: '+str(p))
    if notify: notify(15,'Loading four final Stage-2 ensemble checkpoints…')
    records=cached_manifest(str(csv_path),csv_path.stat().st_mtime_ns,str(root))
    crops,diseases=class_catalogue(records)
    if len(records)!=8000 or (len(crops),len(diseases))!=(55,175):
        raise ValueError('Expected the original 8,000-row training CSV, with 55 crop and 175 disease IDs.')
    members=cached_ensemble(tuple((arm,str(p),p.stat().st_mtime_ns) for arm,p in paths.items()))
    if notify: notify(42,'Predicting with the four-model ensemble and computing disease attribution…')
    result=predict_ensemble(members,image,crops,diseases)
    if notify: notify(88,'Checking predicted composition against training metadata…')
    result.update(head='ConvNeXt–CLIP P3 ensemble · raw ranking', size=224, resize_short=256,
                  composition_status=composition_status(records,result['crop'],result['disease']))
    return result


def predict(image: Image.Image, config: dict, notify=None):
    if config.get('model_mode')=='convnext_clip_ensemble':
        return predict_final(image,config,notify)
    if config.get('model_mode')=='convnext_clip':
        return predict_dual(image,config,notify)
    ckpt=abs_path(config['checkpoint_path'])
    csv_path=abs_path(config['training_csv'])
    root=abs_path(config['training_image_root'])
    if not ckpt.is_file():
        raise FileNotFoundError('Trained model is not connected. Place your Stage 2 checkpoint at ' + str(ckpt))
    if not csv_path.is_file():
        raise FileNotFoundError('Training CSV is not connected. Add original train.csv at ' + str(csv_path))
    if notify: notify(15, 'Loading trained model and training label metadata…')
    records=cached_manifest(str(csv_path),csv_path.stat().st_mtime_ns,str(root))
    crops,diseases=class_catalogue(records)
    if len(crops)!=55 or len(diseases)!=175:
        raise ValueError(f'Expected canonical class mapping: 55 crops and 175 diseases. Found {len(crops)} crops and {len(diseases)} diseases.')
    model=cached_model(str(ckpt),ckpt.stat().st_mtime_ns)
    head=config.get('inference_head','P2')
    if head not in ('P2','P3'):
        raise ValueError('inference_head must be P2 or P3 in config.json')
    prototypes=None
    if head=='P3':
        path=abs_path(config['prototypes_path'])
        if not path.is_file():
            raise FileNotFoundError('P3 prototypes missing: '+str(path))
        prototypes=load_prototypes(path)
    size=int(config['model_input_size'])
    short=int(config['resize_shorter_side'])
    if notify: notify(42,'Predicting the crop–disease class…')
    pred=predict_p2(model,image,crops,diseases,head=head,prototypes=prototypes,size=size,resize_short=short,top_k=1)
    ci=crops.index(pred.crop); di=diseases.index(pred.disease)
    if notify: notify(70,'Creating the AI attention view…')
    cam=gradcam_for_prediction(model,image,crop_index=ci,disease_index=di,
                               head=head,prototypes=prototypes,explain='Disease',size=size,resize_short=short)
    return {'crop':pred.crop,'disease':pred.disease,'head':head,
            'size':size,'resize_short':short,'cam':cam,
            'composition_status':composition_status(records,pred.crop,pred.disease)}


for key,value in {
    'image':None,'upload_sha':'','source_name':'','prediction':None,'error':'',
    'clear_id':0,'analysis_id':0
}.items():
    if key not in st.session_state:
        st.session_state[key]=value

if CONFIG.get('model_mode')=='convnext_clip_ensemble':
    is_ready=abs_path(CONFIG['training_csv']).is_file() and checkpoint_paths_ready(final_paths(CONFIG))
else:
    is_ready=abs_path(CONFIG['training_csv']).is_file() and (all(abs_path(CONFIG[k]).is_file() for k in ('crop_checkpoint_path','disease_checkpoint_path')) if CONFIG.get('model_mode')=='convnext_clip' else abs_path(CONFIG['checkpoint_path']).is_file())
render_header(is_ready)
workflow_stage=3 if st.session_state.prediction is not None else (1 if st.session_state.image is not None else 0)
render_steps(workflow_stage)

left,right=st.columns([1,1],gap='large')
with left:
    st.markdown('<div class="section-label">Leaf photograph</div>',unsafe_allow_html=True)
    if st.session_state.prediction is not None:
        current_view=st.radio('Image view', ['Normal View','AI View'],horizontal=True,index=1,
                 key=f'view_{st.session_state.analysis_id}',label_visibility='collapsed')
    else:
        current_view='Normal View'
    render_photo(st.session_state.image,current_view,
                 st.session_state.prediction['cam'] if st.session_state.prediction is not None else None)

    uploaded=st.file_uploader('Upload plant image',type=['jpg','jpeg','png','webp'],
                              accept_multiple_files=False,key=f'upload_{st.session_state.clear_id}',
                              label_visibility='collapsed')
    if uploaded is not None:
        raw=uploaded.getvalue()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=st.session_state.upload_sha:
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    picture=ImageOps.exif_transpose(im).convert('RGB')
                    picture.load()
                st.session_state.image=picture
                st.session_state.source_name=uploaded.name
                st.session_state.upload_sha=digest
                st.session_state.prediction=None
                st.session_state.analysis_id+=1
                st.session_state.error=''
                st.rerun()
            except (UnidentifiedImageError,OSError,ValueError) as exc:
                st.error('Could not open that image: '+str(exc))

    c_clear,c_analyse=st.columns([.65,1.35],gap='small')
    with c_clear:
        clear=st.button('CLEAR',use_container_width=True)
    with c_analyse:
        analyse=st.button('ANALYSE IMAGE',type='primary',use_container_width=True)
    if clear:
        st.session_state.image=None
        st.session_state.source_name=''
        st.session_state.upload_sha=''
        st.session_state.prediction=None
        st.session_state.error=''
        st.session_state.analysis_id+=1
        st.session_state.clear_id+=1
        st.rerun()
    if analyse:
        if st.session_state.image is None:
            st.session_state.error='Please upload a plant image first.'
        else:
            progress=st.progress(0,text='Preparing image…')
            try:
                def report(value:int,message:str):
                    progress.progress(value,text=message)
                result=predict(st.session_state.image,CONFIG,notify=report)
                progress.progress(100,text='Analysis complete')
                st.session_state.prediction=result
                st.session_state.analysis_id+=1
                st.session_state.error=''
                st.rerun()
            except Exception as exc:
                progress.empty()
                st.session_state.prediction=None
                st.session_state.error=f'{type(exc).__name__}: {exc}'
    if st.session_state.error:
        st.error(st.session_state.error)

with right:
    st.markdown('<div class="section-label">Identification</div>',unsafe_allow_html=True)
    render_result(st.session_state.prediction)

# LeafLens training-reference gallery patch
if st.session_state.prediction is not None:
    try:
        from leaflens.training_gallery import gallery_html
        gallery_csv = abs_path(CONFIG['training_csv'])
        gallery_root = abs_path(CONFIG['training_image_root'])
        gallery_records = cached_manifest(str(gallery_csv), gallery_csv.stat().st_mtime_ns, str(gallery_root))
        gallery_prediction = st.session_state.prediction
        st.markdown(gallery_html(gallery_records, gallery_prediction['crop'],
                                 gallery_prediction['disease'], gallery_root), unsafe_allow_html=True)
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f'Training reference images unavailable: {exc}')

st.markdown('<p class="app-footer">LeafLens · AI-assisted identification, not a confirmed diagnosis.</p>',unsafe_allow_html=True)
