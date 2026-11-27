"""Run from LeafLens_Simple to insert the gallery without replacing app.py."""
from pathlib import Path
import shutil
import ast

base = Path(__file__).resolve().parent
app = base / 'app.py'
if not app.is_file() or not (base / 'leaflens' / 'data.py').is_file():
    raise SystemExit('Run this inside the existing LeafLens_Simple project, after extracting the patch there.')
source = app.read_text(encoding='utf-8')
marker = '# LeafLens training-reference gallery patch'
if marker in source:
    print('Training gallery patch already installed.')
    raise SystemExit(0)
anchor = "st.markdown('<p class=\"app-footer\">LeafLens · AI-assisted identification, not a confirmed diagnosis.</p>',unsafe_allow_html=True)"
if source.count(anchor) != 1 or 'def cached_manifest(' not in source or 'def abs_path(' not in source:
    raise SystemExit('Your app.py differs from the supplied LeafLens source. No changes made. Upload your current app.py for a compatible patch.')
block = '''# LeafLens training-reference gallery patch
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

'''
updated = source.replace(anchor, block + anchor)
ast.parse(updated)
backup = app.with_name('app.py.before_training_gallery.bak')
if not backup.exists():
    shutil.copy2(app, backup)
app.write_text(updated, encoding='utf-8')
print('PASS: training-reference gallery installed. Existing app.py backed up to app.py.before_training_gallery.bak')
