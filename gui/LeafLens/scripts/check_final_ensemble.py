"""Offline checkpoint verification and optional smoke test (no training)."""
import argparse, json, sys, hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from leaflens.dual_model import FINAL_ARMS, load_ensemble, predict_ensemble
from leaflens.data import load_training_manifest, class_catalogue
from PIL import Image

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image',help='Optional real photo to run prediction and attribution on.')
    ap.add_argument('--sha256-file',help='Optional SHA256SUMS.txt downloaded from the GitHub release.')
    args=ap.parse_args()
    config=json.loads((BASE/'config.example.json').read_text())
    if (BASE/'config.json').exists(): config.update(json.loads((BASE/'config.json').read_text()))
    if config['model_mode']!='convnext_clip_ensemble': raise ValueError('Active config selects a legacy model. Set model_mode to convnext_clip_ensemble.')
    paths={arm:BASE/Path(config['ensemble_checkpoints'][arm]).expanduser() for arm in FINAL_ARMS}
    hashes={}
    if args.sha256_file:
        for line in Path(args.sha256_file).read_text().splitlines():
            if line.strip():
                h,n=line.split(None,1); hashes[Path(n.strip().lstrip('*')).name]=h.lower()
    for arm,path in paths.items():
        expected=FINAL_ARMS[arm][3]
        if not path.is_file() or path.stat().st_size!=expected: raise ValueError(f'Missing or incomplete {path} (expected {expected} bytes)')
        if hashes:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
            if h.hexdigest()!=hashes.get(path.name): raise ValueError(f'SHA256 mismatch or missing checksum: {path.name}')
        print(f'OK: {path.name} ({expected:,} bytes)',flush=True)
    records=load_training_manifest(BASE/config['training_csv'],BASE/config['training_image_root'],require_images=False)
    crops,diseases=class_catalogue(records)
    assert (len(records),len(crops),len(diseases))==(8000,55,175)
    ensemble=load_ensemble(paths)
    print('PASS: all four Stage-2 checkpoints loaded strictly; canonical training labels verified.',flush=True)
    if args.image:
        with Image.open(args.image) as im: result=predict_ensemble(ensemble,im,crops,diseases)
        print('Crop:',result['crop'],result['crop_score'])
        print('Disease:',result['disease'],result['disease_score'])
        print('Top 3:',result['disease_top3'])
        print('Attribution shape:',result['cam'].shape)
        print('PASS: real-image smoke test. This is not official-test accuracy.')
if __name__=='__main__': main()
