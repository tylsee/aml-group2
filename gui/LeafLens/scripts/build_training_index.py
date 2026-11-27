"""Optional: build a cosine-similarity reference index from TRAINING images only.

Usage (after installing requirements-model.txt and adding model + training data):
    python scripts/build_training_index.py
This never reads official evaluation data or retrains the model.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from leaflens.data import load_training_manifest
from leaflens.model import load_p2_checkpoint, preprocess


def resolve(path):
    p=Path(path).expanduser()
    return p if p.is_absolute() else BASE/p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='config.json',help='Local JSON config, relative to project root')
    ap.add_argument('--batch-size',type=int,default=8)
    args=ap.parse_args()
    cfg=json.loads((BASE/'config.example.json').read_text())
    user=resolve(args.config)
    if user.is_file():
        cfg.update(json.loads(user.read_text()))
    records=load_training_manifest(resolve(cfg['training_csv']),resolve(cfg['training_image_root']))
    model=load_p2_checkpoint(resolve(cfg['checkpoint_path']))
    import torch
    names=[]; crops=[]; diseases=[]
    batch=[]; batch_names=[]
    failed=[]
    def flush():
        if not batch:
            return
        with torch.inference_mode():
            _,_,zc,zd=model(torch.cat(batch,dim=0))
        names.extend(batch_names)
        crops.extend(zc.cpu().numpy())
        diseases.extend(zd.cpu().numpy())
        batch.clear(); batch_names.clear()
    for i,r in enumerate(records,1):
        try:
            with Image.open(r.image_path) as image:
                image=ImageOps.exif_transpose(image).convert('RGB')
                image.load()
            batch.append(preprocess(image,size=int(cfg['model_input_size']),resize_short=int(cfg['resize_shorter_side'])))
            batch_names.append(r.filename)
        except (OSError,ValueError,UnidentifiedImageError) as e:
            failed.append((r.filename,str(e)))
        if len(batch)>=args.batch_size:
            flush()
        if i%500==0 or i==len(records):
            print(f'Read {i}/{len(records)} training rows; embedded {len(names)}; unreadable: {len(failed)}',flush=True)
    flush()
    if not names:
        raise RuntimeError('No training images were readable. Check training_image_root.')
    out=resolve(cfg['training_embeddings_index'])
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,filenames=np.array(names),crop_embeddings=np.stack(crops).astype('float32'),disease_embeddings=np.stack(diseases).astype('float32'))
    print(f'Saved TRAINING-ONLY embedding index: {out} ({len(names)} images)')
    if failed:
        print('Unreadable training images (first 15):',failed[:15])

if __name__=='__main__':main()
