"""Verify release hashes and load strictly. Optional real-image smoke test."""
import argparse
from PIL import Image
from ensemble import check_files, load_ensemble, predict
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--image")
    args=p.parse_args()
    check_files()
    print("PASS: all four SHA-256 digests match published release",flush=True)
    model=load_ensemble()
    print("PASS: all four Stage-2 models loaded strictly",flush=True)
    if args.image:
        with Image.open(args.image) as im:
            result=predict(model,im)
        print({k:v for k,v in result.items() if k!="cam"})
