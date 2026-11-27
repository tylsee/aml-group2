"""Download/verify the four release assets, then activate ensemble mode.
Run from any directory with the project's Python. Requires authenticated GitHub CLI
only when a verified local copy cannot be found. Never deletes legacy checkpoints.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = 'tylsee/aml-group2'
RELEASE = 'new-approach-models'
ASSETS = {
    'crop_b1': (111507819, '427e9d8e7cb0b1dc4cfe002e5bfce9069e4693204d451f32d1233b74eb132df1'),
    'crop_clip': (344964043, '8d3cc6d275b4de2051cc63e437b8ed15f92f5fd2c8dcfdeb2d25e0df3dd12918'),
    'disease_clip': (345238603, 'a8716d5660223f3fcdc7658845b63e4328495204dfeeeca451eeab9a065293e1'),
    'disease_small_reg': (198466235, '3e1c30a2a04e06b15cec58154b98425a4ede4ca348fd973ea3014a7e50ab3fc7'),
}

def verified(path, spec):
    if not path.is_file() or path.stat().st_size != spec[0]:
        return False
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() == spec[1]

def activate_config():
    path = BASE / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    defaults = json.loads((BASE / 'config.example.json').read_text(encoding='utf-8'))
    # Keep all user preferences, especially their external training_image_root.
    config['model_mode'] = 'convnext_clip_ensemble'
    config['ensemble_checkpoints'] = defaults['ensemble_checkpoints']
    if path.exists():
        backup = BASE / 'config.json.before_final_ensemble.bak'
        if not backup.exists(): shutil.copy2(path, backup)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)
    print('Activated final ensemble. Training image root:', config.get('training_image_root', defaults['training_image_root']))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only', action='store_true', help='Do not download; look in models and Downloads only.')
    args = parser.parse_args()
    folder = BASE / 'models'; folder.mkdir(exist_ok=True)
    for arm, spec in ASSETS.items():
        name = arm + '_best_train_acc.pt'; dest = folder / name
        if verified(dest, spec):
            print('Verified existing:', name, flush=True); continue
        candidate = Path.home() / 'Downloads' / name
        # Download into a separate temporary directory; never leave a partial file in models.
        with tempfile.TemporaryDirectory(prefix='leaflens-', dir=folder) as temp:
            fresh = Path(temp) / name
            if verified(candidate, spec):
                print('Copying verified Downloads file:', name, flush=True)
                shutil.copy2(candidate, fresh)
            elif args.local_only:
                raise SystemExit(f'No verified local copy: {name}. Download it from https://github.com/{REPO}/releases/tag/{RELEASE}.')
            else:
                if not shutil.which('gh'):
                    raise SystemExit('Install GitHub CLI (brew install gh on macOS, winget install --id GitHub.cli on Windows), then gh auth login. Or download manually and rerun --local-only.')
                print('Downloading:', name, flush=True)
                subprocess.run(['gh', 'release', 'download', RELEASE, '-R', REPO, '-D', temp, '-p', name], check=True)
            if not verified(fresh, spec):
                raise SystemExit(f'Checksum or size mismatch: {name}. Existing file not replaced.')
            if dest.exists():
                backup = dest.with_name(dest.name + '.invalid.bak')
                if backup.exists(): raise SystemExit(f'Move existing invalid file out of models first: {dest}')
                dest.rename(backup)
            fresh.replace(dest)
        print('Verified:', name, flush=True)
    activate_config()
    print('All four release assets verified by byte size and SHA-256. Run scripts/check_final_ensemble.py next.')

if __name__ == '__main__': main()
