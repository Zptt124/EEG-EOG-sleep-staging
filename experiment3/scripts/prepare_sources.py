"""Prepare pinned dependencies and metadata; optional official backbone download."""
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
B_COMMIT='6e7df9978c48c84bbef4d2f7e9d66e9ccd93b923'
EEGDM_COMMIT='003b5a2a016b2f02788ea5693d8b7cd00b6bd969'
CKPT_SHA='0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef'

def git(*args,cwd=REPO):
    return subprocess.run(['git',*args],cwd=cwd,check=True,capture_output=True).stdout

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',action='store_true',help='Also download pinned official source and 95MB checkpoint')
    args=parser.parse_args()
    vendor=ROOT/'vendor'
    vendor.mkdir(exist_ok=True)
    target=vendor/'experiment2-review-6e7df99'
    stamp=target/'SOURCE_COMMIT.txt'
    if not stamp.exists():
        try:
            git('cat-file','-e',B_COMMIT+'^{commit}')
        except subprocess.CalledProcessError:
            git('fetch','origin',B_COMMIT)
        archive=git('archive','--format=zip',B_COMMIT,'manifests','reports','src','scripts','tests')
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for item in z.infolist():
                destination=(target/item.filename).resolve()
                if not destination.is_relative_to(target.resolve()):
                    raise ValueError('Unexpected archive path')
            z.extractall(target)
        stamp.write_text(B_COMMIT+'\n',encoding='utf-8')
    if stamp.read_text().strip()!=B_COMMIT:
        raise ValueError('B snapshot version mismatch')
    if (REPO/'manifests/split_subjects.csv').read_bytes()!=(target/'manifests/split_subjects.csv').read_bytes():
        # Accept platform line-ending changes, but no changes to split contents.
        if (REPO/'manifests/split_subjects.csv').read_text()!=(target/'manifests/split_subjects.csv').read_text():
            raise ValueError('A/B split mismatch')
    features=ROOT/'features/eeg_cond14_v01'
    features.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'manifests/eeg_cond14_v01').iterdir():
        if source.name=='summary.json':
            # A published summary describes the external cache, not a local completed cache.
            continue
        destination=features/source.name
        if destination.exists() and destination.read_bytes()!=source.read_bytes():
            if destination.read_text()!=source.read_text():
                raise ValueError(f'Existing metadata differs: {destination.name}')
        if not destination.exists():
            shutil.copy2(source,destination)
    if args.backbone:
        upstream=vendor/'EEGDM'
        if not upstream.exists():
            git('clone','--no-checkout','https://github.com/jhpuah/EEGDM.git',str(upstream))
            git('checkout','--detach',EEGDM_COMMIT,cwd=upstream)
        if git('rev-parse','HEAD',cwd=upstream).decode().strip()!=EEGDM_COMMIT:
            raise ValueError('EEGDM source version mismatch')
        checkpoint=ROOT/'checkpoints/backbone.ckpt'
        checkpoint.parent.mkdir(exist_ok=True)
        if not checkpoint.exists():
            tmp=checkpoint.with_suffix('.download')
            urllib.request.urlretrieve('https://huggingface.co/jhpuah/eegdm/resolve/main/backbone.ckpt',tmp)
            with tmp.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=CKPT_SHA:
                    raise ValueError('Downloaded checkpoint hash mismatch')
            tmp.replace(checkpoint)
        with checkpoint.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=CKPT_SHA:
                raise ValueError('Checkpoint hash mismatch')
    print(json.dumps(dict(status='ready',b_commit=B_COMMIT,backbone_prepared=args.backbone,
        full_feature_files_included=False)))

if __name__=='__main__':
    main()
