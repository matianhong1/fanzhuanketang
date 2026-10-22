"""Download official GTSRB archives without installing any dependencies.

Interrupted transfers use a .part file and HTTP Range when supported.
ZIP CRCs are verified before atomic rename and extraction rejects path traversal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_signs.data import ARCHIVE_BASE, SOURCE, safe_extract

FILES = {'GTSRB_Final_Training_Images.zip':276294756,
         'GTSRB_Final_Test_Images.zip':88978620,
         'GTSRB_Final_Test_GT.zip':99620}

def download(data_dir='data',extract=True):
    root=Path(data_dir).resolve(); raw=root/'raw'; raw.mkdir(parents=True,exist_ok=True)
    provenance={'source':SOURCE,'official_archive_index':ARCHIVE_BASE+'published-archive.html','files':[]}
    for filename, expected_size in FILES.items():
        target=raw/filename; part=raw/(filename+'.part'); url=ARCHIVE_BASE+filename
        if not target.exists():
            for attempt in range(4):
                try:
                    offset=part.stat().st_size if part.exists() else 0
                    req=urllib.request.Request(url,headers={'User-Agent':'GTSRB-course-project/1.0',**({'Range':f'bytes={offset}-'} if offset else {})})
                    with urllib.request.urlopen(req,timeout=90) as response:
                        append=bool(offset and response.status==206)
                        with open(part,'ab' if append else 'wb') as f:
                            while True:
                                chunk=response.read(1024*1024)
                                if not chunk: break
                                f.write(chunk)
                    if part.stat().st_size != expected_size:
                        raise ValueError(f'Unexpected byte count for {filename}: {part.stat().st_size}; expected {expected_size}')
                    with zipfile.ZipFile(part) as z:
                        bad=z.testzip()
                        if bad: raise ValueError(f'ZIP CRC failed: {bad}')
                    part.replace(target)
                    break
                except Exception:
                    if attempt==3: raise
                    time.sleep(2*(attempt+1))
        if target.stat().st_size != expected_size:
            raise ValueError(f'Incorrect existing archive size: {target}; remove/re-download manually')
        with zipfile.ZipFile(target) as z:
            bad=z.testzip()
            if bad: raise ValueError(f'Existing ZIP CRC failed: {target}: {bad}')
        h=hashlib.sha256()
        with target.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        digest=h.hexdigest()
        provenance['files'].append({'filename':filename,'url':url,'bytes':target.stat().st_size,'sha256_observed':digest,'hash_note':'Locally observed SHA256, not publisher-provided authenticity signature'})
        print(f'{filename}: verified {target.stat().st_size} bytes, SHA256 {digest}',flush=True)
        if extract:
            marker=root/'extracted'/(filename+'.extracted')
            if not marker.exists() or marker.read_text(encoding='utf-8')!=digest:
                safe_extract(target,root/'extracted'); marker.write_text(digest,encoding='utf-8')
    (root/'download_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--no-extract',action='store_true')
    args=parser.parse_args()
    try: download(args.data_dir,not args.no_extract)
    except Exception as exc:
        print(f'Download failed: {exc}\nManual source: {ARCHIVE_BASE}published-archive.html\nPlace these archives under {Path(args.data_dir).resolve()/"raw"}: '+', '.join(FILES),file=sys.stderr)
        raise SystemExit(1)
