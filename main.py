"""Command-line entry point; run from repository root."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'4')
os.environ.setdefault('MPLCONFIGDIR',os.path.abspath('.mplconfig'))
import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
import skimage
import matplotlib
import PIL
import threadpoolctl
from threadpoolctl import threadpool_limits
from traffic_signs.data import prepare_data
from traffic_signs.models import train_models
from traffic_signs.evaluation import evaluate, robustness, predict_file
from traffic_signs.materials import foundation,write_report
from traffic_signs.labels import CLASS_NAMES
from traffic_signs.plotting import gallery
from PIL import Image

def main():
    parser=argparse.ArgumentParser(description='GTSRB HOG + SVM 可重复实验')
    parser.add_argument('command',choices=['prepare','train','evaluate','robustness','assets','all','predict'])
    parser.add_argument('--config',default='configs/full.json')
    parser.add_argument('--quick',action='store_true',help='只选择原始 ID 0..9，写入独立 quick 目录')
    parser.add_argument('--prepared',action='store_true',help='复用本次配置已通过的图片审计；仅在数据未更改时使用')
    parser.add_argument('--model',default=None)
    parser.add_argument('--image')
    parser.add_argument('--roi',nargs=4,type=int,metavar=('X1','Y1','X2','Y2'))
    parser.add_argument('--output',default='outputs/prediction.png')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.quick:
        config.update(classes=list(range(10)),output_dir='outputs/quick',assets_dir='presentation_assets/quick')
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if args.command=='predict':
        if not args.image:parser.error('predict requires --image')
        bundle=joblib.load(args.model or out/'models/hog_svm.joblib')
        prediction=predict_file(bundle,args.image,args.roi)
        result={'class_id':prediction,'chinese_name':CLASS_NAMES[prediction],'model':bundle['name'],'input':args.image}
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
        gallery([Image.open(args.image).convert('RGB')],[f'预测：{prediction} {CLASS_NAMES[prediction]}'],target.with_suffix(''),cols=1)
        target.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False));return
    # Every command records configuration; use the same config for a given output directory.
    saved=out/'run_config.json'
    if saved.exists() and json.loads(saved.read_text(encoding='utf-8'))!=config:
        raise ValueError('输出目录已有其他配置，请更改 output_dir，防止混合实验。')
    saved.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    env={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'skimage':skimage.__version__,'matplotlib':matplotlib.__version__,'pillow':PIL.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__,'threads':4,
         'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('main.py'),*Path('traffic_signs').glob('*.py')]}}
    (out/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    if args.command in ['all','train']:
        (out/'training_environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
        for source_name in env['source_sha256']:
            source=Path(source_name);target=out/'source_snapshot'/source
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    if args.prepared:
        audit=json.loads((out/'data_audit.json').read_text(encoding='utf-8'))
        if audit['seed']!=config['seed'] or audit['classes']!=config['classes'] or any(audit[k] for k in ['train_val_track_overlap','cross_split_identical_file_groups','cross_split_identical_rgb_groups','corrupt_or_invalid_images']):
            raise ValueError('已有审计未通过或配置不匹配，请重新 prepare。')
        manifest=pd.read_csv(out/'manifest.csv',encoding='utf-8-sig')
        if manifest.groupby('split').size().to_dict()!=audit['counts']:
            raise ValueError('已有清单计数不匹配，请重新 prepare。')
        print('Reusing completed data audit; source images must remain unchanged.',flush=True)
    else:
        manifest=prepare_data(config)
    if args.command=='prepare':return
    if args.command in ['all','assets']:
        print('Generating data and principle assets...',flush=True);foundation(manifest,config)
    if args.command in ['all','train']:
        print('Training and validation selection...',flush=True)
        with threadpool_limits(limits=4):bundles=train_models(manifest,config)
    elif args.command!='assets':bundles={name:joblib.load(out/'models'/f'{name}.joblib') for name in ['hog_svm','hog_knn','pixels_svm']}
    results=None;robust=None
    if args.command in ['all','evaluate']:
        print('Evaluating official test set...',flush=True);results=evaluate(manifest,config,bundles)
        print(results.to_string(index=False),flush=True)
    if args.command in ['all','robustness']:
        print('Evaluating fixed robustness subset...',flush=True);robust=robustness(manifest,config,bundles['hog_svm'])
    if results is None and (out/'metrics.csv').exists():results=pd.read_csv(out/'metrics.csv')
    if robust is None and (out/'robustness.csv').exists():robust=pd.read_csv(out/'robustness.csv')
    write_report(config,results,robust)
    print(f'Completed: {out}; {config["assets_dir"]}',flush=True)

if __name__=='__main__':
    main()
