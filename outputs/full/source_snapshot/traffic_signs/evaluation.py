"""Held-out metrics, fixed-model perturbations and traceable example selection."""
import json
import time
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from .features import extract_features, preprocess, feature_vector
from .labels import CLASS_NAMES
from .plotting import plt, save, gallery, heatmap
from PIL import Image

def metrics(y,p,classes):
    pr,re,f,_=precision_recall_fscore_support(y,p,labels=classes,average='macro',zero_division=0)
    return dict(accuracy=accuracy_score(y,p),macro_precision=pr,macro_recall=re,macro_f1=f)

def roi_of(row):
    return tuple(int(row[k]) for k in ['x1','y1','x2','y2'])

def predict_file(bundle,path,roi=None):
    c=bundle['config']
    if c['use_roi'] and roi is None:
        raise ValueError('单图预测没有 ROI 元数据；请使用 use_roi=false 的完整流程模型。')
    v=feature_vector(preprocess(path,c,roi=roi),c,bundle['kind'])
    return int(bundle['pipeline'].predict(v[None,:])[0])

def evaluate(manifest,config,bundles):
    out=Path(config['output_dir']); assets=Path(config['assets_dir'])/'04_实验分析'
    assets.mkdir(parents=True,exist_ok=True)
    test=manifest[manifest.split=='test'].reset_index(drop=True)
    classes=config['classes']; y=test.label.to_numpy(); rows=[]
    predictions={}
    verification={}
    for name,bundle in bundles.items():
        X=extract_features(test,config,bundle['kind'])
        t=time.perf_counter()
        with threadpool_limits(limits=4):
            pred=bundle['pipeline'].predict(X)
        seconds=time.perf_counter()-t
        row={'model':name,**metrics(y,pred,classes),'train_seconds':bundle['timings']['train_seconds'],
             'predict_seconds':seconds,'test_count':len(y),'predict_ms_per_image':seconds*1000/len(y)}
        rows.append(row); predictions[name]=pred
        frame=test.copy();frame['prediction']=pred;frame['correct']=y==pred
        frame.to_csv(out/f'{name}_test_predictions.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(classification_report(y,pred,labels=classes,output_dict=True,zero_division=0)).T.to_csv(out/f'{name}_class_metrics.csv',encoding='utf-8-sig')
        cm=confusion_matrix(y,pred,labels=classes)
        pd.DataFrame(cm,index=classes,columns=classes).to_csv(out/f'{name}_confusion.csv')
        heatmap(cm,[str(i) for i in classes],assets/f'{name}_confusion',f'{name} 官方测试集混淆矩阵（类别 ID）')
        loaded=joblib.load(out/'models'/f'{name}.joblib')
        verification[name]={'reload_batch_predictions_identical':bool(np.array_equal(pred[:32],loaded['pipeline'].predict(X[:32]))),
                           'single_image_matches_batch':predict_file(loaded,test.iloc[0].path,roi_of(test.iloc[0]) if config['use_roi'] else None)==int(pred[0])}
    results=pd.DataFrame(rows);results.to_csv(out/'metrics.csv',index=False,encoding='utf-8-sig')
    results.to_csv(assets/'metrics.csv',index=False,encoding='utf-8-sig')
    for fields,filename,title in [(['accuracy','macro_f1'],'model_scores','官方测试集：准确率与宏平均 F1'),(['train_seconds','predict_seconds'],'model_times','最终拟合与整批测试预测耗时（秒）')]:
        fig,ax=plt.subplots(figsize=(10,5));results.set_index('model')[fields].plot.bar(ax=ax,rot=0)
        ax.set_title(title);ax.set_ylabel('分数' if filename=='model_scores' else '秒')
        if filename=='model_scores':ax.set_ylim(0,1.08)
        for container in ax.containers:ax.bar_label(container,fmt='%.3f',fontsize=10)
        save(fig,assets/filename)
    fig,ax=plt.subplots(figsize=(14,3));ax.axis('off')
    display=results[['model','accuracy','macro_f1','train_seconds','predict_seconds']].copy()
    for col in display.columns[1:]:display[col]=display[col].map(lambda x:f'{x:.4f}')
    table=ax.table(cellText=display.values,colLabels=['方法','准确率','宏平均 F1','拟合 / 秒','预测 / 秒'],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(12);table.scale(1,2)
    ax.set_title('同一官方测试集；耗时不含特征提取与验证选参')
    save(fig,assets/'metrics_table')
    pred=predictions['hog_svm'];cm=confusion_matrix(y,pred,labels=classes)
    off=cm.copy();np.fill_diagonal(off,0)
    order=np.argsort(off.ravel())[::-1]; chosen=[];pairs=[]
    for flat in order:
        i,j=np.unravel_index(flat,off.shape)
        if off[i,j]==0:break
        pairs.append({'true_id':classes[i],'predicted_id':classes[j],'count':int(off[i,j])})
        for k in [i,j]:
            if k not in chosen and len(chosen)<8:chosen.append(k)
        if len(pairs)>=12:break
    pd.DataFrame(pairs).to_csv(out/'top_confusions.csv',index=False,encoding='utf-8-sig')
    if chosen:
        heatmap(cm[np.ix_(chosen,chosen)],[f'{classes[k]} {CLASS_NAMES[classes[k]]}' for k in chosen],assets/'confusion_local','HOG＋SVM 主要混淆类别局部计数（非完整错误分布）',True)
    example_rows=[]
    demo=out/'demo_inputs';demo.mkdir(exist_ok=True)
    for correct in [True,False]:
        idx=np.flatnonzero((y==pred)==correct)
        # Deterministic round-robin by true class: first filename in each class, then fill.
        selected=[]
        for label in classes:
            hits=[int(i) for i in idx if y[i]==label]
            if hits:selected.append(hits[0])
        selected=selected[:12]
        images=[];captions=[]
        for i in selected:
            r=test.iloc[i];im=Image.open(r.path).convert('RGB')
            caption=f'真实：{y[i]} {CLASS_NAMES[y[i]]}\n预测：{pred[i]} {CLASS_NAMES[pred[i]]}'
            images.append(im.copy());captions.append(caption)
            target=demo/f'{"success" if correct else "failure"}_{i:05d}.png';im.save(target)
            gallery([im.copy()],[caption],assets/'examples'/target.stem,cols=1)
            example_rows.append({'input':str(target),'source':r.path,'true_id':int(y[i]),'predicted_id':int(pred[i]),'correct':correct,'selection':'按真实类别 ID 顺序，每类首个匹配测试样本，最多12张'})
        gallery(images,captions,assets/('correct_cases' if correct else 'error_cases'),f'HOG＋SVM {"正确" if correct else "错误"}案例；仅用于解释，不代表整体比例')
    pd.DataFrame(example_rows).to_csv(out/'demo_cases.csv',index=False,encoding='utf-8-sig')
    (out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    if not all(all(v.values()) for v in verification.values()):raise AssertionError('模型重载或单图预测一致性检查失败')
    return results

def robustness(manifest,config,bundle):
    out=Path(config['output_dir']);assets=Path(config['assets_dir'])/'04_实验分析'
    test=manifest[manifest.split=='test']
    parts=[g.sample(min(len(g),config['robust_per_class']),random_state=config['seed']) for _,g in test.groupby('label')]
    subset=pd.concat(parts).sort_values(['label','path']).reset_index(drop=True)
    subset.to_csv(out/'robustness_subset.csv',index=False,encoding='utf-8-sig')
    rows=[];levels={'brightness':[1.,.75,.5,.25,.1],'blur':[0.,.5,1.,1.5,2.]}
    sample=subset.iloc[0]
    from .features import load_rgb
    all_predictions=[]
    for kind,values in levels.items():
        ims=[];caps=[]
        for value in values:
            b=value if kind=='brightness' else 1.;s=value if kind=='blur' else 0.
            X=extract_features(subset,config,'hog',brightness=b,blur=s,cache=False)
            with threadpool_limits(limits=4):p=bundle['pipeline'].predict(X)
            rows.append({'interference':kind,'strength':value,'n':len(subset),**metrics(subset.label,p,config['classes'])})
            tmp=subset[['path','label']].copy();tmp['interference']=kind;tmp['strength']=value;tmp['prediction']=p;all_predictions.append(tmp)
            im=load_rgb(sample.path,config,roi=roi_of(sample) if config['use_roi'] else None,brightness=b,blur=s)
            ims.append(im);caps.append(f'{kind}={value:g}\n真：{sample.label}  预测：{int(p[0])}')
            gallery([im],[caps[-1]],assets/'perturbation_singles'/f'{kind}_{value:g}',cols=1)
        gallery(ims,caps,assets/f'{kind}_same_image',f'同一测试图片；先变暗后模糊，再尺寸统一；当前单独改变 {kind}',cols=5)
    frame=pd.DataFrame(rows);frame.to_csv(out/'robustness.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_predictions).to_csv(out/'robustness_predictions.csv',index=False,encoding='utf-8-sig')
    for kind in levels:
        part=frame[frame.interference==kind]
        fig,ax=plt.subplots(figsize=(9,5))
        for metric,label in [('accuracy','准确率'),('macro_f1','宏平均 F1')]:ax.plot(part.strength,part[metric],marker='o',label=label)
        if kind=='brightness':ax.invert_xaxis()
        ax.set(title=f'固定 HOG＋SVM；每类最多 {config["robust_per_class"]} 张，共 {len(subset)} 张',xlabel='亮度倍率（1 为基线）' if kind=='brightness' else '高斯模糊半径（原始分辨率像素）',ylabel='分数',ylim=(0,1.05))
        ax.legend();ax.grid(alpha=.25);save(fig,assets/f'{kind}_robustness')
    return frame
