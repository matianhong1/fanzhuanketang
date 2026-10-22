"""Assets in the order of the four-person presentation, without slides."""
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from .features import preprocess, load_rgb, hog_visualization, feature_vector
from .labels import CLASS_NAMES
from .plotting import plt, save, gallery

def foundation(manifest,config):
    assets=Path(config['assets_dir']);a=assets/'01_任务与数据';b=assets/'02_特征提取';c=assets/'03_分类模型'
    for d in [a,b,c]:d.mkdir(parents=True,exist_ok=True)
    train=manifest[manifest.split=='train'].sort_values(['label','path'])
    chosen=train.groupby('label',sort=True).head(1)
    ims=[];caps=[]
    for _,r in chosen.iterrows():
        im=Image.open(r.path).convert('RGB');cap=f'{r.label} {CLASS_NAMES[r.label]}'
        ims.append(im.copy());caps.append(cap)
        gallery([im.copy()],[cap],a/'class_singles'/f'class_{r.label:02d}',cols=1)
    gallery(ims,caps,a/'class_samples','训练集各类别首个文件样例（不按清晰度挑选）',cols=6)
    for start in range(0,len(ims),12):
        gallery(ims[start:start+12],caps[start:start+12],a/f'class_samples_page_{start//12+1:02d}','训练集类别样例（分页用于投影）',cols=4)
    counts=pd.crosstab(manifest.label,manifest.split).reindex(config['classes'])
    counts.to_csv(a/'split_counts.csv',encoding='utf-8-sig')
    for split in ['train','val','test']:
        fig,ax=plt.subplots(figsize=(15,5));ax.bar(counts.index.astype(str),counts[split])
        ax.set(title=f'{split} 集类别分布：共 {counts[split].sum()} 张',xlabel='类别 ID（中文映射见 class_names.json）',ylabel='样本数')
        ax.tick_params(axis='x',labelsize=9);save(fig,a/f'class_distribution_{split}')
    fig,ax=plt.subplots(figsize=(15,5));counts[['train','val','test']].plot.bar(stacked=True,ax=ax,width=.85)
    ax.set(title='固定轨迹分组划分与官方测试集',xlabel='原始类别 ID',ylabel='样本数');ax.tick_params(axis='x',labelrotation=0,labelsize=9)
    save(fig,a/'class_distribution_all')
    r=chosen.iloc[0];original=Image.open(r.path).convert('RGB');roi=tuple(int(r[k]) for k in ['x1','y1','x2','y2'])
    gray=preprocess(r.path,config,roi=roi if config['use_roi'] else None)
    gallery([original,load_rgb(r.path,config,roi=roi if config['use_roi'] else None),gray],['输入原图（已裁剪标志）',f'统一为 {gray.shape[1]}×{gray.shape[0]}；'+('ROI 裁剪' if config['use_roi'] else '不追加 ROI 裁剪'),'灰度 [0,1]；RGB→L'],a/'preprocessing',cols=3)
    feature_examples=chosen.iloc[np.linspace(0,len(chosen)-1,min(6,len(chosen)),dtype=int)]
    for _,r in feature_examples.iterrows():
        original=Image.open(r.path).convert('RGB')
        roi=tuple(int(r[k]) for k in ['x1','y1','x2','y2'])
        gray=preprocess(r.path,config,roi=roi if config['use_roi'] else None)
        h=hog_visualization(gray,config);hi=np.percentile(h,99)
        h=np.clip(h/(hi if hi else 1),0,1)
        gallery([original,gray,h],['原图',f'{r.label} {CLASS_NAMES[r.label]}\n灰度图','HOG 梯度方向强度（增强显示）'],b/f'hog_triplet_{r.label:02d}',cols=3)
        for label,im in [('original',original),('gray',gray),('hog',h)]:
            gallery([im],[label],b/'singles'/f'{r.label:02d}_{label}',cols=1)
    svm_diagram(c,config['seed'])
    (assets/'class_names.json').write_text(json.dumps(CLASS_NAMES,ensure_ascii=False,indent=2),encoding='utf-8')
    chosen.to_csv(a/'example_selection.csv',index=False,encoding='utf-8-sig')

def svm_diagram(directory,seed):
    X,y=make_blobs(n_samples=40,centers=[[-2,-1],[2,1]],cluster_std=.7,random_state=seed)
    model=SVC(kernel='linear',C=1).fit(X,y)
    xx,yy=np.meshgrid(np.linspace(-4,4,250),np.linspace(-3,3,250));z=model.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(9,6));ax.contourf(xx,yy,z>0,alpha=.12,colors=['#2463A6','#E49A29'])
    ax.scatter(X[:,0],X[:,1],c=y,cmap='coolwarm',s=55,edgecolors='white')
    ax.contour(xx,yy,z,levels=[-1,0,1],colors='#333333',linestyles=['--','-','--'])
    sv=model.support_vectors_;ax.scatter(sv[:,0],sv[:,1],s=180,facecolors='none',edgecolors='black',label='支持向量')
    ax.set(title='二维模拟数据：SVM 分类边界与间隔\n仅解释原理，不代表交通标志真实特征空间',xlabel='模拟特征 1',ylabel='模拟特征 2')
    ax.text(.02,.02,'实线：决策分数 0；虚线：±1；间隔宽度 2/‖w‖',transform=ax.transAxes,fontsize=11)
    ax.legend();save(fig,Path(directory)/'svm_2d_principle')
    pd.DataFrame({'x1':X[:,0],'x2':X[:,1],'label':y,'support_vector':np.isin(np.arange(len(y)),model.support_)}).to_csv(Path(directory)/'svm_2d_simulation.csv',index=False)

def write_report(config,results=None,robust=None):
    out=Path(config['output_dir']);assets=Path(config['assets_dir'])
    assets.mkdir(parents=True,exist_ok=True)
    lines=['# 汇报素材索引','',f'运行配置：`{out / "run_config.json"}`。正式测试集用于最终评价；验证集按训练轨迹分组；最终模型选参后用训练＋验证重拟合。',
           '所有交通标志指标来自真实 GTSRB 运行；二维 SVM 图是明确标注的模拟教学材料。',
           '','|部分 / 要回答的问题|素材路径|实验条件、实际观察与结论边界|','|---|---|---|',
           '|任务与数据：类别、样本是否均衡？|01_任务与数据/class_samples.png；class_distribution_*.png；split_counts.csv|每类训练集文件名排序首图，不按清晰程度选择；分布图分别标明 train/val/test。类别不均衡，应结合宏平均指标。|',
           '|预处理：模型到底看到什么？|01_任务与数据/preprocessing.png|RGB→32×32→灰度[0,1]；默认不追加 ROI；只识别已裁剪标志，不做检测。图像缩放可能丢失小字细节。|',
           '|特征：HOG 如何表达边缘？|02_特征提取/hog_triplet_*.png；singles/|同组图片、统一参数，HOG 324 维；灰度 1024 维；HOG 显示做强度拉伸，图不是模型完整特征向量。|',
           '|模型：边界、间隔和支持向量？|03_分类模型/svm_2d_principle.png/.svg；svm_2d_simulation.csv|固定种子二维模拟数据；仅解释 SVM 原理，不代表真实 324 维交通标志空间。|',
           '|实验：模型和特征差异？|04_实验分析/metrics.csv；metrics_table.png；model_scores.png/.svg；model_times.png/.svg|同一划分和测试集；SVM 选同一组 C；验证集宏 F1 选参。拟合耗时只含最终拟合，预测只含特征到类别；特征和选参耗时另见 model_summary.json。单次 CPU 耗时受机器负载影响。|',
           '|实验：哪些类别容易混淆？|04_实验分析/*_confusion.png/.svg；confusion_local.png/.svg|完整矩阵为各模型真实计数；局部图按 HOG＋SVM 非对角错误最多的类别选择，只展示局部计数，不代表所有错误。原始矩阵见 outputs。|',
           '|案例：具体识别效果？|04_实验分析/correct_cases.png；error_cases.png；examples/；outputs 下 demo_cases.csv 与 demo_inputs/|从正确/错误样本中按类别顺序各取首个文件、最多12张；成功失败案例只用于解释，整体性能以全部测试指标为准。|',
           '|鲁棒性：亮度和模糊影响？|04_实验分析/*_robustness.png/.svg；*_same_image.png；perturbation_singles/|固定 HOG＋SVM；官方测试集每类固定抽样最多30张；各强度使用同一批图。亮度倍率1为基线、模糊0为基线，干扰先于缩放；模糊单位为原图像素，对不同尺寸影响不完全一致。未测试联合干扰或真实天气。|']
    if results is not None:
        columns=['model','accuracy','macro_precision','macro_recall','macro_f1','train_seconds','predict_seconds']
        lines+=['','## 实际测试观察','', '|方法|准确率|宏Precision|宏Recall|宏F1|拟合秒|预测秒|', '|---|---|---|---|---|---|---|']
        for _,r in results.iterrows():
            lines.append('|'+str(r['model'])+'|'+'|'.join(f'{r[k]:.4f}' for k in columns[1:])+'|')
        best=results.sort_values('macro_f1',ascending=False).iloc[0]
        lines += [f'本次三种方法中，{best.model} 的测试宏平均 F1 最高（{best.macro_f1:.4f}）。这是本次固定配置下的观察，不是对所有数据和参数的普遍结论。']
        hog=results.set_index('model').loc['hog_svm'];pix=results.set_index('model').loc['pixels_svm']
        delta=hog.macro_f1-pix.macro_f1
        lines += [f'HOG＋SVM 相比灰度像素＋同种 SVM 的宏 F1 差值为 {delta:+.4f}；保留全部结果，不能仅凭维数或少量案例解释因果。']
    if robust is not None:
        lines += ['','## 固定子集鲁棒性观察','']
        for kind,group in robust.groupby('interference',sort=False):
            first,last=group.iloc[0],group.iloc[-1]
            lines += [f'- {kind}：基线准确率 {first.accuracy:.4f}、宏 F1 {first.macro_f1:.4f}；最大设定干扰准确率 {last.accuracy:.4f}、宏 F1 {last.macro_f1:.4f}。中间强度见曲线与 CSV，不假定单调变化。']
    audit_path=out/'data_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        lines+=['','## 数据审计与隔离','',f'本次清单各划分数量：{audit.get("counts",{})}。详细审计见 `{audit_path}`。']
        manifest_path=out/'manifest_with_exclusions.csv'
        if manifest_path.exists():
            audit_manifest=pd.read_csv(manifest_path)
            excluded=audit_manifest[audit_manifest.split=='excluded']
            if len(excluded):
                lines += [f'隔离了 {len(excluded)} 张训练来源图片，轨迹为 {sorted(excluded.track.unique())}。原因是官方训练与测试出现完全相同图片；隔离完整匹配训练轨迹，保留官方测试集。此操作只基于内容哈希，不根据测试预测或指标选择样本。原始重复和隔离记录保留在运行目录。', '即便保留测试集，训练池已做去重隔离，故须把本次设置与直接使用全部官方训练图片的结果区分；已知完全重复为零不等于证明所有近似图片或物理身份关系独立。']
    lines+=['','## 追溯与复现','',f'- 运行配置、版本、源代码 SHA256：`{out}/run_config.json`、`{out}/environment.json`。',f'- 数据检查、分组和内容重叠审计：`{out}/data_audit.json`；逐图划分：`{out}/manifest.csv`。',f'- 完整选参过程：`{out}/validation_search.csv`；重载检查：`{out}/verification.json`。',f'- 固定干扰子集与逐图预测：`{out}/robustness_subset.csv`、`{out}/robustness_predictions.csv`。', '- 统计和曲线保留 SVG；图像拼图和单图为高清 PNG。所有结果仅覆盖 GTSRB 已裁剪德国交通标志；不支持道路目标检测，也不能外推真实车载安全性能。', '- 模型决策分数不是概率。没有使用增强，官方训练/测试物理轨迹独立性依数据发布定义；检查记录已知 ID 和内容哈希重叠。']
    size=config['image_size'];width,height=(size,size) if isinstance(size,int) else size
    gray=np.zeros((height,width),dtype=np.float32)
    hd=len(feature_vector(gray,config,'hog'));pdims=gray.size
    text='\n'.join(lines).replace('32×32',f'{width}×{height}').replace('HOG 324 维',f'HOG {hd} 维').replace('真实 324 维',f'真实 {hd} 维').replace('灰度 1024 维',f'灰度 {pdims} 维').replace('最多30张',f'最多{config["robust_per_class"]}张')
    if config['use_roi']:text=text.replace('默认不追加 ROI','使用标注 ROI')
    (assets/'汇报素材索引.md').write_text(text,encoding='utf-8')
    (out/'实验说明.md').write_text(text,encoding='utf-8')
