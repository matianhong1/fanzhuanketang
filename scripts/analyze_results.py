"""Evidence-based discussion; hypotheses remain explicitly untested."""
import argparse
import json
from pathlib import Path
import sys
import os
import re
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mplconfig'))
from traffic_signs.labels import CLASS_NAMES

def analyze(output):
    output=Path(output)
    config=json.loads((output/'run_config.json').read_text(encoding='utf-8'))
    scores=pd.read_csv(output/'metrics.csv').set_index('model')
    predictions=pd.read_csv(output/'hog_svm_test_predictions.csv')
    summary=json.loads((output/'model_summary.json').read_text(encoding='utf-8'))
    audit=json.loads((output/'data_audit.json').read_text(encoding='utf-8'))
    robust=pd.read_csv(output/'robustness.csv')
    hog=scores.loc['hog_svm'];pixels=scores.loc['pixels_svm'];knn=scores.loc['hog_knn']
    wrong=predictions[predictions.label!=predictions.prediction]
    lines=['# 实验结果分析','',f'结果来源：`{output}/metrics.csv`、逐图预测、选参表、数据审计与干扰 CSV。未根据测试结果重选参数。',
           '',f'HOG＋SVM 测试准确率 {hog.accuracy:.2%}、宏 F1 {hog.macro_f1:.4f}；HOG＋KNN 为 {knn.accuracy:.2%}、{knn.macro_f1:.4f}；灰度像素＋SVM 为 {pixels.accuracy:.2%}、{pixels.macro_f1:.4f}。',
           f'HOG＋SVM 相比灰度像素＋同种 SVM 的宏 F1 差值为 {hog.macro_f1-pixels.macro_f1:+.4f}。本次不能声称 HOG 一定优于原始像素。',
           '', '## 用错误分布分析结果','',f'HOG＋SVM 共错分 {len(wrong)} / {len(predictions)} 张。']
    speed=predictions[predictions.label.isin(range(9))]
    if len(speed):
        speedwrong=speed[speed.label!=speed.prediction]
        internal=speedwrong.prediction.isin(range(9)).sum()
        lines += [f'类别 ID 0–8（限速及解除80限速）共有 {len(speed)} 张，准确率 {(speed.label==speed.prediction).mean():.2%}；其中 {len(speedwrong)} 张错分，{internal} 张仍被判为该组其他类别。该组本身的样本量较大，应同时看条件准确率和错误计数，不能仅凭错误数判断类别难度。']
    pairs=wrong.groupby(['label','prediction']).size().sort_values(ascending=False).head(5)
    if (hog.accuracy-pixels.accuracy)*(hog.macro_f1-pixels.macro_f1)<0:
        lines += ['', '本次准确率与宏 F1 的排序不同，不能用单一指标宣称某种特征全面获胜。只有一个固定分组划分和随机种子，未做重复划分或显著性检验；微小差距仅作为本次观察。']
    lines+=['','主要混淆方向（计数）：','']
    for (truth,pred),n in pairs.items():
        lines += [f'- {truth} {CLASS_NAMES[truth]} → {pred} {CLASS_NAMES[pred]}：{n} 张。']
    per_class=[]
    for label in config['classes']:
        support=int(predictions.label.eq(label).sum());predicted=int(predictions.prediction.eq(label).sum())
        tp=int((predictions.label.eq(label)&predictions.prediction.eq(label)).sum())
        per_class.append({'class_id':label,'name_zh':CLASS_NAMES[label],'support':support,'f1':2*tp/(support+predicted) if support+predicted else 0.})
    weak=pd.DataFrame(per_class).sort_values(['f1','class_id'])
    weak.to_csv(output/'hog_svm_weak_classes.csv',index=False,encoding='utf-8-sig')
    lines+=['','逐类 F1 最低的五类（按完整测试结果排序，不用案例推断比例）：','']
    for r in weak.head(5).itertuples():
        lines += [f'- {r.class_id} {r.name_zh}：测试 {r.support} 张，F1 {r.f1:.4f}。']
    lines += ['', '宏平均 F1 对各类别赋同等权重。错误计数最多与逐类 F1 最低的类别未必相同，不能只用数字细节解释整体表现；类别不均衡与较少的训练轨迹也可能影响小类别，仍需单独验证。']
    size=config['image_size'];w,h=(size,size) if isinstance(size,int) else size
    cell='×'.join(map(str,config['hog']['pixels_per_cell']));dim=summary['hog_svm']['feature_dimension']
    lines += ['', f'解释假设（未做因果验证）：{w}×{h} 输入与{cell} cell 的 HOG 为{dim}维，局部梯度直方图可能丢失区分限速数字所需的细节；灰度像素保留位置与亮度，而该数据中的裁剪标志可能有较稳定的对齐。因此，特征压缩并不必然提高分类性能。以上是假设；当前结果同时受尺寸、HOG参数、线性分类器、灰度转换和分组划分影响，不能把差异归因于某一个因素。',
              '本次各方法只在同一验证集的小候选参数范围内选择，并未充分优化所有算法。若继续研究，应事先规定 HOG cell/输入尺寸等验证实验和独立评价协议，不能在当前测试集上反复挑选最好结果。',
              '', '## 验证结果与泛化','']
    for name,row in scores.iterrows():
        lines += [f'- {name}：选中参数 {summary[name]["selected_params"]}，验证宏 F1 {summary[name]["validation_macro_f1"]:.4f}，测试宏 F1 {row.macro_f1:.4f}。验证时拟合 train，测试时重新拟合 train＋val，两者训练数量不同，差值不能直接当作同一模型的过拟合量。']
    lines += ['', '## 鲁棒性与耗时','']
    for kind,g in robust.groupby('interference',sort=False):
        first,last=g.iloc[0],g.iloc[-1]
        lines += [f'- {kind}，固定 {int(first.n)} 张同一子集：基线准确率 {first.accuracy:.2%}、宏 F1 {first.macro_f1:.4f}；最强设定干扰准确率 {last.accuracy:.2%}、宏 F1 {last.macro_f1:.4f}。曲线中的细小非单调变化应保留。']
    lines += ['- 上述基线来自每类均衡抽样的固定子集，不能与完整不均衡测试集的准确率直接相减来量化干扰损失。模糊发生在原始分辨率，不同图片的相对模糊程度不同；未覆盖联合干扰与真实天气。',
              '- 拟合与预测耗时不含读取和特征计算；KNN 的快速拟合主要保存样本，分类计算发生在预测阶段，不能仅看拟合时间判断总成本。特征准备/缓存查询和选参耗时另见 model_summary.json；HOG 两模型共享提取时间，不重复相加。',
              '', '## 数据可信度与边界','',f'保留划分数量：{audit["counts"]}。训练/验证轨迹重叠 {audit["train_val_track_overlap"]}，保留数据跨集合完全相同文件组 {audit["cross_split_identical_file_groups"]}，相同 RGB 组 {audit["cross_split_identical_rgb_groups"]}。']
    if audit.get('excluded_rows',0):
        lines += [f'官方数据最初存在 {audit["cross_split_identical_file_groups_before_decontamination"]} 组跨集合完全重复；隔离训练轨迹 {audit["excluded_tracks"]} 共 {audit["excluded_rows"]} 张，官方测试集不变。本次属于隔离后协议，训练池并非完整39,209张，不能隐去此差异直接与标准榜单比较。']
    lines += ['没有同源增强；哈希检查只能确认完全相同图片，不能证明所有近似图片或物理标志实例独立。案例按固定规则选取，只解释单个成功和失败，不能代替总体指标。任务仅覆盖已裁剪德国交通标志，未评估目标检测、国内类别或实际车载场景。']
    text='\n'.join(lines)+'\n'
    (output/'结果分析.md').write_text(text,encoding='utf-8')
    assets=Path(config['assets_dir'])
    (assets/'结果分析.md').write_text(text,encoding='utf-8')
    from traffic_signs.plotting import gallery
    from PIL import Image
    images=[];captions=[];selected=[]
    for r in weak.itertuples():
        hits=wrong[wrong.label.eq(r.class_id)].sort_values('path')
        if hits.empty:continue
        case=hits.iloc[0];image=Image.open(case.path).convert('RGB')
        caption=f'真实：{r.class_id} {r.name_zh}\n预测：{int(case.prediction)} {CLASS_NAMES[int(case.prediction)]}\n该类测试 n={r.support}；F1={r.f1:.3f}'
        images.append(image.copy());captions.append(caption)
        gallery([image],[caption],assets/'04_实验分析'/'weak_class_singles'/f'class_{r.class_id:02d}',cols=1)
        selected.append({'path':case.path,'true_id':r.class_id,'predicted_id':int(case.prediction),'class_support':r.support,'class_f1':r.f1,'selection':'事后按完整测试逐类F1升序，每类取文件名排序首个错例；仅作解释，不参与调参'})
        if len(selected)==6:break
    gallery(images,captions,assets/'04_实验分析'/'error_cases_weak_classes','逐类 F1 最低类别的错误案例（事后解释，不替代完整指标）',cols=3)
    pd.DataFrame(selected).to_csv(output/'weak_class_cases.csv',index=False,encoding='utf-8-sig')
    index=assets/'汇报素材索引.md'
    content=index.read_text(encoding='utf-8')
    content=re.sub(r'(?m)^(本次三种方法中，)',r'\n\1',content)
    marker='\n## 进一步结果分析\n'
    content=content.split(marker)[0]+marker+'\n详见同目录 `结果分析.md`：主要混淆方向、特征表现的解释假设、选参范围、泛化与鲁棒性限制。\n\n`04_实验分析/error_cases_weak_classes.png` 与 `weak_class_singles/`：按完整测试逐类F1升序选最多6类，每类选文件名首个错误样本；事后解释用途，不参与模型选择，不替代全部指标。逐项路径和条件见运行目录 `weak_class_cases.csv`。\n'
    index.write_text(content,encoding='utf-8')
    print(f'Written {output}/结果分析.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/full');args=p.parse_args();analyze(args.output)
