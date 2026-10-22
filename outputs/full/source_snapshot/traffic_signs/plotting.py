"""Consistent projection-friendly figures; numerical sources are saved separately."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

COLORS = ['#2463A6', '#E49A29', '#409983', '#B55775']
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 17, 'axes.labelsize': 14,
                     'axes.unicode_minus': False, 'svg.fonttype': 'none',
                     'axes.prop_cycle': matplotlib.cycler(color=COLORS), 'figure.facecolor': 'white'})

def save(fig, path, vector=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path)+'.png', dpi=220, bbox_inches='tight')
    if vector:
        fig.savefig(str(path)+'.svg', bbox_inches='tight')
    plt.close(fig)

def gallery(images, captions, path, title='', cols=4):
    if not images:
        return
    rows = int(np.ceil(len(images)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.0), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, im, caption in zip(axes.flat, images, captions):
        ax.imshow(im, cmap='gray' if np.asarray(im).ndim == 2 else None)
        ax.set_title(caption, fontsize=11)
    fig.suptitle(title, fontsize=17)
    save(fig, path, vector=False)

def heatmap(cm, labels, path, title, annotate=False):
    n=len(labels)
    fig,ax=plt.subplots(figsize=(max(9,n*.31),max(7,n*.28)))
    im=ax.imshow(cm,cmap='Blues')
    ax.set(xticks=range(n),yticks=range(n),xticklabels=labels,yticklabels=labels,
           xlabel='预测类别',ylabel='真实类别',title=title)
    ax.tick_params(axis='x',labelrotation=90,labelsize=9)
    ax.tick_params(axis='y',labelsize=9)
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=10,
                        color='white' if cm[i,j]>cm.max()*.55 else 'black')
    fig.colorbar(im,ax=ax,shrink=.8,label='样本数')
    save(fig,path)
