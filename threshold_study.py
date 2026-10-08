"""Offline numerical evaluation adapted from the supplied k_threshold_sim.py.
Synthetic scalar rates only; no network deployment or traffic generation.
"""
import json, csv, platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).parent/'results'/'thresholds';OUT.mkdir(parents=True,exist_ok=True)
NAMES=['Trimmed mean','Printed algorithm','Trimmed mean + 3 SD','Median + 3 scaled MAD','99.9th percentile']
SHORT=['Mean','Printed','Mean + 3 SD','Median + 3 MAD','P99.9']
def generate(seed, cap=True):
    rng=np.random.default_rng(seed); n=10080
    body=rng.lognormal(np.log(5.5),.45,n)
    mask=rng.random(n)<.03
    tail=8*(rng.pareto(1.5,n)+1)
    if cap: tail=np.minimum(tail,45)
    x=np.where(mask,tail,body)
    strong=rng.uniform(50,200,200)
    medium=rng.uniform(20,40,200)
    subtle=rng.uniform(8,20,200)
    return x[:8640],x[8640:],strong,medium,subtle
def thresholds(v):
    z=(v-v.mean())/v.std(ddof=0); c=v[np.abs(z)<=3]; o=v[np.abs(z)>3]
    med=np.median(v); mad=1.4826*np.median(np.abs(v-med))
    return [c.mean(),o.mean() if len(o) else np.nan,c.mean()+3*c.std(ddof=0),med+3*mad,np.percentile(v,99.9)],len(o)
def run(seed, cap=True):
    tr,te,a,b,c=generate(seed,cap); ks,kept=thresholds(tr)
    rows=[]
    for name,k in zip(NAMES,ks):
        rows.append(dict(rule=name,K=float(k),FPR=float(100*np.mean(te>k)),strong=float(100*np.mean(a>k)),medium=float(100*np.mean(b>k)),subtle=float(100*np.mean(c>k)),drift=float(100*np.mean(te*1.5>k)),FP=int(np.sum(te>k)),TN=int(np.sum(te<=k))))
    return rows,dict(n=len(tr),mean=float(tr.mean()),median=float(np.median(tr)),sd=float(tr.std()),skew=float(np.mean((tr-tr.mean())**3)/tr.std()**3),kept=kept)
single,stats=run(42)
allrows=[run(s)[0] for s in range(20)]
uncapped=[run(s,False)[0] for s in range(20)]
def summary(runs):
    return [dict(rule=n,**{m:dict(mean=float(np.mean([r[i][m] for r in runs])),sd=float(np.std([r[i][m] for r in runs],ddof=1))) for m in ['K','FPR','strong','medium','subtle','drift']}) for i,n in enumerate(NAMES)]
results=dict(single=single,stats=stats,multi=summary(allrows),uncapped=summary(uncapped),environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),seeds=list(range(20)))
(OUT/'results.json').write_text(json.dumps(results,indent=2))
with (OUT/'experiment_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['seed','cap']+list(single[0]));w.writeheader()
    for cap in [True,False]:
        for s in range(20):
            for r in run(s,cap)[0]:w.writerow(dict(seed=s,cap=cap,**r))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
colors=['#253261','#8c97a8','#237fa1','#4a836e','#c48a31']
tr,te,*_=generate(42)
fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
ax.hist(tr,bins=np.arange(0,47,1),color='#d7e3ef',edgecolor='white');ax.set(xlabel='Synthetic rate (Mbps)',ylabel='Training samples')
for i in [0,2,3,4]:ax.axvline(single[i]['K'],color=colors[i],label=f"{SHORT[i]}: {single[i]['K']:.2f}",lw=1.8)
ax.legend(frameon=False,fontsize=9);fig.savefig(OUT/'distribution.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
sm=results['multi'];y=np.arange(5)
axs[0].barh(y,[r['FPR']['mean'] for r in sm],xerr=[r['FPR']['sd'] for r in sm],color=colors,capsize=3)
axs[0].set(yticks=y,yticklabels=SHORT,xlabel='Benign false-positive rate (%)',title='Stationary benign day');axs[0].invert_yaxis()
for i,r in enumerate(sm):axs[0].text(r['FPR']['mean']+r['FPR']['sd']+.7,i,f"{r['FPR']['mean']:.1f}%",va='center',fontsize=9)
axs[0].set_xlim(0,55)
for i,key in enumerate(['strong','medium','subtle']):
    axs[1].plot(y,[r[key]['mean'] for r in sm],marker=['o','s','^'][i],label=['50–200 Mbps','20–40 Mbps','8–20 Mbps'][i])
axs[1].set(xticks=y,xticklabels=['Mean','Printed','Mean\n+ 3 SD','Median\n+ 3 MAD','P99.9'],ylabel='Synthetic anomaly detection rate (%)',ylim=(-3,105),title='Sensitivity to anomaly magnitude');axs[1].legend(frameon=False,fontsize=9)
fig.savefig(OUT/'sensitivity.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained');w=.35
ax.bar(y-w/2,[r['FPR']['mean'] for r in sm],w,label='Stationary test day',color='#253261')
ax.bar(y+w/2,[r['drift']['mean'] for r in sm],w,label='Benign test day × 1.5',color='#237fa1')
ax.set(xticks=y,xticklabels=SHORT,ylabel='False-positive rate (%)',ylim=(0,85));ax.legend(frameon=False)
fig.savefig(OUT/'drift.png',dpi=200);plt.close(fig)
print(json.dumps(results,indent=2))
