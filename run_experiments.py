"""Rebuild all system-simulation evidence. Run: python run_experiments.py"""
from pathlib import Path
from dataclasses import replace,asdict
import json,csv,platform,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nfv_model import Model,Config,workload,calibrated,incident_schedule
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
SEEDS=list(range(30));POLLS=[1,10,30,60];RULES=['Trimmed mean','Mean + 3 SD','Fixed 15']
def save_csv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def describe(rows,metrics):
 d={}
 for k in metrics:
  a=np.array([r[k] for r in rows if r[k] is not None],dtype=float)
  d[k]={'mean':float(a.mean()) if len(a) else None,'sd':float(a.std(ddof=1)) if len(a)>1 else 0.,'n':len(a)}
 return d
rows=[];loss=[];ha=[];capacity=[];allmodels=[]
for seed in SEEDS:
 x,l=workload(seed);phase=np.random.default_rng(seed+90000).uniform(0,1);failure=350+phase
 ks=calibrated(seed)
 for poll in POLLS:
  for rule in RULES:
   cfg=Config(poll=poll,failure=failure);m=Model(x,l,ks[rule],cfg,seed);r=m.run();rows.append({'seed':seed,'rule':rule,**r})
  for prob in [0,.1,.3]:
   r=Model(x,l,15,Config(poll=poll,failure=failure,telemetry_loss=prob),seed).run();loss.append({'seed':seed,'loss_probability':prob,**r})
 for adv in [.5,1,2]:
  r=Model(x,l,15,Config(policy=False,failure=350+phase*adv,advert=adv),seed).run();ha.append({'seed':seed,'advert_s':adv,'backup':True,**r})
 r=Model(x,l,15,Config(policy=False,failure=failure,backup=False),seed).run();ha.append({'seed':seed,'advert_s':1,'backup':False,**r})
 for scale in [1,2,4]:
  for cap in [50,100,200]:
   xx,ll=workload(seed,scale=scale);r=Model(xx,ll,15,Config(policy=False,capacity=cap,failure=None),seed).run();capacity.append({'seed':seed,'load_scale':scale,'capacity_Mbps':cap,**r})
metrics=['K','sent','received','window_FPR_pct','benign_policy_drop_pct','malicious_forwarded_pct','incident_detection_pct','large_detection_s','large_action_s','short_detected','counter_RMSE_Mbps','current_rate_RMSE_Mbps','outage_s','command_count','forwarded_MB']
summary=[{'rule':rule,'poll_s':poll,**describe([r for r in rows if r['rule']==rule and r['poll_s']==poll],metrics)} for poll in POLLS for rule in RULES]
loss_sum=[{'poll_s':p,'loss_probability':v,**describe([r for r in loss if r['poll_s']==p and r['loss_probability']==v],metrics)} for p in POLLS for v in [0,.1,.3]]
ha_sum=[{'advert_s':a,'backup':b,**describe([r for r in ha if r['advert_s']==a and r['backup']==b],['outage_s','outage_drop_MB','mean_forwarded_Mbps'])} for a,b in [(.5,True),(1,True),(2,True),(1,False)]]
cap_sum=[{'load_scale':s,'capacity_Mbps':c,**describe([r for r in capacity if r['load_scale']==s and r['capacity_Mbps']==c],['mean_forwarded_Mbps','offered_MB','congestion_drop_MB'])} for s in [1,2,4] for c in [50,100,200]]
example_x,example_l=workload(7);example=Model(example_x,example_l,15,Config(poll=10,failure=350+np.random.default_rng(90007).uniform(0,1)),7);example_r=example.run()
for name,data in [('system_runs.csv',rows),('telemetry_loss_runs.csv',loss),('ha_runs.csv',ha),('capacity_runs.csv',capacity)]:save_csv(name,data)
(OUT/'example_trace.json').write_text(json.dumps({'polls':example.polls,'commands':example.commands,'incidents':incident_schedule(7),'events':example.logs,'summary':example_r},indent=2))
result={'seeds':SEEDS,'polls':POLLS,'rules':RULES,'config':asdict(Config()),'summary':summary,'loss_summary':loss_sum,'ha_summary':ha_sum,'capacity_summary':cap_sum,'example':example_r,'run_counts':{'main':len(rows),'loss':len(loss),'HA':len(ha),'capacity':len(capacity)},'environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},'source_sha256':hashlib.sha256((ROOT/'nfv_model.py').read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
navy='#253261';blue='#237FA1';green='#4A836E'
# Integrate event segments into one-second plotting bins, preserving total bytes.
bins=np.zeros((600,2));up=np.ones(600)
for a,b,o,f,state in example.timeline:
 while a<b-1e-10:
  i=min(int(a+1e-10),599);e=min(b,i+1.);bins[i]+=np.array([o,f])*(e-a);up[i]=min(up[i],state);a=e
fig,axs=plt.subplots(2,1,figsize=(9,4.7),layout='constrained',gridspec_kw={'height_ratios':[3,1]})
axs[0].plot(np.arange(600),bins[:,0],color='#9EADB9',lw=1,label='Offered');axs[0].plot(np.arange(600),bins[:,1],color=blue,lw=1.2,label='Forwarded')
for _,a,b,_ in incident_schedule(7):axs[0].axvspan(a,b,color='#F1D9B7',alpha=.65)
axs[0].set(ylabel='Aggregate rate (Mbps)',xlim=(0,600));axs[0].legend(frameon=False,ncol=2)
failure=example.c.failure;recovery=example.recovery
axs[1].step([340,failure,recovery,365],[1,0,1,1],where='post',color=navy);axs[1].set(ylim=(-.1,1.2),yticks=[0,1],yticklabels=['Down','Up'],xlabel='Simulated time (s)',xlim=(340,365))
fig.savefig(OUT/'system_trace.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for rule,col in zip(RULES,[navy,blue,green]):
 rr=[r for r in summary if r['rule']==rule]
 axs[0].errorbar(POLLS,[r['large_detection_s']['mean'] for r in rr],yerr=[r['large_detection_s']['sd'] for r in rr],marker='o',capsize=3,label=rule,color=col)
 axs[1].plot(POLLS,[r['benign_policy_drop_pct']['mean'] for r in rr],marker='o',label=rule,color=col)
axs[0].set(xlabel='Polling interval (s)',ylabel='Large-event detection delay (s)');axs[1].set(xlabel='Polling interval (s)',ylabel='Benign bytes blocked by policy (%)');axs[1].legend(frameon=False,fontsize=9)
fig.savefig(OUT/'control_tradeoff.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for lossprob,col in zip([0,.1,.3],[navy,blue,green]):
 rr=[r for r in loss_sum if r['loss_probability']==lossprob]
 axs[0].plot(POLLS,[r['large_detection_s']['mean'] for r in rr],marker='o',label=f'{int(lossprob*100)}% message loss',color=col)
 axs[1].plot(POLLS,[r['current_rate_RMSE_Mbps']['mean'] for r in rr],marker='o',color=col)
axs[0].set(xlabel='Polling interval (s)',ylabel='Large-event detection delay (s)');axs[0].legend(frameon=False,fontsize=9);axs[1].set(xlabel='Polling interval (s)',ylabel='Current-rate RMSE (Mbps)')
fig.savefig(OUT/'monitoring_delay.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
axs[0].bar(['0.5 s','1 s','2 s'],[r['outage_s']['mean'] for r in ha_sum[:3]],yerr=[r['outage_s']['sd'] for r in ha_sum[:3]],capsize=4,color=blue);axs[0].set(xlabel='Advertisement interval',ylabel='Simulated outage (s)')
for c,col in zip([50,100,200],[navy,blue,green]):
 rr=[r for r in cap_sum if r['capacity_Mbps']==c];axs[1].plot([1,2,4],[r['mean_forwarded_Mbps']['mean'] for r in rr],marker='o',label=f'{c} Mbps capacity',color=col)
axs[1].set(xlabel='Offered-load multiplier',ylabel='Mean forwarded rate (Mbps)');axs[1].legend(frameon=False,fontsize=9)
fig.savefig(OUT/'availability_capacity.png',dpi=200);plt.close(fig)
print(json.dumps({'run_counts':result['run_counts'],'summary':summary,'ha':ha_sum,'capacity':cap_sum},indent=2))
