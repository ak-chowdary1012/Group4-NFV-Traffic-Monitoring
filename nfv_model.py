"""Fluid discrete-event reconstruction of Yang et al.'s monitoring/control design.
All capacities and delays are simulation inputs. No packets or live control.
"""
from dataclasses import dataclass, asdict
import heapq, math
import numpy as np

BYTES_PER_MBPS_SECOND = 125000.0

@dataclass(frozen=True)
class Config:
    duration: float = 600.0
    poll: float = 10.0
    capacity: float = 100.0
    telemetry_loss: float = 0.0
    telemetry_min: float = 0.05
    telemetry_max: float = 0.25
    processing: float = 0.02
    installation: float = 0.10
    block_ttl: float = 60.0
    failure: float | None = None
    backup: bool = True
    advert: float = 1.0
    priority: int = 150
    startup: float = 2.0
    policy: bool = True

# Each tuple is (source index, start, end, injected total offered Mbps).
INCIDENTS = [(1,100.0,120.0,25.0),(0,200.0,260.0,60.0),(2,280.0,283.0,80.0)]

def incident_schedule(seed):
    rng=np.random.default_rng(seed+30000)
    return [(src,a+int(shift),b+int(shift),r) for (src,a,b,r),shift in zip(INCIDENTS,rng.integers(0,60,3))]

def workload(seed, duration=600, scale=1.0):
    rng=np.random.default_rng(seed+20000)
    rates=rng.lognormal(np.log(5.5),.25,(int(duration),4))
    # Ordinary bursts remain legitimate, including the scheduled backup below.
    mask=rng.random(rates.shape)<.02
    rates=np.where(mask,np.minimum(8*(rng.pareto(1.5,rates.shape)+1),35),rates)
    labels=np.zeros_like(rates,dtype=bool)
    for src,a,b,r in incident_schedule(seed):
        rates[int(a):int(b),src]=r;labels[int(a):int(b),src]=True
    rates[420:440,3]=35.0
    return rates*scale,labels

def calibrated(seed):
    rng=np.random.default_rng(seed);n=10080
    body=rng.lognormal(np.log(5.5),.45,n);mask=rng.random(n)<.03
    tail=np.minimum(8*(rng.pareto(1.5,n)+1),45)
    tr=np.where(mask,tail,body)[:8640]
    z=(tr-tr.mean())/tr.std(ddof=0);clean=tr[np.abs(z)<=3]
    return {'Trimmed mean':float(clean.mean()),'Mean + 3 SD':float(clean.mean()+3*clean.std(ddof=0)),'Fixed 15':15.0}

class Model:
    def __init__(self,rates,labels,threshold,config=Config(),seed=0):
        self.rates=np.asarray(rates,dtype=float);self.labels=np.asarray(labels,dtype=bool)
        self.c=config;self.k=float(threshold);self.seed=seed;self.incidents=incident_schedule(seed)
        if self.rates.ndim!=2 or self.rates.shape!=self.labels.shape or np.any(self.rates<0):raise ValueError('Nonnegative 2D rates and matching labels required')
        if self.rates.shape[0]<math.ceil(config.duration):raise ValueError('Workload shorter than duration')
        if config.poll<=0 or config.capacity<=0 or not 0<=config.telemetry_loss<=1 or not 0<config.advert or not 1<=config.priority<=254:raise ValueError('Invalid polling/capacity/HA/loss parameters')
        if config.telemetry_min<0 or config.telemetry_max<config.telemetry_min or min(config.processing,config.installation,config.startup,config.block_ttl)<0:raise ValueError('Negative/reversed delay')
        self.n=self.rates.shape[1];self.q=[];self.serial=0;self.now=0.;self.current=self.rates[0].copy();self.mal=self.labels[0].copy()
        self.active=True;self.epoch=0;self.counter=np.zeros(self.n);self.prev=self.counter.copy();self.prev_t=0.
        self.window_mal=np.zeros(self.n);self.until=np.zeros(self.n);self.pending=np.zeros(self.n,dtype=bool)
        self.total=np.zeros(5) # offered, forwarded, policy drop, outage drop, congestion drop
        self.benign_offered=0.;self.benign_blocked=0.;self.mal_offered=0.;self.mal_forwarded=0.
        self.polls=[];self.commands=[];self.timeline=[];self.logs=[];self.recovery=None;self.sent=0;self.received=0;self.lost=0;self.epoch_skips=0;self.benign_windows=0;self.false_windows=0
        self.detected={};self.actioned={};self.err=[];self.instant_err=[]
        self.rng=np.random.default_rng(seed+50000+int(config.poll*1000))
    def schedule(self,t,kind,data=None,priority=2):
        if t<=self.c.duration+1e-10:
            self.serial+=1;heapq.heappush(self.q,(float(t),priority,self.serial,kind,data))
    def integrate(self,t):
        dt=t-self.now
        if dt<=0:return
        offered=self.current*dt*BYTES_PER_MBPS_SECOND
        benign=~self.mal;self.benign_offered+=offered[benign].sum();self.mal_offered+=offered[self.mal].sum()
        blocked=(self.until>self.now+1e-10) if self.c.policy else np.zeros(self.n,dtype=bool)
        if self.active:
            self.counter+=offered;self.window_mal+=dt*self.mal
            policy=np.where(blocked,offered,0);allowed=np.where(blocked,0,offered)
            ratio=min(1.,self.c.capacity/max(float(self.current[~blocked].sum()),1e-30))
            forwarded=allowed*ratio;congestion=allowed-forwarded;outage=np.zeros(self.n)
        else:policy=np.zeros(self.n);forwarded=np.zeros(self.n);congestion=np.zeros(self.n);outage=offered
        self.total+=np.array([offered.sum(),forwarded.sum(),policy.sum(),outage.sum(),congestion.sum()])
        self.benign_blocked+=policy[benign].sum();self.mal_forwarded+=forwarded[self.mal].sum()
        self.timeline.append((self.now,t,float(self.current.sum()),float(forwarded.sum()/dt/BYTES_PER_MBPS_SECOND),int(self.active)))
        self.now=t
    def run(self):
        for t in range(1,math.ceil(self.c.duration)):self.schedule(t,'rate',t,0)
        for j,t in enumerate(np.arange(self.c.poll,self.c.duration+1e-9,self.c.poll)):self.schedule(t,'poll',j,1)
        if self.c.failure is not None:self.schedule(self.c.failure,'fail',None,-1)
        self.schedule(self.c.duration,'end',None,10)
        while self.q:
            t,_,_,kind,data=heapq.heappop(self.q);self.integrate(t)
            if kind=='rate':self.current=self.rates[data].copy();self.mal=self.labels[data].copy()
            elif kind=='fail':
                self.active=False;self.logs.append({'time':t,'event':'master_failed'})
                if self.c.backup:
                    # Periodic advertisements start at 0. No explicit shutdown advert.
                    last=math.floor((t-1e-12)/self.c.advert)*self.c.advert
                    mdi=(3+(256-self.c.priority)/256)*self.c.advert
                    self.schedule(last+mdi,'timeout',None,-1)
            elif kind=='timeout':self.logs.append({'time':t,'event':'master_down_timeout'});self.schedule(t+self.c.startup,'recover',None,-1)
            elif kind=='recover':
                self.active=True;self.epoch+=1;self.counter=np.zeros(self.n);self.prev=self.counter.copy();self.prev_t=t;self.window_mal=np.zeros(self.n);self.recovery=t
                self.logs.append({'time':t,'event':'backup_active','epoch':self.epoch})
            elif kind=='poll':
                if not self.active:continue
                dt=t-self.prev_t
                if dt<=1e-10:continue
                delta=self.counter-self.prev;measured=delta/dt/BYTES_PER_MBPS_SECOND
                # Reference computed from the original offered trace over the same window.
                ref=np.zeros(self.n)
                a=self.prev_t
                while a<t-1e-10:
                    b=min(t,math.floor(a+1e-10)+1.);ref+=self.rates[min(int(a+1e-10),len(self.rates)-1)]*(b-a);a=b
                ref/=dt
                self.err.extend((measured-ref).tolist())
                self.polls.append({'time':t,'start':self.prev_t,'duration':dt,'epoch':self.epoch,'rate':measured.tolist()})
                payload={'time':t,'start':self.prev_t,'rate':measured.copy(),'mal':self.window_mal>1e-10,'epoch':self.epoch}
                self.prev=self.counter.copy();self.prev_t=t;self.window_mal=np.zeros(self.n);self.sent+=1
                # Both draws consumed at every poll, even when the message is lost.
                lose=self.rng.random()<self.c.telemetry_loss;delay=self.rng.uniform(self.c.telemetry_min,self.c.telemetry_max)
                if lose:self.lost+=1
                else:self.schedule(t+delay+self.c.processing,'receive',payload,2)
            elif kind=='receive':
                if data['epoch']!=self.epoch:self.epoch_skips+=1;continue
                self.received+=1;rates=data['rate'];flags=rates>self.k
                instant=self.rates[min(int(t),len(self.rates)-1)]
                self.instant_err.extend((rates-instant).tolist())
                benign=~data['mal'];self.benign_windows+=int(benign.sum());self.false_windows+=int((flags&benign).sum())
                for src in np.flatnonzero(flags):
                    for j,(s,a,b,r) in enumerate(self.incidents):
                        if data['mal'][src] and src==s and a<=t<b and data['time']>a and data['start']<b:self.detected.setdefault(j,t-a)
                    if self.c.policy and self.active and not self.pending[src] and self.until[src]<=t:
                        self.pending[src]=True;self.schedule(t+self.c.installation,'install',(int(src),data['epoch']),3)
            elif kind=='install':
                src,epoch=data;self.pending[src]=False
                if epoch!=self.epoch or not self.active:continue
                self.until[src]=t+self.c.block_ttl;self.schedule(self.until[src],'expire',(src,self.until[src]),0)
                self.commands.append({'time':t,'source':src,'until':self.until[src]})
                for j,(s,a,b,r) in enumerate(self.incidents):
                    if src==s and a<=t<b:self.actioned.setdefault(j,t-a)
            elif kind=='expire':
                src,expiry=data
                if abs(self.until[src]-expiry)<1e-9:self.until[src]=0.
        offered,forwarded,policy,outage,congestion=self.total
        assert abs(offered-forwarded-policy-outage-congestion)<max(1e-5,offered*1e-12)
        summary={'K':self.k,'poll_s':self.c.poll,'sent':self.sent,'received':self.received,'lost':self.lost,'stale_epoch_messages':self.epoch_skips,'benign_windows':self.benign_windows,'false_windows':self.false_windows,
        'window_FPR_pct':100*self.false_windows/max(self.benign_windows,1),'benign_policy_drop_pct':100*self.benign_blocked/max(self.benign_offered,1),
        'malicious_forwarded_pct':100*self.mal_forwarded/max(self.mal_offered,1),'incidents_detected':len(self.detected),'incident_detection_pct':100*len(self.detected)/3,
        'large_detection_s':self.detected.get(1),'large_action_s':self.actioned.get(1),'short_detected':int(2 in self.detected),
        'counter_RMSE_Mbps':float(np.sqrt(np.mean(np.square(self.err)))) if self.err else None,
        'current_rate_RMSE_Mbps':float(np.sqrt(np.mean(np.square(self.instant_err)))) if self.instant_err else None,
        'offered_MB':offered/1e6,'forwarded_MB':forwarded/1e6,'policy_drop_MB':policy/1e6,'outage_drop_MB':outage/1e6,'congestion_drop_MB':congestion/1e6,
        'mean_forwarded_Mbps':forwarded/BYTES_PER_MBPS_SECOND/self.c.duration,
        'outage_s':(self.recovery-self.c.failure if self.recovery is not None else self.c.duration-self.c.failure) if self.c.failure is not None else 0.,'command_count':len(self.commands)}
        return summary
