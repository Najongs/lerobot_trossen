import glob,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B='/home/trossen-ai/.cache/huggingface/lerobot/kiroaiseoul/'
OUT='/tmp/claude-1000/-home-trossen-ai-NAJY-lerobot-trossen/d7611f1b-c591-4fd4-831e-e3683337d644/scratchpad/t02_traj.png'
def load(n): return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(B+n+'/data/*/*.parquet'))])
runs=[('1013_1112 #32 (exec 100)','eval_najy_1013_1112_hot_t02_e100','#2a78d6'),('1013_1114 J14 (exec 100)','eval_najy_1013_1114_hot_t02_e100','#eb6834')]
tr=load('task02_pickup_tubes'); FPS=21
eps=[np.stack(g['observation.state'].values) for _,g in tr.groupby('episode_index')]
T=max(len(e) for e in eps)
pad=np.stack([np.vstack([e,np.repeat(e[-1:],T-len(e),0)]) for e in eps])
labels=['j0','j1','j2','j3','j4','j5','grip']
data={n:np.stack(load(d)['observation.state'].values) for n,d,_ in runs}
tmax=max(len(v) for v in data.values())/FPS
fig,ax=plt.subplots(7,2,figsize=(13,15),sharex=True,facecolor='#fcfcfb')
for c,(off,arm) in enumerate([(0,'Left arm'),(7,'Right arm')]):
    for r in range(7):
        a=ax[r,c]; j=off+r; k=1 if r==6 else 57.2958; t=np.arange(T)/FPS
        a.set_facecolor('#fcfcfb')
        a.fill_between(t,np.percentile(pad[:,:,j],10,0)*k,np.percentile(pad[:,:,j],90,0)*k,color='#c3c2b7',alpha=.45,lw=0,label='train p10–p90')
        a.plot(t,np.median(pad[:,:,j],0)*k,color='#8a8984',lw=1.2,ls='--',label='train median')
        for n,d,col in runs:
            s=data[n]; a.plot(np.arange(len(s))/FPS,s[:,j]*k,color=col,lw=2,label=n)
        for x in np.arange(100,int(tmax*FPS),100)/FPS: a.axvline(x,color='#d9d8d4',lw=.8,zorder=0)
        a.set_ylabel(labels[r]+(' [m]' if r==6 else ' [deg]'),color='#52514e',fontsize=9)
        a.tick_params(colors='#52514e',labelsize=8)
        for sp in ('top','right'): a.spines[sp].set_visible(False)
        a.grid(axis='y',color='#ebeae6',lw=.6)
        if r==0: a.set_title(arm,color='#0b0b0b',fontsize=12)
    ax[6,c].set_xlabel('time [s]  (training at 21 Hz; vertical lines = chunk swap every 100 ticks)',color='#52514e',fontsize=8)
ax[0,0].legend(loc='upper right',fontsize=8,frameon=False)
fig.suptitle('task02 pick up tubes — measured joint state, exec 100 runs vs training demos',color='#0b0b0b',fontsize=13)
plt.xlim(0,min(tmax,40)); fig.tight_layout(rect=[0,0,1,.98]); fig.savefig(OUT,dpi=90); print(OUT)
for n,d,_ in runs:
    s=data[n]; print(n,'len %.1fs'%(len(s)/FPS))
    for side,off in (('L',0),('R',7)):
        print('  ',side,'max-reach deg (j1,j2,j3):',[round(float(np.max(np.abs(s[:,off+i]-s[0,off+i]))*57.3)) for i in (1,2,3)],'| grip min/max %.3f/%.3f'%(s[:,off+6].min(),s[:,off+6].max()))
print('train median max-reach L (j1,j2,j3):',[round(float(np.median([np.max(np.abs(e[:,i]-e[0,i])) for e in eps]))*57.3) for i in (1,2,3)],' R:',[round(float(np.median([np.max(np.abs(e[:,7+i]-e[0,7+i])) for e in eps]))*57.3) for i in (1,2,3)])
print('train grip L median min/max %.3f/%.3f  R %.3f/%.3f'%(np.median([e[:,6].min() for e in eps]),np.median([e[:,6].max() for e in eps]),np.median([e[:,13].min() for e in eps]),np.median([e[:,13].max() for e in eps])))
