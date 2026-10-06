import pandas as pd, numpy as np, glob, os, sys
H=os.path.expanduser("~/.cache/huggingface/lerobot/kiroaiseoul/")
J=[0,1,2,3,4,5,7,8,9,10,11,12]
def load(name):
    fs=sorted(glob.glob(H+name+"/data/*/*.parquet"))
    d=pd.concat([pd.read_parquet(f,columns=["episode_index","frame_index","observation.state"]) for f in fs])
    s=np.stack(d["observation.state"].values)[:,J]
    return s, d["episode_index"].values, d["frame_index"].values
T4,e4,f4=load("task04_pour_liquid_from_tubes_to_beaker"); T5,e5,f5=load("task05_tube_disposal")
print("train frames task04",len(T4),"task05",len(T5))
def nn(X,T):
    out=np.empty(len(X)); 
    for i in range(0,len(X),200):
        dd=((X[i:i+200,None,:]-T[None,::3,:])**2).sum(-1); out[i:i+200]=np.sqrt(dd.min(1))
    return out
for run in sys.argv[1:]:
    e=pd.read_parquet(H+run+"/data/chunk-000/file-000.parquet")
    X=np.stack(e["observation.state"].values)[:,J]
    d4=nn(X,T4); d5=nn(X,T5); n=len(X); fps=21
    print(f"\n{run}: {n} frames | 전체 task04 쪽이 더 가까운 프레임 {np.mean(d4<d5)*100:.0f}%  | 중앙 거리 t04 {np.median(d4):.2f} t05 {np.median(d5):.2f}")
    for a in range(0,n,fps*5):
        b=min(n,a+fps*5); s=X[a:b]
        mv=np.abs(np.diff(X[a:b+1],axis=0)).sum(1).mean() if b-a>1 else 0
        print(f"  {a/fps:4.0f}-{b/fps:3.0f}s  이동 {mv:.3f}  t04근접 {np.median(d4[a:b]):.2f}  t05근접 {np.median(d5[a:b]):.2f}  → {'t04' if np.median(d4[a:b])<np.median(d5[a:b]) else 't05'} | L j0,j1,j4 {np.degrees(X[a,[0,1,4]]).round().astype(int)} R j0,j1,j4 {np.degrees(X[a,[6,7,10]]).round().astype(int)}")
