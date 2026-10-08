"""t05 실기 프레임: 4라운드가 원핫 5/11 과 4/11 에 다른 동작을 내나(원핫 vs 비전), 궤적이 t04·t05 시연 중 어디에 가까운가. 로봇 무접촉."""
import glob, os, numpy as np, torch, pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda"); ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
C="/home/trossen-ai/.cache/huggingface/lerobot/kiroaiseoul/"
def snap(m): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--kiroaiseoul--{m}/snapshots/*/"))[0]
R="1008_1135_chain_r4_t05_e30"
# 1) 실기 궤적이 t04·t05 시연 중 어디에 가까운가 (10/06 stage_nearness 와 같은 뜻)
def states(repo):
    fs=sorted(glob.glob(C+repo+'/data/*/*.parquet')); d=pd.concat([pd.read_parquet(p, columns=['frame_index','observation.state']) for p in fs]).sort_values('frame_index'); return np.stack(d['observation.state'].to_numpy())[:,ARM]
T4=states('task04_pour_liquid_from_tubes_to_beaker'); T5=states('task05_tube_disposal'); E=states('eval_'+R)[52:]
idx=np.arange(0,len(E),21)
d4=np.array([np.sqrt(((T4-E[i])**2).sum(1)).min() for i in idx]); d5=np.array([np.sqrt(((T5-E[i])**2).sum(1)).min() for i in idx])
print(f"실기 {R} 정책 {len(E)}프레임: 초별 최근접 — task04 시연 중앙 {np.median(d4):.2f} / task05 시연 중앙 {np.median(d5):.2f} | t05 가 더 가까운 초 {int((d5<d4).sum())}/{len(idx)}")
print("  초별 (t04/t05):", " ".join(f"{a:.2f}/{b:.2f}" for a,b in zip(d4[:20],d5[:20])))
# 2) 원핫 교체: 실기 첫 프레임·시연 t05 첫 프레임에 5/11 vs 4/11 → 청크 차이, 그리고 그리퍼 열림(버리기 = 그리퍼 염)
ds_e = LeRobotDataset('kiroaiseoul/eval_'+R, episodes=[0]); ds5 = LeRobotDataset('kiroaiseoul/task05_tube_disposal', episodes=[0,20,40]); 
epi=np.array([int(x) for x in ds5.hf_dataset['episode_index']]); firsts={e:int(np.where(epi==e)[0][0]) for e in (0,20,40)}
items=[("실기 @0s", ds_e[52]), ("실기 @4s", ds_e[52+84]), ("실기 @8s", ds_e[52+168])]+[(f"시연 t05 ep{e} @0s", ds5[firsts[e]]) for e in (0,20,40)]+[(f"시연 t05 ep0 @2s", ds5[firsts[0]+60])]
for name,m,env in [("4라운드 60K","NAJY_act_all11_r4_27D_60k_s1000",True),("M1","NAJY_act_all11_hot_27D_120k_s1000",False)]:
    ck=snap(m); pol=ACTPolicy.from_pretrained(ck).to(dev).eval(); pre,post=make_pre_post_processors(pol.config, pretrained_path=ck); cams=[c for c in pol.config.input_features if c.startswith("observation.images")]
    print(f"\n== {name}: 청크(30) — 원핫별 [팔 이동 rad · 그리퍼 명령 최대 L/R · p] 와 5/11 vs 4/11 청크 차이(L2 평균)")
    for label,it in items:
        outs={}
        for h in (4,3,None):
            hot=np.zeros(11,np.float32)
            if h is not None: hot[h]=1
            obs={"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2,np.float32), hot])}
            if env: obs["observation.environment_state"]=hot.copy()
            for c in cams: obs[c]=(it[c].permute(1,2,0).numpy()*255).round().astype(np.uint8)
            with torch.inference_mode(): a=post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs,dev,None,None))))[0].float().cpu().numpy()
            outs[h]=a
        def desc(a): return f"이동 {np.linalg.norm(a[-1,ARM]-a[0,ARM]):.2f} g {a[:,6].max():.3f}/{a[:,13].max():.3f}" + (f" p{a[-1,16]:.2f}" if a.shape[1]>16 else "")
        diff=np.linalg.norm(outs[4][:,ARM]-outs[3][:,ARM],axis=1).mean()
        print(f"   {label:<18}: 5/11 [{desc(outs[4])}] | 4/11 [{desc(outs[3])}] | 0벡터 [{desc(outs[None])}] | 5↔4 차이 {diff:.3f}")
    del pol; torch.cuda.empty_cache()
