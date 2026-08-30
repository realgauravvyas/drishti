"""Full correctness validation: confusion matrix, calibration, ranking."""
import json,os,sys
import numpy as np
from collections import Counter
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import config as C
from metrics import auc
from fusion import FusionEngine
from simulate import EPICENTRE,MAGNITUDE,SCENARIO_BBOX
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
S=C.DAMAGE_STATES

def main():
    reports=json.load(open(os.path.join(D,"reports.json"),encoding="utf-8"))
    truth=json.load(open(os.path.join(D,"ground_truth.json"),encoding="utf-8"))
    eng=FusionEngine(epicentre=EPICENTRE,magnitude=MAGNITUDE,aoi=SCENARIO_BBOX)
    eng.prepare(reports); out,_=eng.fuse(24.0)
    sids=[s for s in out if s in truth]

    print("="*72); print("CONFUSION MATRIX  (rows = truth, cols = DRISHTI label)"); print("="*72)
    M=np.zeros((4,4),int)
    for s in sids: M[S.index(truth[s]["state"]),S.index(out[s]["state"])]+=1
    print("%-14s"%"" + "".join("%14s"%c[:12] for c in S)+"%10s"%"recall")
    for i,r in enumerate(S):
        rec=M[i,i]/max(1,M[i].sum())
        print("%-14s"%r[:13] + "".join("%14d"%M[i,j] for j in range(4))+"%9.0f%%"%(100*rec))
    print("%-14s"%"precision" + "".join("%13.0f%%"%(100*M[j,j]/max(1,M[:,j].sum())) for j in range(4)))
    acc=np.trace(M)/M.sum(); adj=sum(M[i,j] for i in range(4) for j in range(4) if abs(i-j)<=1)/M.sum()
    print("\nexact accuracy %.1f%%   within-one-band %.1f%%"%(100*acc,100*adj))

    # severe/not-severe operating point
    y=np.array([1 if truth[s]["state"] in ("MAJOR","CATASTROPHIC") else 0 for s in sids])
    p=np.array([out[s]["distribution"]["MAJOR"]+out[s]["distribution"]["CATASTROPHIC"] for s in sids])
    d=np.array([out[s]["dsi"] for s in sids])
    print("\nROC-AUC (severe vs not): %.3f    Brier: %.4f"%(auc(d,y),float(((p-y)**2).mean())))

    print("\n"+"="*72); print("CALIBRATION  P(severe) vs observed frequency"); print("="*72)
    print("%-16s %8s %12s %12s"%("predicted band","n","mean pred","observed"))
    ece=0.0
    for lo,hi in [(0,.15),(.15,.3),(.3,.45),(.45,.6),(.6,.75),(.75,1.01)]:
        m=(p>=lo)&(p<hi)
        if m.sum()<5: continue
        mp,ob=float(p[m].mean()),float(y[m].mean())
        ece+=m.sum()/len(p)*abs(mp-ob)
        print("%-16s %8d %12.3f %12.3f"%("%.2f-%.2f"%(lo,hi),int(m.sum()),mp,ob))
    print("Expected Calibration Error: %.4f"%ece)

    pm=Counter(out[s]["state"] for s in sids); tm=Counter(truth[s]["state"] for s in sids)
    print("\nmix  truth: "+" ".join("%s=%.0f%%"%(k[:4],100*tm[k]/len(sids)) for k in S))
    print("mix  DRISHTI: "+" ".join("%s=%.0f%%"%(k[:4],100*pm[k]/len(sids)) for k in S))
    json.dump({"confusion":M.tolist(),"states":S,"accuracy":float(acc),
               "within_one":float(adj),"auc":float(auc(d,y)),
               "brier":float(((p-y)**2).mean()),"ece":float(ece)},
              open(os.path.join(D,"validation.json"),"w"),indent=1)
    print("\nsaved -> data/validation.json")

if __name__=="__main__": main()
