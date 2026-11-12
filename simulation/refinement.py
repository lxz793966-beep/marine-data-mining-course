"""Independent confirmation of the sixteen-grid refinement."""
import json,hashlib,math
import numpy as np
from extension import ROOT,OUT,generate_pair,gridfit,score,stats,save,write_rows
from experiment import peak
R=json.loads((ROOT/'refinement_protocol.json').read_text(encoding='utf-8'))
def fit_pair(points):return np.array([gridfit(points,2,m,np.zeros(2)) for m in [2,4]])
def main():
    values=[];rows=[]
    for si,sigma in enumerate(R['calibration_widths']):
        per=[]
        for i in range(R['calibration_n']):
            seed=R['calibration_seed_base']+si*10000+i
            _,_,a,b=next(generate_pair(seed,800,.6,sigma,distances=(0,)))
            lengths=score(fit_pair(b)-fit_pair(a));per.append(lengths)
            for m,l in zip(R['methods'],lengths):rows.append(dict(sigma=sigma,seed=seed,method=m,length=l))
        values.append(np.sort(per,axis=0)[math.ceil((len(per)+1)*.96)-1])
        print('refinement calibration',sigma,flush=True)
    thresholds=np.max(values,axis=0)
    save('refinement_thresholds.json',dict(protocol_sha256=hashlib.sha256((ROOT/'refinement_protocol.json').read_bytes()).hexdigest(),by_width=np.array(values).tolist(),shared=thresholds.tolist(),rank=1153))
    write_rows('refinement_calibration.csv.gz',rows);rows=[];groups=[];rng=np.random.default_rng(R['bootstrap_seed'])
    for si,sigma in enumerate(R['test_widths']):
        collected={d:[] for d in R['distances_km']}
        for i in range(R['test_n']):
            seed=R['test_seed_base']+si*10000+i;first=None
            for d,truth,a,b in generate_pair(seed,800,.6,sigma):
                if first is None:first=fit_pair(a)
                delta=fit_pair(b)-first;l=score(delta);e=np.linalg.norm(delta-truth,axis=1);collected[d].append((e,l))
                for j,m in enumerate(R['methods']):rows.append(dict(sigma=sigma,seed=seed,distance=d,method=m,truth_x=truth[0],truth_y=truth[1],dx=delta[j,0],dy=delta[j,1],length=l[j],error=e[j]))
        for d,v in collected.items():
            e=np.array([a[0] for a in v]);l=np.array([a[1] for a in v]);diff=e[:,1]-e[:,0];boot=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(axis=1)
            groups.append(dict(sigma=sigma,distance=d,pairs=len(v),methods={m:stats(e[:,j],l[:,j],thresholds[j],R['tail_probability']) for j,m in enumerate(R['methods'])},difference=dict(mean=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())))
        print('refinement test',sigma,flush=True)
    write_rows('refinement_test.csv.gz',rows);save('refinement_summary.json',dict(thresholds=thresholds.tolist(),groups=groups))
if __name__=='__main__':main()
