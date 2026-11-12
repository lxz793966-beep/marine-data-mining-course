"""Independent density, grid-design and hotspot-structure experiments."""
from pathlib import Path
import argparse,csv,gzip,io,itertools,json,math,hashlib,time,platform
import numpy as np
from scipy.stats import beta
import scipy

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'extension'
P=json.loads((ROOT/'extension_protocol.json').read_text(encoding='utf-8'))
METHODS=P['methods']
OLD=json.loads((ROOT/'results/closed_loop_thresholds.json').read_text(encoding='utf-8'))['shared_by_method_km']
ORIGINS=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=float)

def save(name,obj):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def write_rows(name,rows):
    with (OUT/name).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as gz:
            with io.TextIOWrapper(gz,encoding='utf-8',newline='') as text:
                w=csv.DictWriter(text,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def peak_fast(points,origin,width=2.):
    idx=np.floor((points-origin)/width).astype(np.int64)
    low=idx.min(axis=0); span=idx.max(axis=0)-low+1
    codes=(idx[:,0]-low[0])*span[1]+idx[:,1]-low[1]
    counts=np.bincount(codes,minlength=int(span.prod()))
    occupied=np.flatnonzero(counts)
    centers=origin+width*(np.column_stack((occupied//span[1],occupied%span[1]))+low+.5)
    eligible=np.all((centers>=4)&(centers<=36),axis=1)
    if not eligible.any(): raise ValueError('No eligible peak')
    selected=eligible&(counts[occupied]==counts[occupied[eligible]].max())
    return centers[selected].mean(axis=0)

def fit(points):
    peaks=np.array([peak_fast(points,o) for o in ORIGINS])
    return np.vstack([peaks[0],peak_fast(points,np.zeros(2),1.),peaks.mean(axis=0),np.median(peaks,axis=0)])

def gridfit(points,h,m,base):
    origins=base+h*np.array(list(itertools.product(range(m),repeat=2)))/m
    return np.median([peak_fast(points,o,h) for o in origins],axis=0)

def cloud(rng,center,n,rho,sigma=1.2,scenario='isotropic',period=0):
    nh=round(n*rho)
    if scenario.startswith('double') or scenario=='weight_switch':
        weight=.5 if scenario=='double_equal' else (.3 if scenario=='weight_switch' and period else .7)
        n1=round(nh*weight)
        means=np.vstack([np.repeat(center[None,:],n1,axis=0),np.repeat((center+[6,0])[None,:],nh-n1,axis=0)])
        transform=np.eye(2)*sigma
    elif scenario=='ellipse':
        means=np.repeat(center[None,:],nh,axis=0)
        angle=np.pi/4; rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        transform=rot@np.diag([1.2,2.4])
    else:
        means=np.repeat(center[None,:],nh,axis=0); transform=np.eye(2)*sigma
    hot=means+rng.normal(size=(nh,2))@transform.T
    bad=np.any((hot<0)|(hot>=40),axis=1)
    while bad.any():
        hot[bad]=means[bad]+rng.normal(size=(bad.sum(),2))@transform.T
        bad=np.any((hot<0)|(hot>=40),axis=1)
    return np.vstack([hot,rng.uniform(0,40,(n-nh,2))])

def score(delta): return np.round(np.linalg.norm(delta,axis=-1),12)
def bounds(k,n,tail=.025):
    return [0. if k==0 else float(beta.ppf(tail,k,n-k+1)),1. if k==n else float(beta.ppf(1-tail,k+1,n-k))]
def stats(errors,lengths,threshold,tail=.025):
    hit=(lengths>=1)&(lengths>threshold);k=int(hit.sum())
    return dict(mean_error=float(np.mean(errors)),p90_error=float(np.quantile(errors,.9)),mean_length=float(lengths.mean()),hits=k,rate=float(hit.mean()),bounds=bounds(k,len(hit),tail))
def comparison(delta,rng):
    boot=delta[rng.integers(0,len(delta),(P['bootstrap_repeats'],len(delta)))].mean(axis=1)
    return dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())

def generate_pair(seed,n,rho,sigma,scenario='isotropic',distances=(0,1,2,3)):
    rng=np.random.default_rng(seed); center=rng.uniform(18,20,2)
    first=cloud(rng,center,n,rho,sigma,scenario,0)
    angle=rng.uniform(0,2*np.pi); direction=np.array([np.cos(angle),np.sin(angle)])
    for d in distances:
        truth=d*direction
        second=cloud(rng,center+truth,n,rho,sigma,scenario,1)
        yield d,truth,first,second

def density_calibrate():
    cfg=P['density']; rows=[];thresholds=[]
    for c,(n,rho) in enumerate(itertools.product(cfg['n_reports'],cfg['hotspot_fractions'])):
        per_width=[]
        for si,sigma in enumerate(cfg['calibration_widths']):
            values=[]
            for i in range(cfg['calibration_n']):
                seed=cfg['calibration_seed_base']+(c*2+si)*10000+i
                _,_,a,b=next(generate_pair(seed,n,rho,sigma,distances=(0,)))
                lengths=score(fit(b)-fit(a)); values.append(lengths)
                for m,l in zip(METHODS,lengths): rows.append(dict(n=n,rho=rho,sigma=sigma,seed=seed,method=m,length=l))
            rank=math.ceil((len(values)+1)*.96)
            thresholds_w=np.sort(values,axis=0)[rank-1]; per_width.append(thresholds_w)
        thresholds.append(dict(n=n,rho=rho,rank=rank,by_width=np.array(per_width).tolist(),shared=np.max(per_width,axis=0).tolist()))
        print('calibration',n,rho,thresholds[-1]['shared'],flush=True)
    write_rows('density_calibration.csv.gz',rows)
    save('density_thresholds.json',dict(protocol_sha256=hashlib.sha256((ROOT/'extension_protocol.json').read_bytes()).hexdigest(),thresholds=thresholds))

def density_test():
    cfg=P['density'];frozen=json.loads((OUT/'density_thresholds.json').read_text()); assert frozen['protocol_sha256']==hashlib.sha256((ROOT/'extension_protocol.json').read_bytes()).hexdigest()
    thresholds={(r['n'],r['rho']):r['shared'] for r in frozen['thresholds']}
    rows=[]; groups=[]; rng_boot=np.random.default_rng(P['bootstrap_seed'])
    for c,(n,rho,sigma) in enumerate(itertools.product(cfg['n_reports'],cfg['hotspot_fractions'],cfg['test_widths'])):
        values={d:[] for d in P['distances_km']}; errors={d:[] for d in P['distances_km']}
        for i in range(cfg['test_n']):
            seed=cfg['test_seed_base']+c*10000+i
            for d,truth,a,b in generate_pair(seed,n,rho,sigma):
                delta=fit(b)-fit(a);lengths=score(delta);err=np.linalg.norm(delta-truth,axis=1)
                values[d].append(lengths);errors[d].append(err)
                for j,m in enumerate(METHODS):
                    rows.append(dict(n=n,rho=rho,sigma=sigma,seed=seed,distance=d,method=m,truth_x=truth[0],truth_y=truth[1],dx=delta[j,0],dy=delta[j,1],length=lengths[j],error=err[j]))
        for d in P['distances_km']:
            e=np.array(errors[d]);l=np.array(values[d]);r=dict(n=n,rho=rho,sigma=sigma,distance=d,pairs=len(e),methods={})
            for j,m in enumerate(METHODS):
                r['methods'][m]=dict(original=stats(e[:,j],l[:,j],OLD[m],P['test_tail_probability']),recalibrated=stats(e[:,j],l[:,j],thresholds[n,rho][j],P['test_tail_probability']))
            r['comparisons']={m:comparison(e[:,3]-e[:,j],rng_boot) for j,m in enumerate(METHODS[:3])}
            groups.append(r)
        print('density test',n,rho,sigma,flush=True)
    write_rows('density_test.csv.gz',rows); save('density_summary.json',groups)

def geometry():
    cfg=P['geometry'];rows=[];groups=[];rng_boot=np.random.default_rng(P['bootstrap_seed']+1)
    for si,sigma in enumerate(cfg['sigma_km']):
        collect={}
        for i in range(cfg['n']):
            seed=cfg['seed_base']+si*10000+i
            origin_rng=np.random.default_rng(seed+1000000)
            bases={h:origin_rng.uniform(0,h,2) for h in cfg['widths_km']}
            initial={}
            for d,truth,a,b in generate_pair(seed,800,.6,sigma):
                for h,m in itertools.product(cfg['widths_km'],cfg['origins_per_axis']):
                    if (h,m) not in initial:initial[h,m]=gridfit(a,h,m,bases[h])
                    delta=gridfit(b,h,m,bases[h])-initial[h,m];error=float(np.linalg.norm(delta-truth));length=float(score(delta))
                    collect.setdefault((h,m,d),[]).append((error,length))
                    rows.append(dict(sigma=sigma,seed=seed,distance=d,width=h,axis_origins=m,grids=m*m,origin_x=bases[h][0],origin_y=bases[h][1],truth_x=truth[0],truth_y=truth[1],dx=delta[0],dy=delta[1],length=length,error=error))
        for (h,m,d),v in collect.items():
            ar=np.array(v);base=np.array(collect[h,2,d])[:,0]
            groups.append(dict(sigma=sigma,width=h,axis_origins=m,grids=m*m,distance=d,pairs=len(v),mean_error=float(ar[:,0].mean()),p90_error=float(np.quantile(ar[:,0],.9)),mean_length=float(ar[:,1].mean()),delta_vs_four=comparison(ar[:,0]-base,rng_boot)))
        print('geometry',sigma,flush=True)
    write_rows('geometry.csv.gz',rows);save('geometry_summary.json',groups)

def structure():
    cfg=P['structure'];rows=[];groups=[];rng_boot=np.random.default_rng(P['bootstrap_seed']+2)
    for si,scenario in enumerate(cfg['scenarios']):
        ds=[0] if scenario=='weight_switch' else P['distances_km'];col={d:[] for d in ds}
        for i in range(cfg['n']):
            seed=cfg['seed_base']+si*10000+i
            for d,truth,a,b in generate_pair(seed,800,.6,1.2,scenario,ds):
                delta=fit(b)-fit(a);l=score(delta);e=np.linalg.norm(delta-truth,axis=1)
                col[d].append((e,l))
                for j,m in enumerate(METHODS):rows.append(dict(scenario=scenario,seed=seed,distance=d,method=m,truth_x=truth[0],truth_y=truth[1],dx=delta[j,0],dy=delta[j,1],length=l[j],error='' if scenario=='weight_switch' else e[j]))
        for d,v in col.items():
            e=np.array([a[0] for a in v]);l=np.array([a[1] for a in v]);r=dict(scenario=scenario,distance=d,pairs=len(v),methods={})
            for j,m in enumerate(METHODS):
                r['methods'][m]=stats(e[:,j],l[:,j],OLD[m])
                if scenario=='weight_switch':r['methods'][m]['mean_error']=r['methods'][m]['p90_error']=None
            if scenario!='weight_switch':r['comparisons']={m:comparison(e[:,3]-e[:,j],rng_boot) for j,m in enumerate(METHODS[:3])}
            groups.append(r)
        print('structure',scenario,flush=True)
    write_rows('structure.csv.gz',rows);save('structure_summary.json',groups)

def timing():
    rows=[]
    for n in [200,800,3200]:
        for i in range(35):
            rng=np.random.default_rng(50000000+n*100+i);points=cloud(rng,np.array([19.,19.]),n,.6)
            for m in rng.permutation([1,2,3,4]):
                t=time.perf_counter_ns(); gridfit(points,2,int(m),np.zeros(2)); elapsed=(time.perf_counter_ns()-t)/1e6
                if i>=5:rows.append(dict(n=n,repeat=i-5,grids=int(m*m),milliseconds=elapsed))
    write_rows('timing.csv.gz',rows)
    result=[]
    for n,g in itertools.product([200,800,3200],[1,4,9,16]):
        v=[r['milliseconds'] for r in rows if r['n']==n and r['grids']==g]
        result.append(dict(n=n,grids=g,median_ms=float(np.median(v)),q25_ms=float(np.quantile(v,.25)),q75_ms=float(np.quantile(v,.75))))
    save('timing_summary.json',dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,results=result))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['calibrate','density','geometry','structure','timing']);args=p.parse_args();OUT.mkdir(exist_ok=True)
    {'calibrate':density_calibrate,'density':density_test,'geometry':geometry,'structure':structure,'timing':timing}[args.phase]()
if __name__=='__main__':main()
