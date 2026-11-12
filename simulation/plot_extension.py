"""Figures for density robustness, design sensitivity and structural stress tests."""
from pathlib import Path
import json,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from plot_closed_loop import style
from extension import OUT,cloud,fit

M=['single_2km','single_1km','four_mean','four_median']
LABELS=['2 km 单网格','1 km 单网格','四网格均值','四网格中位数']
C=['#5d7182','#98a8b2','#1f6378','#a85423']
def load(s):return json.loads((OUT/s).read_text(encoding='utf-8'))
def save(fig,name):
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{suffix}',dpi=400,facecolor='white',bbox_inches='tight')
    plt.close(fig)
def legend(fig,ax):fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=7.5,bbox_to_anchor=(.52,-.01))

def density():
    data=load('density_summary.json')
    fig,axes=plt.subplots(1,3,figsize=(6.2,2.6),sharey=True)
    for ax,rho,tag in zip(axes,[.3,.6,.9],'abc'):
        for m,label,c in zip(M,LABELS,C):
            y=[np.mean([r['methods'][m]['original']['mean_error'] for r in data if r['n']==n and r['rho']==rho and r['distance']==1]) for n in [200,400,800,1600]]
            ax.plot(range(4),y,marker='o',ms=3,lw=1.1,color=c,label=label)
        ax.set_title(f'{tag}  热点占比 {rho:.0%}',loc='left',fontweight='bold',fontsize=9)
        ax.set_xticks(range(4),['200','400','800','1600'],fontsize=7);ax.set_xlabel('每期报文数');ax.grid(axis='y',lw=.4,color='#dce1e4')
    axes[0].set_ylabel('1 km 位移的平均向量误差 (km)');legend(fig,axes[0]);fig.subplots_adjust(bottom=.27,wspace=.16);save(fig,'density_errors')

def calibration():
    data=load('density_summary.json');matrices=[np.zeros((4,3)) for _ in range(3)]
    for i,n in enumerate([200,400,800,1600]):
        for j,rho in enumerate([.3,.6,.9]):
            rows=[r for r in data if r['n']==n and r['rho']==rho]
            null=[r['methods']['four_median'] for r in rows if r['distance']==0]
            alt=[r['methods']['four_median'] for r in rows if r['distance']==2]
            matrices[0][i,j]=max(r['original']['rate'] for r in null)*100
            matrices[1][i,j]=max(r['recalibrated']['bounds'][1] for r in null)*100
            matrices[2][i,j]=min(r['recalibrated']['bounds'][0] for r in alt)*100
    fig,axes=plt.subplots(1,3,figsize=(6.2,2.65))
    for ax,values,title,vmax in zip(axes,matrices,['a  原阈值最大误报率','b  重校准最大误报上界','c  重校准 2 km 检出下界'],[max(10,matrices[0].max()),max(5,matrices[1].max()),100]):
        ax.imshow(values,cmap='Blues',vmin=0,vmax=vmax,aspect='auto')
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:.1f}%',ha='center',va='center',fontsize=8,color='white' if v>.65*vmax else '#17232a')
        ax.set_xticks(range(3),['30%','60%','90%']);ax.set_yticks(range(4),[200,400,800,1600]);ax.set_xlabel('热点报文占比');ax.set_title(title,fontsize=8.5,loc='left',fontweight='bold')
    axes[0].set_ylabel('每期报文数');fig.subplots_adjust(wspace=.30,bottom=.20,top=.85);save(fig,'density_calibration')

def geometry():
    data=load('geometry_summary.json');fig,axes=plt.subplots(1,2,figsize=(6.0,2.75),sharey=True)
    matrices=[]
    for sigma in [1.2,2.]:matrices.append(np.array([[next(r['mean_error'] for r in data if r['sigma']==sigma and r['width']==h and r['grids']==g and r['distance']==1) for g in [1,4,9,16]] for h in [1,2,3,4]]))
    top=max(a.max() for a in matrices)
    for ax,sigma,values,tag in zip(axes,[1.2,2.],matrices,'ab'):
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=top,aspect='auto')
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,color='white' if v>.65*top else '#17232a')
        ax.set_xticks(range(4),[1,4,9,16]);ax.set_yticks(range(4),[1,2,3,4]);ax.set_xlabel('网格数量');ax.set_title(f'{tag}  热点标准差 {sigma:g} km',loc='left',fontweight='bold',fontsize=9)
    axes[0].set_ylabel('格宽 (km)');fig.subplots_adjust(wspace=.18,bottom=.20,right=.83,top=.86);cb=fig.colorbar(im,cax=fig.add_axes([.87,.20,.025,.65]));cb.set_label('1 km 位移的平均误差 (km)',fontsize=8);save(fig,'grid_geometry')

def structural():
    data=load('structure_summary.json');scenarios=['isotropic','ellipse','double_dominant','double_equal'];names=['圆形单峰','椭圆单峰','主次双峰','等强双峰']
    fig,axes=plt.subplots(1,2,figsize=(6.2,2.9))
    for j,(m,label,c) in enumerate(zip(M,LABELS,C)):
        for ax,d,kind in zip(axes,[2,0],['mean_error','rate']):
            vals=[next(r['methods'][m][kind] for r in data if r['scenario']==s and r['distance']==d) for s in scenarios]
            ax.bar(np.arange(4)+(j-1.5)*.19,vals,width=.18,color=c,label=label)
    for ax in axes:ax.set_xticks(range(4),names,fontsize=7);ax.grid(axis='y',lw=.4,color='#dce1e4');ax.set_axisbelow(True)
    axes[0].set_title('a  真位移 2 km',loc='left',fontweight='bold');axes[0].set_ylabel('平均向量误差 (km)')
    axes[1].set_title('b  热点结构保持静止',loc='left',fontweight='bold');axes[1].set_ylabel('原校准阈值下的误报率');axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0));axes[1].axhline(.05,ls=':',color='#444',lw=.8)
    legend(fig,axes[0]);fig.subplots_adjust(left=.10,right=.99,bottom=.25,top=.84,wspace=.36);save(fig,'structure_results')

def examples():
    fig,axes=plt.subplots(1,3,figsize=(6.3,2.7),sharex=True,sharey=True)
    rng=np.random.default_rng(80000000);center=rng.uniform(18,20,2);source=[]
    labels=['a  椭圆热点','b  双热点初期 70:30','c  双热点后期 30:70']
    for ax,scenario,period,label in zip(axes,['ellipse','weight_switch','weight_switch'],[0,0,1],labels):
        points=cloud(rng,center,800,.6,1.2,scenario,period);est=fit(points)[3]
        source.extend([dict(panel=label,point=i,x=p[0],y=p[1],component='hotspot' if i<480 else 'background') for i,p in enumerate(points)])
        ax.scatter(points[480:,0],points[480:,1],s=4,c='#d7dce0',alpha=.45)
        ax.scatter(points[:480,0],points[:480,1],s=4,c='#557b8b',alpha=.45)
        centers=np.array([center]) if scenario=='ellipse' else np.array([center,center+[6,0]])
        ax.scatter(centers[:,0],centers[:,1],marker='+',c='black',s=45,lw=1,label='生成中心')
        ax.scatter(*est,marker='D',edgecolors='#a85423',facecolors='none',s=32,lw=1.1,label='四网格中位数')
        ax.set_xlim(12,32);ax.set_ylim(12,28);ax.set_aspect('equal');ax.set_title(label,loc='left',fontsize=8.5,fontweight='bold');ax.set_xlabel('东方 (km)');ax.grid(lw=.4,color='#dce1e4')
    axes[0].set_ylabel('北方 (km)');fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='lower center',fontsize=8);fig.subplots_adjust(bottom=.24,wspace=.10);save(fig,'structure_design')
    from extension import write_rows
    write_rows('structure_design_points.csv.gz',source)

def efficiency():
    data=load('timing_summary.json')['results'];fig,ax=plt.subplots(figsize=(6,2.6))
    for n,c,marker in zip([200,800,3200],['#5d7182','#1f6378','#a85423'],['s','o','^']):
        rows=[r for r in data if r['n']==n];g=[r['grids'] for r in rows];v=np.array([r['median_ms'] for r in rows]);lo=np.array([r['q25_ms'] for r in rows]);hi=np.array([r['q75_ms'] for r in rows])
        ax.errorbar(g,v,yerr=[v-lo,hi-v],marker=marker,c=c,lw=1.2,capsize=2,label=f'{n} 条报文')
    ax.set_xticks([1,4,9,16]);ax.set_xlabel('网格数量');ax.set_ylabel('单期定位耗时 (ms)');ax.grid(axis='y',lw=.4,color='#dce1e4');ax.legend(ncol=3,loc='upper left',fontsize=8);fig.tight_layout();save(fig,'computational_cost')

def refinement():
    data=load('refinement_summary.json')['groups'];fig,axes=plt.subplots(1,2,figsize=(6.0,2.7),sharey=True)
    for ax,d,tag in zip(axes,[1,2],'ab'):
        for j,(m,label,c) in enumerate(zip(['four_median','sixteen_median'],['四网格中位数','十六网格中位数'],['#a85423','#1f6378'])):
            rows=[r for r in data if r['distance']==d]
            ax.bar(np.arange(3)+(j-.5)*.28,[r['methods'][m]['mean_error'] for r in rows],width=.27,color=c,label=label)
        ax.set_xticks(range(3),['1.2','1.6','2.0']);ax.set_xlabel('热点标准差 (km)');ax.set_title(f'{tag}  真位移 {d} km',loc='left',fontweight='bold');ax.grid(axis='y',lw=.4,color='#dce1e4');ax.set_axisbelow(True)
    axes[0].set_ylabel('平均向量误差 (km)');fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='lower center',fontsize=8);fig.subplots_adjust(bottom=.25,wspace=.15);save(fig,'refinement_validation')

if __name__=='__main__':
    style();plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    for func in [density,calibration,geometry,structural,examples,efficiency,refinement]:func()
