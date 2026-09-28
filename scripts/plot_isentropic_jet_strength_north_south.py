#!/usr/bin/env python3
"""North/south TC-relative sensitivity comparison using existing validated readers."""
import inspect, json, hashlib, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import plot_isentropic_jet_strength_sensitivity as s
m=s.m
# Reuse the unchanged reader; append hemisphere bins while native fields are in memory.
source=inspect.getsource(s.read_case)
source=source.replace("theta_bin_K=cfg['theta_bin_K'])", "theta_bin_K=cfg['theta_bin_K'],partition='TC-relative y: North >=0, South <0; independent side wbar')")
source=source.replace("   d=m.bins(theta,rho,w,area,mask,ed)", """   d=m.bins(theta,rho,w,area,mask,ed)
   sides={name:m.bins(theta,rho,w,area,mask & test,ed) for name,test in [('North',yy>=0),('South',yy<0)]}
   for name,part in sides.items():
    err=float(np.max(abs(np.sum(part['Fz_raw']*np.diff(ed),axis=1)-part['net_raw'])/np.maximum(part['gross']+abs(part['net_raw']),1)))
    if err>1e-10:raise ValueError('Hemisphere raw flux recovery failed')
    d.update({name+'_'+key:value for key,value in part.items()})
   for key in ['M','Fz_raw','count','domain_mass','net_raw']:
    if not np.allclose(sides['North'][key]+sides['South'][key],d[key],rtol=1e-10,atol=1e-3):raise ValueError('Hemisphere partition recovery failed: '+key)""")
namespace=dict(s.__dict__)
exec(compile(source,__file__+':reader_adapter','exec'),namespace)
read_case=namespace['read_case']

def main():
 cfg=dict(center_h=100.,start_h=97.,end_h=103.,radius_km=1000.,theta_bin_K=1.,level_step_kg_s=1e7,bottom_clearance_m=1.,near_surface_m=1000.,minimum_top_m=10000.,xlim=[380.,320.],ylim=[0.,18.],cases={label:'/data/zhangyx/DATA/'+file for label,file in s.FILES.items()},partition='North y>=TC centre; South y<TC centre',mean_removal='independent dry-air mass-weighted wbar in each hemisphere')
 out=ROOT/'output/isentropic_jet_strength_North_South_97_103h';out.mkdir(parents=True,exist_ok=True);m.savejson(out/'config.json',cfg)
 results={};summaries=[];schemas={};weights=[]
 keys=['M','Fz','Fz_raw','count','domain_mass','net_raw','wbar','gross','mass_error','flux_error']
 for label,path in cfg['cases'].items():
  hourly,schema=read_case(label,Path(path),cfg,out);schemas[label]=schema
  a=hourly['alpha']
  assert np.array_equal(hourly['time_h'],np.arange(97.,104.)) and np.allclose(a,1/7)
  for side in ['North','South']:
   name=label+'_'+side;d={}
   for key in keys:
    array=hourly[side+'_'+key];d[key]=np.sum(array*a.reshape((len(a),)+(1,)*(array.ndim-1)),axis=0)
   d.update(z_m=hourly['z_m'],theta_edges_K=hourly['theta_edges_K'])
   d['Psi']=np.c_[np.zeros(len(d['z_m'])),np.cumsum(d['Fz']*np.diff(d['theta_edges_K']),axis=1)]
   d['support']=(d['count']>=1)&(d['M']>0)
   cycle,report=s.outer_cycle(name,d,cfg,out);d['cycle']=cycle;results[name]=d
   report={key:value for key,value in report.items() if key not in ['trials','abc']}
   report.update(hemisphere=side,experiment=label,mass_recovery_error=float(d['mass_error']),corrected_flux_recovery_error=float(d['flux_error']),Psi_endpoint_max_abs_kg_s=float(abs(d['Psi'][:,-1]).max()))
   summaries.append(report)
   arrays={key:value for key,value in d.items() if isinstance(value,np.ndarray)}
   np.savez_compressed(out/(name+'_mean.npz'),**arrays,time_h=hourly['time_h'],alpha=a,M_t=np.gradient(hourly[side+'_M'],hourly['time_h']*3600,axis=0))
   print(name,report.get('level_kg_s'),report.get('zmax_m'),flush=True)
  weights.extend(dict(case=label,time_h=float(t),weight=float(w)) for t,w in zip(hourly['time_h'],a))
 m.savejson(out/'summary.json',summaries);m.savejson(out/'input_schemas.json',schemas);m.savecsv(out/'time_weights.csv',weights)
 peak=max(float(abs(d['Fz']).max()/1e9) for d in results.values());maximum=max(.1,float(10**np.ceil(np.log10(peak))))
 pos=np.array([.01,.02,.05,.1,.2,.5,1,2,5,10,20,50,100]);pos=np.r_[pos[pos<maximum],maximum];levels=np.r_[-pos[::-1],pos];n=len(pos)
 colors=np.vstack([plt.cm.Purples(np.linspace(.95,.2,n)),[[1,1,1,1]],plt.cm.YlOrRd(np.linspace(.15,.95,n))]);cmap=ListedColormap(colors);cmap.set_under(colors[0]);cmap.set_over(colors[-1]);norm=BoundaryNorm(levels,cmap.N);lp=np.array([.5,1,2,5,10,20]);style=(levels,cmap,norm,np.r_[-lp[::-1],lp])
 def overlay(ax):
  for label in s.FILES:
   for side,line in [('North','-'),('South','--')]:
    cycle=results[label+'_'+side]['cycle']
    if cycle:
     v=cycle['vertices'];ax.plot(v[:,0],v[:,1]/1000,color=s.COLORS[label],ls=line,lw=1.6,label=label+' '+side)
    else:ax.plot([],[],color=s.COLORS[label],ls=line,label=label+' '+side+' (unavailable)')
  ax.set(title='(k) Outermost resolved closed contours: North / South',xlabel=r'Ice-reference $\theta_e$ (K)',ylabel='Height (km)',xlim=cfg['xlim'],ylim=cfg['ylim']);ax.set_yticks(np.arange(0,19,3));ax.legend(loc='upper right',ncol=2,fontsize=10,frameon=False)
 def save(fig,name):
  fig.savefig(out/(name+'.png'),dpi=200);fig.savefig(out/(name+'.pdf'));plt.close(fig)
 fig=plt.figure(figsize=(23,16),layout='constrained');grid=fig.add_gridspec(3,5);axes=[]
 for row,side in enumerate(['North','South']):
  for col,label in enumerate(s.FILES):
   ax=fig.add_subplot(grid[row,col]);axes.append(ax);d=results[label+'_'+side];cycle=d['cycle']
   level=f"$\\Psi={cycle['level_kg_s']/1e9:.2f}\\times10^9$ kg/s" if cycle else 'No qualified closed deep contour'
   im=m.panel(ax,d,d['theta_edges_K'],d['z_m'],style,f'({chr(97+row*5+col)}) {label} — {side}\n{level}',cfg['xlim']);ax.set_yticks(np.arange(0,19,3))
 overlay(fig.add_subplot(grid[2,1:4]));fig.suptitle('Jet-strength sensitivity: TC-relative North / South, 97–103 h time mean',fontsize=18)
 fig.colorbar(im,ax=axes,label=r'$F_z$ ($10^9$ kg s$^{-1}$ K$^{-1}$)',shrink=.75);save(fig,'isentropic_strength_North_South')
 fig,ax=plt.subplots(figsize=(9,8),layout='constrained');overlay(ax);fig.suptitle('97–103 h time mean');save(fig,'outer_cycles_North_South')
 m.savejson(out/'plot_config.json',dict(Fz_scale=1e9,Fz_levels=levels.tolist(),Psi_levels=style[3].tolist(),colors=s.COLORS,North_linestyle='solid',South_linestyle='dashed',xlim=cfg['xlim'],ylim=cfg['ylim']))
 m.savejson(out/'provenance.json',dict(script=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reader_adapter_source=source,reused_script=str(Path(s.__file__).resolve()),reused_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),python=sys.version))
 text=['# 100 h窗口：TC南北侧等熵敏感性对照','','97–103 h的7个真实时次等权平均；CTRL、JET15、标准无后缀JET30、JET45、JET60。','','- 前5子图北侧，后5子图南侧，第11图叠加10条轮廓；颜色区分试验，实线北侧、虚线南侧。','- 每时刻以原中心跟踪方法划分1000 km圆域：y>=中心为北，y<中心为南。中心排网格仅归北，无重复/遗漏；这是网格中心划分，并非精确切分中心排面积。','- 无预先方位平均。逐格热力学、分侧分箱后窗口平均，再积分Psi和提线。','- 两侧独立扣除各侧每层干空气质量加权平均w。保留原始Fz_raw、net_raw和wbar；原始质量及通量可恢复整圆，但独立修正后的南北Fz之和不必等于整圆修正Fz。','- 复用前版热力学、中心、读取与选线实现；只新增独立适配入口，未修改原脚本或核心。','- 每侧独立选最外围可分辨闭合深胞，沿用1e7 kg/s步长、1m底部裕量、最低<=1km、顶部10–18km、abc方向检查；各线输送量不同，比较形状。没有合格线时明示缺失，不补造。','- 数学闭合不证明半圆控制体质量/能量闭合，南北交换和非稳态尚未作预算。','- 半圆通量是该半圆的实际积分，不乘2，因此幅度不能直接与整圆积分等同。','- 横轴380→320 K、高度0–18km；统一10子图填色尺度。','','|Case|Side|Psi kg/s|Top km|','|---|---|---:|---:|']
 for report in summaries:
  text.append(f"|{report['experiment']}|{report['hemisphere']}|{report.get('level_kg_s','unavailable')}|{report.get('zmax_m',float('nan'))/1000:.3f}|")
 text+=['','PNG/PDF：isentropic_strength_North_South（11子图）、outer_cycles_North_South（独立叠加）。逐时缓存保存整圆及North_/South_分箱量，各侧_mean.npz保存均值、Psi与M_t。其余选线、中心、时间、输入及检查记录均在此目录。','','复跑：','```sh','cd /data1/home/zhangyx/project/TC_dynamic','/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python scripts/plot_isentropic_jet_strength_north_south.py','```']
 (out/'README.md').write_text('\n'.join(text),encoding='utf-8');print('Completed',out,flush=True)
if __name__=='__main__':main()
