#!/usr/bin/env python3
"""Three-by-three full-disk isentropic comparison at 60, 100, and 150 h."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import plot_isentropic_jet_strength_sensitivity as s
m=s.m
CASES={'CTRL':s.FILES['CTRL'],'JET30':s.FILES['JET30']}
CENTERS=[60.,100.,150.];HALF=3.

def main():
 out=ROOT/'output/isentropic_jet_strength_CTRL_JET30_60_100_150h';out.mkdir(parents=True,exist_ok=True)
 cfg=dict(centers_h=CENTERS,half_window_h=HALF,window_samples=7,radius_km=1000.,theta_bin_K=1.,level_step_kg_s=1e7,bottom_clearance_m=1.,near_surface_m=1000.,minimum_top_m=10000.,xlim=[380.,320.],ylim=[0.,18.],cases={k:'/data/zhangyx/DATA/'+v for k,v in CASES.items()},jet_case='standard JET30 (no U30 suffix)',domain='complete 1000-km TC-centered circle')
 m.savejson(out/'config.json',cfg);results={};summaries=[];weights=[];schemas={}
 for center in CENTERS:
  start,end=center-HALF,center+HALF;cache_dir=ROOT/f'output/isentropic_jet_strength_{start:g}_{end:g}h';cache_dir.mkdir(parents=True,exist_ok=True)
  wcfg=dict(center_h=center,start_h=start,end_h=end,radius_km=cfg['radius_km'],theta_bin_K=cfg['theta_bin_K'],level_step_kg_s=cfg['level_step_kg_s'],bottom_clearance_m=cfg['bottom_clearance_m'],near_surface_m=cfg['near_surface_m'],minimum_top_m=cfg['minimum_top_m'])
  for label,file in CASES.items():
   hourly,schema=s.read_case(label,Path(cfg['cases'][label]),wcfg,cache_dir);schemas[f'{label}_{center:g}h']=schema
   a=hourly['alpha'];d={}
   for k in ['M','Fz','Fz_raw','count','domain_mass','net_raw','wbar','gross','mass_error','flux_error']:
    v=hourly[k];d[k]=np.sum(v*a.reshape((len(a),)+(1,)*(v.ndim-1)),axis=0)
   d.update(z_m=hourly['z_m'],theta_edges_K=hourly['theta_edges_K']);d['Psi']=np.c_[np.zeros(len(d['z_m'])),np.cumsum(d['Fz']*np.diff(d['theta_edges_K']),axis=1)];d['support']=(d['count']>=1)&(d['M']>0)
   cycle,selection=s.outer_cycle(f'{label}_{center:g}h',d,wcfg,out);d['cycle']=cycle
   if not cycle:raise ValueError(f'No qualified outer closed deep cycle: {label} {center:g}h')
   results[(label,center)]=d
   summary={k:v for k,v in selection.items() if k not in ['trials','abc']};summary.update(case=label,center_h=center,start_h=start,end_h=end,sample_count=len(a),weights=a.tolist(),mass_error=float(d['mass_error']),flux_error=float(d['flux_error']),Psi_endpoint_max_kg_s=float(abs(d['Psi'][:,-1]).max()))
   summaries.append(summary)
   for t,wt in zip(hourly['time_h'],a):weights.append(dict(case=label,center_h=center,time_h=float(t),weight=float(wt)))
   arrays={k:v for k,v in d.items() if isinstance(v,np.ndarray)};np.savez_compressed(out/f'{label}_{center:g}h_mean.npz',**arrays,time_h=hourly['time_h'],alpha=a)
   print(label,center,'Psi',selection['level_kg_s'],'top_km',selection['zmax_m']/1000,flush=True)
 m.savejson(out/'summary.json',summaries);m.savejson(out/'input_schemas.json',schemas);m.savecsv(out/'time_weights.csv',weights)
 peak=max(float(abs(d['Fz']).max()/1e9) for d in results.values());maximum=max(.1,float(10**np.ceil(np.log10(peak))))
 pos=np.array([.01,.02,.05,.1,.2,.5,1,2,5,10,20,50,100]);pos=np.r_[pos[pos<maximum],maximum];levels=np.r_[-pos[::-1],pos];n=len(pos)
 colors=np.vstack([plt.cm.Purples(np.linspace(.95,.2,n)),[[1,1,1,1]],plt.cm.YlOrRd(np.linspace(.15,.95,n))]);cmap=ListedColormap(colors);cmap.set_under(colors[0]);cmap.set_over(colors[-1]);norm=BoundaryNorm(levels,cmap.N);lp=np.array([.5,1,2,5,10,20]);style=(levels,cmap,norm,np.r_[-lp[::-1],lp])
 fig,axes=plt.subplots(3,3,figsize=(15,13),constrained_layout=True);ed=next(iter(results.values()))['theta_edges_K'];z=next(iter(results.values()))['z_m'];im=None
 # First row CTRL, second row standard JET30; third row overlays both outer cycles.
 for j,center in enumerate(CENTERS):
  for i,label in enumerate(['CTRL','JET30']):
   d=results[(label,center)];c=d['cycle'];title=f'({chr(97+i*3+j)}) {label}, {center:g} h\n$\\Psi={c["level_kg_s"]/1e9:.2f}\\times10^9$ kg/s'
   im=m.panel(axes[i,j],d,ed,z,style,title,cfg['xlim']);axes[i,j].set_yticks(np.arange(0,19,3))
  ax=axes[2,j]
  for label,color in [('CTRL',s.COLORS['CTRL']),('JET30',s.COLORS['JET30'])]:
   c=results[(label,center)]['cycle'];v=c['vertices'];ax.plot(v[:,0],v[:,1]/1000,color=color,lw=2,label=f'{label}: {c["level_kg_s"]:.2e} kg/s')
   for name,p in c['points'].items():ax.plot(p['theta_e_K'],p['z_m']/1000,'o',color=color,ms=3)
  ax.set(title=f'({chr(103+j)}) Outer cycles, {center:g} h',xlabel=r'Ice-reference $\theta_e$ (K)',ylabel='Height (km)',xlim=cfg['xlim'],ylim=cfg['ylim']);ax.set_yticks(np.arange(0,19,3));ax.legend(fontsize=8,frameon=False)
 fig.suptitle('Full 1000-km TC-centered isentropic circulation; 7-h means (±3 h)',fontsize=15)
 fig.colorbar(im,ax=list(axes[:2,:].flat),label=r'$F_z$ ($10^9$ kg s$^{-1}$ K$^{-1}$)',shrink=.82)
 fig.savefig(out/'isentropic_cycles_60_100_150h.png',dpi=220);fig.savefig(out/'isentropic_cycles_60_100_150h.pdf');plt.close(fig)
 m.savejson(out/'plot_config.json',dict(Fz_scale=1e9,Fz_levels=levels.tolist(),Psi_levels=style[3].tolist(),xlim=cfg['xlim'],ylim=cfg['ylim'],case_colors={k:s.COLORS[k] for k in CASES}))
 m.savejson(out/'provenance.json',dict(script=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reused_script=str(Path(s.__file__).resolve()),reused_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),reused=['read_case','bins','thermo','cycle','outer_cycle','panel']))
 notes=['# CTRL 与标准 JET30 等熵环流比较','', '完整台风中心1000 km圆域。列对应60、100、150 h中心时刻，每列使用前后各3 h的7个逐小时样本等权平均。第一行为CTRL，第二行为标准JET30，第三行为两者最外围可分辨闭合深环流线叠加。', '', '- 未作方位角平均；逐时在原三维网格计算theta_e、分箱和通量，再对窗口结果平均。', '- 横轴380→320 K，高度0–18 km；使用相同颜色标识试验。', '- 第三行每个时刻各试验的Psi等值水平单独列出，因此比较轮廓形态，不代表相同质量输送。', '- 选线标准沿用现有验证规则：1e7 kg/s扫描步长、底部1m裕量、低于1km且顶部10–18km、abc流向检查；无合格线则报错，不补造。', '- 窗口平均轮廓不是气块轨迹，数学闭合不证明能量或控制体闭合。', '', '|Case|中心时刻|窗口|Psi kg/s|顶部km|质量误差|','|---|---:|---|---:|---:|---:|']
 for r in summaries:notes.append(f"|{r['case']}|{r['center_h']:g} h|{r['start_h']:g}–{r['end_h']:g} h|{r['level_kg_s']:.3e}|{r['zmax_m']/1000:.3f}|{r['mass_error']:.2e}|")
 notes+=['','图件：`isentropic_cycles_60_100_150h.png/pdf`；均值数组、逐时权重、输入schema、选中轮廓坐标与abc均保存在本目录及同中心时刻缓存目录。','']
 (out/'README.md').write_text('\n'.join(notes),encoding='utf-8');print('Completed',out,flush=True)
if __name__=='__main__':main()
