#!/usr/bin/env python3
"""Select a common interior deep contour from saved window-mean Psi."""
from pathlib import Path
import csv,hashlib,importlib.util,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/li2023_fig5_thompson_CTRL_JET30_72_78h'
OUT=BASE/'selected_cycle';OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('structure',ROOT/'scripts/plot_li2023_isentropic_time_mean.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
windows=[('74_76h','74–76 h time mean'),('73_77h','73–77 h time mean'),('72_78h','72–78 h time mean')]
data={}
for label in ['CTRL','JET30']:
 data[label]={}
 for name,_ in windows:
  with np.load(BASE/f'{label}_{name}_R1000_dtheta1.npz') as a:d={k:a[k] for k in a.files}
  if not np.all(np.isfinite(d['Psi'])):raise ValueError('Undefined cumulative Psi')
  data[label][name]=d
# A common amplitude slightly beyond every bottom-layer range, with a declared margin.
bottom=max(float(abs(d['Psi'][0].min())) for case in data.values() for d in case.values())
start=np.ceil(1.2*bottom/1e8)*1e8
limit=min(float(-d['Psi'].min()) for case in data.values() for d in case.values())
trials=[];selected=None
for amp in np.arange(start,limit+1,1e8):
 circles={};ok=True;record=dict(amplitude_kg_s=float(amp),cases=[])
 for label,case in data.items():
  circles[label]={}
  for name,d in case.items():
   # Cumulative Psi is defined even in zero-count bins; no state interpolation is done.
   c,candidates=m.cycle(d['theta_edges_K'],d['z_m'],d['Psi'],np.ones_like(d['support'],bool),-amp,1000.)
   good=bool(c is not None and 10000<=c['points']['c']['z_m']<=18000 and c['abc_in_flow_order'])
   record['cases'].append(dict(case=label,window=name,qualified=good,top_km=c['points']['c']['z_m']/1000 if c else None))
   circles[label][name]=(c,candidates);ok &=good
 trials.append(record)
 if ok:selected=float(amp);break
if selected is None:raise ValueError('No common interior deep contour under declared rule')
rows=[]
for label,case in data.items():
 for name,d in case.items():
  c,candidates=circles[label][name];d['cycle']=c;v=c['vertices'];ed=d['theta_edges_K'];z=d['z_m']
  # Quantify local-state support separately from mathematical contour validity.
  iz=np.argmin(abs(v[:,1,None]-z[None,:]),axis=1)
  jt=np.clip(np.searchsorted(ed,v[:,0],side='right')-1,0,len(ed)-2)
  zero=int(np.sum(d['count'][iz,jt]==0))
  row=dict(case=label,window=name,level_kg_s=-selected,zmin_m=float(v[:,1].min()),zmax_m=float(v[:,1].max()),points=len(v),closed=bool(np.allclose(v[0],v[-1],atol=1e-8)),abc_in_flow_order=c['abc_in_flow_order'],zero_count_nearest_box_vertices=zero)
  rows.append(row)
  m.savecsv(OUT/f'{label}_{name}_cycle.csv',[dict(point=i,theta_e_K=float(a[0]),z_m=float(a[1])) for i,a in enumerate(v)])
  m.savejson(OUT/f'{label}_{name}_cycle.json',dict(**row,abc=c['points'],candidates=candidates))
  print(label,name,'selected',-selected,'top',row['zmax_m']/1000,'zero-count vertices',zero,flush=True)
style_cfg=json.loads((BASE/'plot_config.json').read_text())
levels=np.array(style_cfg['Fz_levels']);n=len(levels)//2
colors=np.vstack([plt.cm.Purples(np.linspace(.95,.2,n)),[[1,1,1,1]],plt.cm.YlOrRd(np.linspace(.15,.95,n))])
cmap=ListedColormap(colors);cmap.set_under(colors[0]);cmap.set_over(colors[-1]);norm=BoundaryNorm(levels,cmap.N)
style=(levels,cmap,norm,np.array(style_cfg['Psi_levels']));xlim=(380.,320.)
ed=data['CTRL']['72_78h']['theta_edges_K'];z=data['CTRL']['72_78h']['z_m']
def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=220);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,6),constrained_layout=True)
for ax,label in zip(axes,data):im=m.panel(ax,data[label]['72_78h'],ed,z,style,label,xlim)
fig.suptitle(r'72–78 h time mean; selected $\Psi=-1.3\times10^9$ kg s$^{-1}$',fontsize=13)
fig.colorbar(im,ax=axes,label=r'$F_z$ ($10^9$ kg s$^{-1}$ K$^{-1}$)',shrink=.8);save(fig,'main_selected_cycle')
fig,axes=plt.subplots(2,3,figsize=(14,10),constrained_layout=True)
for i,label in enumerate(data):
 for j,(name,title) in enumerate(windows):im=m.panel(axes[i,j],data[label][name],ed,z,style,f'{label}: {title}',xlim)
fig.suptitle(r'Common selected $\Psi=-1.3\times10^9$ kg s$^{-1}$',fontsize=14)
fig.colorbar(im,ax=axes,label=r'$F_z$ ($10^9$ kg s$^{-1}$ K$^{-1}$)',shrink=.75);save(fig,'window_selected_cycles')
fig,ax=plt.subplots(figsize=(6,7),constrained_layout=True)
for label,color in [('CTRL','red'),('JET30','blue')]:
 c=data[label]['72_78h']['cycle'];v=c['vertices'];ax.plot(v[:,0],v[:,1]/1000,color=color,label=label,lw=2)
 for name,p in c['points'].items():
  ax.plot(p['theta_e_K'],p['z_m']/1000,'o',color=color,ms=4)
  ax.annotate(name,(p['theta_e_K'],p['z_m']/1000),xytext=(5,6 if label=='CTRL' else -12),textcoords='offset points',color=color)
ax.set(xlabel=r'Ice-reference $\theta_e$ (K)',ylabel='Height (km)',title=r'72–78 h: $\Psi=-1.3\times10^9$ kg s$^{-1}$',xlim=xlim,ylim=(0,18));ax.legend();save(fig,'cycle_comparison')
m.savejson(OUT/'selection.json',dict(level_kg_s=-selected,max_bottom_amplitude_kg_s=bottom,minimum_bottom_margin=1.2,amplitude_step_kg_s=1e8,rule='weakest common amplitude beyond 1.2 times maximum bottom range; closed, interior, zmin<=1km and 10km<=top<=18km, abc ordered; deepest candidate then largest area',Psi_mask='only finite cumulative field; local occupancy mask retained for Fz, not imposed on Psi contour',trials=trials,metrics=rows))
m.savejson(OUT/'provenance.json',dict(script=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_dir=str(BASE),reused_cycle_extractor='plot_li2023_isentropic_time_mean.cycle',reused_plotter='plot_li2023_isentropic_time_mean.panel'))
text=['# 自定义代表性深环流线','',f'共同选择 Psi=-{selected:.4e} kg/s。所有正式图采用窗口平均，横轴380→320K。','',
'## 选线逻辑','',
f'1. 使用已保存的CTRL/JET30三个窗口平均Psi。六张图最低25m层的最大负幅度为{bottom:.6e} kg/s。',
'2. 以这个底层幅度的1.2倍为起点，向上取整到1e8 kg/s；随后按1e8步长查找。留出20%数值裕量，避免勉强贴着25m边界闭合。20%是本次公开的操作选择，不是物理常数。',
'3. 对两组所有窗口使用同一个signed幅度；要求闭合、不触及高度/θ边界、最低点<=1km、最高点在10–18km、a→b→c符合流向。满足条件的最小幅度作为代表线，以尽可能保留深环流外围结构。',
'4. 同一级多个合格胞时选顶部最高者，顶部相同选θ–z面积最大者，不依照预期JET/CTRL差异挑线。',
'5. 本次第一候选1.3e9 kg/s即在六张图中通过。保留所有候选与检验记录于selection.json。',
'6. 提取对象是有限的累积Psi场，不用局地count/质量条件平均掩膜切断Psi；空箱累积通量本来有定义，不补造局地热力学状态。Fz填色仍采用原来的局地有效覆盖掩膜。','',
'## 标记与限制','',
'- a整条线最低θe，b近地面<=1km段最高θe，c最高高度；沿(-Psi_z,Psi_theta)定流向。',
'- 这条线是选择出来用于结构比较的等值线，不是台风全部质量输送的外边界，也不代表真实气块轨迹。',
'- 相比Li弱幅度1e7，选得更靠环流胞内部；顶部约13.8km会低于弱外围线约14.6–14.7km。顶部变化部分来自选线规则，不能全归因于物理差异。',
'- 未补地面数据，线底部位于25m以上。其闭合是当前窗口平均累积场的数学闭合，不能作为实际三维控制体闭合证据。',
'- 局地状态支持单独报告zero_count_nearest_box_vertices；这只是最近箱覆盖提示，不是严格沿线状态验证，不开展能量/耗散积分。','',
'## 复跑','', '```sh','cd /data1/home/zhangyx/project/TC_dynamic','/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python scripts/select_li_style_representative_cycle.py','```','',
'## 图与数据','', '- main_selected_cycle.png/pdf：72–78h主图；window_selected_cycles.png/pdf：3/5/7样本平均；cycle_comparison.png/pdf：两组粗线叠加。',
'- *_cycle.csv/json：六条平均轮廓与abc；selection.json：完整选线规则、候选及数值。','']
(OUT/'README.md').write_text('\n'.join(text),encoding='utf-8')
print('Completed',OUT)
