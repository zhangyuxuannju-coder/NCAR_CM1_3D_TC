#!/usr/bin/env python3
"""Window-mean isentropic jet-strength sensitivity, reusing project diagnostics."""
import argparse,csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def module(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('isentropic_structure','scripts/plot_li2023_isentropic_time_mean.py')
tracking=module('centre_baseline','scripts/diagnose_thompson_intensity_baseline.py')
FILES={'CTRL':'cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc','JET15':'cm1out_25N_JET_R2_Thompson_OML100_U15_240h.nc','JET30':'cm1out_25N_JET_R2_Thompson_OML100_240h.nc','JET45':'cm1out_25N_JET_R2_Thompson_OML100_U45_240h.nc','JET60':'cm1out_25N_JET_R2_Thompson_OML100_U60_240h.nc'}
COLORS={'CTRL':'#222222','JET15':'#0072B2','JET30':'#E69F00','JET45':'#009E73','JET60':'#CC79A7'}

def read_case(label,path,cfg,out):
 cache=out/f'{label}_hourly_bins.npz';schemafile=out/f'{label}_schema.json'
 identity=dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,start=cfg['start_h'],end=cfg['end_h'],radius_km=cfg['radius_km'],theta_bin_K=cfg['theta_bin_K'])
 if cache.exists() and schemafile.exists():
  schema=json.loads(schemafile.read_text())
  if schema['identity']==identity:
   with np.load(cache) as a:d={k:a[k] for k in a.files}
   print(label,'reused verified binned cache',flush=True);return d,schema
 with Dataset(path) as ds:
  required=['th','prs','rho','qv','qc','qr','qi','qs','qg','w','zhval','psfc']
  units=dict(th='K',prs='Pa',rho='kg/m^3',w='m/s',time='seconds',xh='km',yh='km',xf='km',yf='km',zh='km',zf='km',zhval='m',psfc='Pa',qv='kg/kg',qc='kg/kg',qr='kg/kg',qi='kg/kg',qs='kg/kg',qg='kg/kg')
  for n,u in units.items():
   if ds[n].units!=u:raise ValueError(f'{label} {n}: unexpected unit')
  for n in required:
   expected=('time','yh','xh') if n=='psfc' else ('time','zf','yh','xh') if n=='w' else ('time','zh','yh','xh')
   if ds[n].dimensions!=expected:raise ValueError(f'{label} {n}: unexpected dimensions')
  if ds['rho'].long_name!='dry-air density':raise ValueError('Density basis unverified')
  g={n:np.array(ds[n][:],float)*1000 for n in ['xh','yh','zh','xf','yf','zf']}
  t=np.array(ds['time'][:],float)/3600;idx,alpha=m.weights(t,cfg['start_h'],cfg['end_h'])
  if not np.isclose(t[idx[0]],cfg['start_h']) or not np.isclose(t[idx[-1]],cfg['end_h']):raise ValueError('Window endpoints missing')
  if len(idx)<2:raise ValueError('Too few window samples')
  centre={};previous=None;records=[]
  for it in range(int(idx[-1])+1):
   raw=np.ma.filled(ds['psfc'][it],np.nan).astype(float)
   smooth=gaussian_filter(raw,sigma=2.)
   iy,ix,fallback=tracking._choose_center(smooth,g['xh']/1000,g['yh']/1000,previous,180.)
   previous=(g['xh'][ix]/1000,g['yh'][iy]/1000)
   if it in idx:
    centre[it]=(g['xh'][ix],g['yh'][iy])
    records.append(dict(case=label,index=it,time_h=float(t[it]),center_x_km=previous[0],center_y_km=previous[1],tracking_fallback=int(fallback),psfc_center_hpa=float(raw[iy,ix]/100),psfc_domain_min_hpa=float(np.nanmin(raw)/100)))
  m.savecsv(out/f'{label}_centres.csv',records)
  samples=[];checks=[];R=cfg['radius_km']*1000;z=g['zh'];ed=np.arange(200,700+cfg['theta_bin_K']/2,cfg['theta_bin_K'])
  for it in idx:
   cx,cy=centre[int(it)];margin=min(cx-g['xf'][0],g['xf'][-1]-cx,cy-g['yf'][0],g['yf'][-1]-cy)
   if R>margin:raise ValueError('TC disk incomplete')
   xxidx=np.flatnonzero(abs(g['xh']-cx)<=R);yyidx=np.flatnonzero(abs(g['yh']-cy)<=R)
   sx=slice(xxidx[0],xxidx[-1]+1);sy=slice(yyidx[0],yyidx[-1]+1)
   def field(n):return np.ma.filled(ds[n][it,:,sy,sx],np.nan).astype(float)
   xx,yy=np.meshgrid(g['xh'][sx]-cx,g['yh'][sy]-cy);mask=np.hypot(xx,yy)<=R
   area=np.diff(g['yf'])[sy,None]*np.diff(g['xf'])[None,sx]
   th=field('th');p=field('prs');rho=field('rho');rv=field('qv');water=[field(n) for n in ['qc','qr','qi','qs','qg']]
   minq=min(float(a.min()) for a in [rv]+water);neg=sum(int(np.sum(a<0)) for a in [rv]+water)
   if minq<-1e-8:raise ValueError('Significant negative water mixing ratio')
   rv=np.maximum(rv,0);water=[np.maximum(a,0) for a in water]
   theta,T=m.thermo(th,p,rv,water[0]+water[1],water[2]+water[3]+water[4])
   if theta.min()<ed[0] or theta.max()>=ed[-1]:raise ValueError('Theta bins incomplete')
   wf=field('w');fraction=(z-g['zf'][:-1])/np.diff(g['zf'])
   w=wf[:-1]*(1-fraction[:,None,None])+wf[1:]*fraction[:,None,None]
   herror=float(abs(field('zhval')-z[:,None,None]).max())
   if herror>.02:raise ValueError('Heights need explicit remapping')
   eos=p/((m.C['Rd']+m.C['Rv']*rv)*T);eerror=float((abs(rho-eos)/eos).max())
   if eerror>1e-4:raise ValueError('Dry density/temperature EOS mismatch')
   d=m.bins(theta,rho,w,area,mask,ed)
   rawerror=float(np.max(abs(np.sum(d['Fz_raw']*np.diff(ed),axis=1)-d['net_raw'])/np.maximum(d['gross']+abs(d['net_raw']),1)))
   if rawerror>1e-10:raise ValueError('Raw flux recovery failed')
   checks.append(dict(case=label,time_h=float(t[it]),EOS_error=eerror,height_error_m=herror,mass_recovery_error=float(d['mass_error']),corrected_flux_recovery_error=float(d['flux_error']),raw_flux_recovery_error=rawerror,theta_min_K=float(theta[mask[None,:,:].repeat(len(z),axis=0)].min()),theta_max_K=float(theta.max()),negative_water_clipped=neg,minimum_water=minq,full_disk=True))
   samples.append(d);print(f'{label} {t[it]:g}h: native binning PASS',flush=True)
  stack={k:np.stack([d[k] for d in samples]) for k in samples[0]}
  stack.update(time_h=t[idx],alpha=alpha,z_m=z,theta_edges_K=ed,M_t=np.gradient(np.stack([d['M'] for d in samples]),t[idx]*3600,axis=0))
  np.savez_compressed(cache,**stack);m.savecsv(out/f'{label}_checks.csv',checks)
  schema=dict(identity=identity,dimensions={n:len(v) for n,v in ds.dimensions.items()},variables={n:dict(dimensions=list(ds[n].dimensions),units=ds[n].units,long_name=ds[n].long_name) for n in required},model_top_m=float(g['zf'][-1]),constant_source='previous validated CM1 constants; independent EOS check repeated',constants=m.C,water_basis='CM1 dry-air mixing ratios',liquid=['qc','qr'],ice=['qi','qs','qg'],actual_times_h=t[idx].tolist(),alpha=alpha.tolist(),max_time_gap_h=float(np.diff(t[idx]).max()),centre_method='reused Gaussian sigma2 psfc minimum, temporally continuous 180km search from simulation start',lowest_scalar_m=float(z[0]),sponge_start='unavailable in output')
  m.savejson(schemafile,schema);return stack,schema

def outer_cycle(label,d,cfg,out):
 ed=d['theta_edges_K'];z=d['z_m'];psi=d['Psi'];bottom=float(-psi[0].min());step=cfg['level_step_kg_s']
 # The outermost line in a resolved discrete scan, not an absolute continuum separatrix.
 start=(np.floor(bottom/step)+1)*step;end=float(-psi.min());trials=[]
 for amp in np.arange(start,end+1,step):
  c,candidates=m.cycle(ed,z,psi,np.ones_like(d['support'],bool),-amp,cfg['near_surface_m'])
  valid=bool(c and c['vertices'][:,1].min()>=z[0]+cfg['bottom_clearance_m'] and cfg['minimum_top_m']<=c['points']['c']['z_m']<=18000 and c['abc_in_flow_order'])
  trials.append(dict(amplitude_kg_s=float(amp),qualified=valid,candidates=candidates))
  if valid:
   v=c['vertices'];iz=np.argmin(abs(v[:,1,None]-z[None,:]),axis=1);jt=np.clip(np.searchsorted(ed,v[:,0],side='right')-1,0,len(ed)-2)
   report=dict(case=label,level_kg_s=float(-amp),bottom_Psi_amplitude=bottom,zmin_m=float(v[:,1].min()),zmax_m=float(v[:,1].max()),theta_min_K=float(v[:,0].min()),theta_max_K=float(v[:,0].max()),closed=True,abc_in_flow_order=c['abc_in_flow_order'],zero_count_nearest_box_vertices=int(np.sum(d['count'][iz,jt]==0)),abc=c['points'],scan_step_kg_s=step,bottom_clearance_m=cfg['bottom_clearance_m'],trials=trials)
   m.savejson(out/f'{label}_outer_cycle.json',report)
   m.savecsv(out/f'{label}_outer_cycle.csv',[dict(point=i,theta_e_K=float(a[0]),z_m=float(a[1])) for i,a in enumerate(v)])
   return c,report
 m.savejson(out/f'{label}_outer_cycle.json',dict(case=label,status='no qualified outer deep closed contour',trials=trials))
 return None,dict(case=label,status='no qualified outer deep closed contour')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--center-h',type=float,default=100.);ap.add_argument('--half-window-h',type=float,default=3.);ap.add_argument('--radius-km',type=float,default=1000.);args=ap.parse_args()
 cfg=dict(center_h=args.center_h,start_h=args.center_h-args.half_window_h,end_h=args.center_h+args.half_window_h,radius_km=args.radius_km,theta_bin_K=1.,level_step_kg_s=1e7,bottom_clearance_m=1.,near_surface_m=1000.,minimum_top_m=10000.,xlim=[380.,320.],ylim=[0.,18.],cases={label:'/data/zhangyx/DATA/'+file for label,file in FILES.items()})
 out=ROOT/f"output/isentropic_jet_strength_{cfg['start_h']:g}_{cfg['end_h']:g}h";out.mkdir(parents=True,exist_ok=True);m.savejson(out/'config.json',cfg)
 results={};schemas={};summaries=[];wrows=[]
 for label,path in cfg['cases'].items():
  h,schema=read_case(label,Path(path),cfg,out);schemas[label]=schema;a=h['alpha'];d={}
  for k in ['M','Fz','Fz_raw','count','domain_mass','net_raw','wbar','gross','mass_error','flux_error']:
   shape=(len(a),)+(1,)*(h[k].ndim-1);d[k]=np.sum(h[k]*a.reshape(shape),axis=0)
  d.update(z_m=h['z_m'],theta_edges_K=h['theta_edges_K']);d['Psi']=np.c_[np.zeros(len(h['z_m'])),np.cumsum(d['Fz']*np.diff(h['theta_edges_K']),axis=1)];d['support']=(d['count']>=1)&(d['M']>0)
  c,selection=outer_cycle(label,d,cfg,out);d['cycle']=c;results[label]=d
  arrays={k:v for k,v in d.items() if isinstance(v,np.ndarray)};np.savez_compressed(out/f'{label}_mean.npz',**arrays,time_h=h['time_h'],alpha=a)
  summary={k:v for k,v in selection.items() if k not in ['trials','abc']};summary.update(Fz_positive_peak_kg_s_K=float(d['Fz'].max()),Fz_negative_peak_kg_s_K=float(d['Fz'].min()),Psi_min_kg_s=float(d['Psi'].min()),mass_recovery_error=float(d['mass_error']),corrected_flux_recovery_error=float(d['flux_error']),Psi_endpoint_max_abs_kg_s=float(abs(d['Psi'][:,-1]).max()))
  summaries.append(summary)
  for t,weight in zip(h['time_h'],a):wrows.append(dict(case=label,time_h=float(t),weight=float(weight)))
  print(label,'selected',selection.get('level_kg_s'),'top',selection.get('zmax_m'),flush=True)
 # Same-grid comparability verified independently; never assume schema identity across cases.
 ref=next(iter(results.values()))
 for d in results.values():
  for k in ['z_m','theta_edges_K']:
   if not np.array_equal(d[k],ref[k]):raise ValueError('Common-grid comparison failed')
 m.savejson(out/'summary.json',summaries);m.savecsv(out/'time_weights.csv',wrows);m.savejson(out/'input_schemas.json',schemas)
 peak=max(float(abs(d['Fz']).max()/1e9) for d in results.values());maximum=max(.1,float(10**np.ceil(np.log10(peak))))
 pos=np.array([.01,.02,.05,.1,.2,.5,1,2,5,10,20,50,100]);pos=np.r_[pos[pos<maximum],maximum];levels=np.r_[-pos[::-1],pos];n=len(pos)
 colors=np.vstack([plt.cm.Purples(np.linspace(.95,.2,n)),[[1,1,1,1]],plt.cm.YlOrRd(np.linspace(.15,.95,n))]);cmap=ListedColormap(colors);cmap.set_under(colors[0]);cmap.set_over(colors[-1]);norm=BoundaryNorm(levels,cmap.N);lp=np.array([.5,1,2,5,10,20]);style=(levels,cmap,norm,np.r_[-lp[::-1],lp])
 ed=ref['theta_edges_K'];z=ref['z_m']
 def overlay(ax,title):
  for label,d in results.items():
   c=d['cycle']
   if c:
    v=c['vertices'];ax.plot(v[:,0],v[:,1]/1000,color=COLORS[label],label=label,lw=1.7)
    for name,p in c['points'].items():
     ax.plot(p['theta_e_K'],p['z_m']/1000,'o',ms=3,color=COLORS[label])
     if label=='CTRL':ax.annotate(name,(p['theta_e_K'],p['z_m']/1000),xytext=(6,8),textcoords='offset points',fontsize=10)
   else:ax.plot([],[],color=COLORS[label],label=label+' (unavailable)')
  ax.set(title=title,xlabel=r'Ice-reference $\theta_e$ (K)',ylabel='Height (km)',xlim=cfg['xlim'],ylim=cfg['ylim']);ax.legend(loc='upper right',fontsize=9,frameon=False);ax.set_yticks(np.arange(0,19,3))
 def save(fig,name):fig.savefig(out/f'{name}.png',dpi=220);fig.savefig(out/f'{name}.pdf');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13,10),constrained_layout=True)
 for i,(label,d) in enumerate(results.items()):
  c=d['cycle'];level=f"$\\Psi={c['level_kg_s']/1e9:.2f}\\times10^9$ kg/s" if c else 'No closed deep contour'
  im=m.panel(axes.flat[i],d,ed,z,style,f'({chr(97+i)}) {label}\n{level}',cfg['xlim']);axes.flat[i].set_yticks(np.arange(0,19,3))
 overlay(axes.flat[5],'(f) Outermost resolved closed contours')
 fig.suptitle(f"Jet-strength sensitivity: {cfg['start_h']:g}–{cfg['end_h']:g} h time mean (center {cfg['center_h']:g} h)",fontsize=14)
 fig.colorbar(im,ax=list(axes.flat[:5]),label=r'$F_z$ ($10^9$ kg s$^{-1}$ K$^{-1}$)',shrink=.8);save(fig,'isentropic_strength_sensitivity')
 fig,ax=plt.subplots(figsize=(6,7),constrained_layout=True);overlay(ax,f"{cfg['start_h']:g}–{cfg['end_h']:g} h: outer closed contours");save(fig,'outer_cycles_comparison')
 m.savejson(out/'plot_config.json',dict(Fz_scale=1e9,Psi_scale=1e9,Fz_levels=levels.tolist(),Psi_levels=style[3].tolist(),colors=COLORS,xlim=cfg['xlim'],ylim=cfg['ylim']))
 m.savejson(out/'provenance.json',dict(script=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,reused=['plot_li2023_isentropic_time_mean.thermo','plot_li2023_isentropic_time_mean.bins','plot_li2023_isentropic_time_mean.cycle','plot_li2023_isentropic_time_mean.panel','diagnose_thompson_intensity_baseline._choose_center']))
 text=['# 急流强度敏感性：等熵窗口平均','',f"主窗口{cfg['start_h']:g}–{cfg['end_h']:g}h，中心{cfg['center_h']:g}h，前后各{args.half_window_h:g}h。只使用指定CTRL、JET15、标准无后缀JET30、JET45、JET60。",'',
 '## 方法和选线','',
 '- 复用前次冰参考theta_e、原始三维分箱/质量通量、流函数和轮廓代码。先逐时非线性转换与分箱，后平均通量；不做集合/预先方位平均。',
 '- 本次重新对五份输入检查维度、单位、干空气密度EOS和水物质；水汽qv，液态qc+qr，固态qi+qs+qg，排除数浓度。',
 '- 每时刻完整台风中心1000km圆域，采用真实非均匀xf/yf面积；w从实际zf线性插值至zh，zhval核对；M kg/(m K)，Fz kg/(s K)，Psi kg/s，不乘Delta z。',
 '- 用原中心函数，从0h逐时连续跟踪Gaussian sigma2平滑psfc最低点，180km局地搜索；centres.csv保留窗口中心和实际中心气压。',
 '- 实际时间/权重见time_weights.csv；等间隔完整端点等权，否则使用原有梯形权重逻辑。',
 '- 各试验独立找最外围可分辨闭合深胞：从刚超过底层最大负Psi幅度的1e7网格水平开始，向内逐级扫描，取第一条闭合且底部离25m等最低层至少1m、最低<=1km、顶部10–18km且abc方向正确的轮廓；同级选最深，再最大面积。',
 '- “最外层”是当前网格、1e7扫描步长和1m下边界裕量下的最外围可用深循环，不是连续场中唯一严格外边界。其他浅/高层局部循环不作为目标。',
 '- 各试验选线水平可能不同：子图f比较外围形态，不比较同等Psi输送量；每幅图列出自己的水平，不能据此把线围面积解释成输送强度。',
 '- Psi在有限累积场上提取，不用局地count掩膜截断；Fz填色仍用count>=1且M>0掩膜。最近箱覆盖单独报告，不能替代严格沿线状态支持验证。',
 '- 没有添加虚构z=0边界；a最低theta_e，b<=1km最高theta_e，c最高高度。数学闭合不是实际轨迹/控制体/能量闭合。',
 '- 所有图θ轴380→320K、高度0–18km；分箱200–700K保存全部59层。真实模型顶和海绵层限制见schema；薄线级别为实现选择。','',
 '## 检查与结果','', '|Case|Psi level (kg/s)|Bottom m|Top km|mass error|Psi endpoint kg/s|','|---|---:|---:|---:|---:|---:|']
 for r in summaries:
  if 'level_kg_s' in r:text.append(f"|{r['case']}|{r['level_kg_s']:.4e}|{r['zmin_m']:.3f}|{r['zmax_m']/1000:.3f}|{r['mass_recovery_error']:.2e}|{r['Psi_endpoint_max_abs_kg_s']:.2e}|")
  else:text.append(f"|{r['case']}|unavailable|||||")
 text+=['','各时刻质量恢复、原始/修正通量恢复、EOS和高度映射检查均在checks.csv。原始net_raw/wbar与Fz_raw保存于缓存。M_t仅提示非稳态，没有独立Ftheta，不能宣称完整连续方程闭合。均值扣除不证明侧向交换消失；同100h阶段不一定强度匹配，JET幅度因果机制不能由这些结构图单独证明。','',
 '## 交付','', '- isentropic_strength_sensitivity.png/pdf：五试验加轮廓叠加的2×3图；outer_cycles_comparison.png/pdf：独立叠加图，五色及图例。',
 '- *_hourly_bins.npz、*_mean.npz：分箱与逐时M_t/实际时间/权重；*_outer_cycle.csv/json：轮廓、abc、候选与选线审计。',
 '- *_centres.csv、*_checks.csv、*_schema.json、summary.json、config.json、plot_config.json、provenance.json。','',
 '```sh','cd /data1/home/zhangyx/project/TC_dynamic',f'/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python scripts/plot_isentropic_jet_strength_sensitivity.py --center-h {args.center_h:g} --half-window-h {args.half_window_h:g}', '```','',
 '复跑若输入身份与配置一致，复用逐时分箱缓存；用户指定时间窗改变时写入新输出目录。','',
 '参考定义：Li et al.(2023) Eq7–8、Appendix A4：https://doi.org/10.1175/JAS-D-22-0186.1；选线为本次自定义，非Li固定1e7复现。','']
 (out/'README.md').write_text('\n'.join(text),encoding='utf-8');print('Completed',out,flush=True)
if __name__=='__main__':main()
