#!/usr/bin/env python3
"""Li et al. (2023) MAFALDA energetics: 75 h validation only.

This independent entry point deliberately stops before the full time series.
It reuses the validated ice-reference thermodynamics, binning, centre tracking,
and contour routines without changing project core code.
"""
from pathlib import Path
import argparse,csv,json,sys,hashlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import plot_li2023_isentropic_time_mean as base
C=base.C; G=9.80665
CASES={'CTRL':'/data/zhangyx/DATA/cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc','JET30':'/data/zhangyx/DATA/cm1out_25N_JET_R2_Thompson_OML100_240h.nc'}
VARS=['T','s','p','alpha_d','rv','rl','ri','rT','gv','gl','gi','K','Phi']

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def centres():
 p=ROOT/'output/thompson_240h_se_attribution/baseline/hourly_intensity_baseline.csv'
 with p.open() as f:rows=list(csv.DictReader(f))
 return {(r['case'],int(r['time_index'])):(float(r['center_x_km'])*1000,float(r['center_y_km'])*1000) for r in rows}
def state(th,p,rho,rv,rl,ri,u,v,w,z):
 theta,T=base.thermo(th,p,rv,rl,ri); Tf=C['Tf'];lv=C['Lv0']+(C['Cpv']-C['Cl'])*(T-Tf)
 H=p*rv/(C['Rd']/C['Rv']+rv)/base.saturation_vapor_pressure_pa(T)
 lnH=np.zeros_like(rv);q=rv>0;lnH[q]=np.log(H[q])
 sd=C['Cpd']*np.log(T/Tf)-C['Rd']*np.log(p/C['p0'])
 sv=C['Cl']*np.log(T/Tf)+lv/T-C['Rv']*lnH
 sl=C['Cl']*np.log(T/Tf);si=C['Ci']*np.log(T/Tf)-C['Lf']/Tf
 s=sd+rv*sv+rl*sl+ri*si
 core=T-Tf-T*np.log(T/Tf)
 gv=C['Cl']*core+C['Rv']*T*lnH;gl=C['Cl']*core;gi=C['Ci']*core-C['Lf']*(1-T/Tf)
 rT=rv+rl+ri
 return theta,dict(T=T,s=s,p=p,alpha_d=1/rho,rv=rv,rl=rl,ri=ri,rT=rT,gv=gv,gl=gl,gi=gi,K=.5*(u*u+v*v+w*w),Phi=np.broadcast_to(G*z[:,None,None],T.shape))
def bin_all(theta,rho,w,area,mask,edges,fields):
 d=base.bins(theta,rho,w,area,mask,edges);nb=len(edges)-1;nz=len(theta)
 for name in VARS:
  num=np.zeros((nz,nb));a=fields[name]
  for k in range(nz):
   good=mask&np.isfinite(theta[k])&np.isfinite(a[k]);idx=np.digitize(theta[k][good],edges)-1;wt=rho[k][good]*area[good]
   ok=(idx>=0)&(idx<nb);num[k]=np.bincount(idx[ok],weights=wt[ok]*a[k][good][ok],minlength=nb)
  d[name+'_num']=num
 return d
def read_case(label,path,out,cfg):
 cache=out/(label+'_hourly_energy_bins.npz');key='JET' if label=='JET30' else label;cmap=centres()
 with Dataset(path) as ds:
  t=np.asarray(ds['time'][:])/3600;center=float(cfg['validation_center_h']);half=float(cfg['validation_half_window_h']);start=center-half;end=center+half;ids=np.flatnonzero((t>=start-1e-6)&(t<=end+1e-6));alpha=np.ones(len(ids))/len(ids)
  x=np.asarray(ds['xh'][:])*1000;y=np.asarray(ds['yh'][:])*1000;xf=np.asarray(ds['xf'][:])*1000;yf=np.asarray(ds['yf'][:])*1000;z=np.asarray(ds['zh'][:])*1000;zf=np.asarray(ds['zf'][:])*1000
  edges=np.arange(cfg['theta_min_K'],cfg['theta_max_K']+cfg['theta_bin_K']/2,cfg['theta_bin_K']);samples=[];checks=[]
  for it in ids:
   cx,cy=cmap[(key,int(it))];R=cfg['radius_km']*1000;margin=min(cx-xf[0],xf[-1]-cx,cy-yf[0],yf[-1]-cy)
   if margin<R:raise ValueError('incomplete disk')
   ix=np.flatnonzero(abs(x-cx)<=R);iy=np.flatnonzero(abs(y-cy)<=R);sx=slice(ix[0],ix[-1]+1);sy=slice(iy[0],iy[-1]+1)
   def f(n):return np.ma.filled(ds[n][it,:,sy,sx],np.nan).astype(float)
   xx,yy=np.meshgrid(x[sx]-cx,y[sy]-cy);mask=np.hypot(xx,yy)<=R;area=np.diff(yf)[sy,None]*np.diff(xf)[None,sx]
   th=f('th');p=f('prs');rho=f('rho');rv=np.maximum(f('qv'),0);qc,qr,qi,qs,qg=[np.maximum(f(n),0) for n in ['qc','qr','qi','qs','qg']]
   wf=f('w');frac=(z-zf[:-1])/np.diff(zf);w=wf[:-1]*(1-frac[:,None,None])+wf[1:]*frac[:,None,None]
   # Linear face-to-centre interpolation on the actual staggered axes.
   uraw=np.ma.filled(ds['u'][it,:,sy,ix[0]:ix[-1]+2],np.nan).astype(float);u=.5*(uraw[:,:,:-1]+uraw[:,:,1:])
   vraw=np.ma.filled(ds['v'][it,:,iy[0]:iy[-1]+2,sx],np.nan).astype(float);v=.5*(vraw[:,:-1]+vraw[:,1:])
   theta,fields=state(th,p,rho,rv,qc+qr,qi+qs+qg,u,v,w,z)
   eos=p/((C['Rd']+C['Rv']*rv)*fields['T']);ee=float(np.max(abs(eos-rho)/eos));he=float(np.max(abs(f('zhval')-z[:,None,None])))
   if ee>1e-4 or he>.02:raise ValueError('EOS/height validation failed')
   d=bin_all(theta,rho,w,area,mask,edges,fields);samples.append(d)
   checks.append(dict(case=label,time_h=float(t[it]),radius_margin_km=float(margin/1000-R/1000),EOS_error=ee,height_error_m=he,mass_error=float(d['mass_error']),flux_error=float(d['flux_error'])))
  keys=samples[0].keys();stack={k:np.stack([q[k] for q in samples]) for k in keys};np.savez_compressed(cache,**stack,time_h=t[ids],alpha=alpha,z_m=z,theta_edges_K=edges)
  return stack,t[ids],alpha,z,edges,checks
def interp_strict(z,tc,field,path):
 # Bilinear interpolation using only nonzero-weight corners. scipy's linear
 # interpolator can propagate NaN from a zero-weight neighbor at an exact grid
 # coordinate (0*NaN), incorrectly marking a supported contour point invalid.
 z=np.asarray(z,float);tc=np.asarray(tc,float);field=np.asarray(field,float);out=np.full(len(path),np.nan)
 def bracket(grid,x):
  k=int(np.searchsorted(grid,x))
  if k<len(grid) and np.isclose(x,grid[k],rtol=0,atol=1e-6):return [(k,1.)]
  if k>0 and np.isclose(x,grid[k-1],rtol=0,atol=1e-6):return [(k-1,1.)]
  if k==0 or k==len(grid):return []
  f=(x-grid[k-1])/(grid[k]-grid[k-1]);return [(k-1,1-f),(k,f)]
 for n,(th,zz) in enumerate(path):
  zi=bracket(z,zz);ti=bracket(tc,th);terms=[(wz*wt,field[iz,jt]) for iz,wz in zi for jt,wt in ti if wz*wt>1e-14]
  if terms and all(np.isfinite(v) for w,v in terms):out[n]=sum(w*v for w,v in terms)
 return out
def integrate(path,st,abc):
 # Path is explicitly closed and ordered a -> b -> c -> a.
 if not np.allclose(path[0],path[-1]):raise ValueError('open path')
 if any(np.any(~np.isfinite(st[k])) for k in VARS):raise ValueError('unsupported state on path')
 mid=lambda x:.5*(x[:-1]+x[1:]);diff=lambda x:np.diff(x)
 Wmax=float(np.sum(mid(st['T'])*diff(st['s'])));WP=float(np.sum(G*mid(st['rT'])*diff(path[:,1])))
 GP=float(-np.sum(mid(st['gv'])*diff(st['rv'])+mid(st['gl'])*diff(st['rl'])+mid(st['gi'])*diff(st['ri'])))
 pd=float(np.sum(mid(st['alpha_d'])*diff(st['p'])));WKE=-pd-WP;res=Wmax-WKE-WP-GP;den=abs(Wmax)+abs(WKE)+abs(WP)+abs(GP)
 seg=-(diff(st['K'])+diff(st['Phi'])+mid(st['alpha_d'])*diff(st['p'])+G*mid(st['rT'])*diff(path[:,1]))
 ib=int(abc['b']['index']);ic=int(abc['c']['index']);wab=float(np.sum(seg[:ic]));wca=float(np.sum(seg[ic:]));whole=float(np.sum(seg))
 return dict(Wmax=Wmax,WKE=WKE,WP=WP,GP=GP,R=res,R_rel=abs(res)/den if den else np.nan,W_ab_bc=wab,W_ca=wca,W_segment_sum=wab+wca,W_segment_whole=whole,segment_closure=wab+wca-whole)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'config/isentropic_energy_li_fig7_8.json');args=ap.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
 allchecks=[];window={};
 for label,path in CASES.items():
  stack,t,a,z,edges,checks=read_case(label,path,out,cfg);allchecks+=checks;d={}
  for k,v in stack.items():d[k]=np.sum(v*a.reshape((len(a),)+(1,)*(v.ndim-1)),axis=0)
  d['Psi']=np.c_[np.zeros(len(z)),np.cumsum(d['Fz']*np.diff(edges),axis=1)];d['support']=(d['count']>=1)&(d['M']>0)
  for name in VARS:d[name]=np.divide(d[name+'_num'],d['M']*np.diff(edges)[None,:],out=np.full_like(d[name+'_num'],np.nan),where=d['M']>0)
  window[label]=(d,z,edges)
 # Audit Li level, then use the already-public common 1.3e9 validation level only if both pass.
 audit={};cycles={};chosen=None
 for lev in cfg['cycle_candidates_kg_s']:
  ok=True;audit[str(lev)]={}
  for label,(d,z,edges) in window.items():
   c,cands=base.cycle(edges,z,d['Psi'],np.ones_like(d['support'],bool),lev,cfg['near_surface_m']);valid=bool(c and c['abc_in_flow_order'] and c['vertices'][:,1].min()>=z[0]+1 and c['vertices'][:,1].max()<z[-1])
   audit[str(lev)][label]=dict(valid_geometry=valid,candidates=cands)
   if not valid:ok=False
   if valid:cycles[label]=c
  if ok:chosen=lev;break
 if chosen is None:raise ValueError('no common geometrically valid fixed level')
 results={};paths={}
 for label,(d,z,edges) in window.items():
  c,_=base.cycle(edges,z,d['Psi'],np.ones_like(d['support'],bool),chosen,cfg['near_surface_m']);path=c['vertices'];tc=.5*(edges[:-1]+edges[1:]);st={n:interp_strict(z,tc,d[n],path) for n in VARS}
  supported={n:int(np.count_nonzero(np.isfinite(v))) for n,v in st.items()};valid=all(q==len(path) for q in supported.values())
  if not valid:results[label]=dict(valid=False,reason='conditional state missing on contour',path_points=len(path),finite_points=supported);continue
  energy=integrate(path,st,c['points']);energy.update(valid=True,cycle_level_kg_s=chosen,path_points=len(path),abc=c['points']);results[label]=energy;paths[label]=(path,st,c)
 dump(out/'cycle_level_audit.json',audit);dump(out/'validation_75h_results.json',results);dump(out/'config.json',cfg)
 with (out/'sample_checks_75h.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=allchecks[0]);w.writeheader();w.writerows(allchecks)
 # Validation figure is produced even when energy is invalid; missing support is visible and no values are fabricated.
 fig,ax=plt.subplots(2,2,figsize=(13,10),layout='constrained')
 for label,color in [('CTRL','red'),('JET30','blue')]:
  d,z,edges=window[label];ax[0,0].contourf(.5*(edges[:-1]+edges[1:]),z/1000,d['Fz']/1e9,levels=21,cmap='RdBu_r');c,_=base.cycle(edges,z,d['Psi'],np.ones_like(d['support'],bool),chosen,cfg['near_surface_m']);v=c['vertices'];ax[0,0].plot(v[:,0],v[:,1]/1000,color=color,label=label)
  tc=.5*(edges[:-1]+edges[1:]);st={n:interp_strict(z,tc,d[n],v) for n in VARS};q=np.arange(len(v));
  for n,a0 in zip(['T','s','p','rT'],ax.flat[1:]):a0.plot(q,st[n],color=color,label=label)
 ax[0,0].set(xlim=(380,320),ylim=(0,18),xlabel='Ice-reference theta_e (K)',ylabel='Height (km)',title=f'75±3 h; Psi={chosen:.2e} kg/s');ax[0,0].legend()
 for n,a0 in zip(['T','s','p','rT'],ax.flat[1:]):a0.set(title=n,xlabel='ordered path vertex');a0.legend()
 fig.savefig(out/'validation_75h.png',dpi=180);fig.savefig(out/'validation_75h.pdf');plt.close(fig)
 dump(out/'provenance.json',dict(script=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reused=['plot_li2023_isentropic_time_mean.thermo','bins','cycle'],note='no missing-state interpolation'))
 print(json.dumps(results,indent=2));print('Completed',out)
if __name__=='__main__':main()
