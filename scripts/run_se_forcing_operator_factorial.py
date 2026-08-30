#!/usr/bin/env python3
"""Four-cell forcing/operator SE attribution on a fully elliptic subdomain."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src._se_pipeline_single import (PipelineConfig,azimuthal_average_from_3d,
    _to_solver_layout_zr_to_rz,solve_se_sor)
from src.se_bui import invert_balanced_theta,build_basic_state,assemble_operator,build_forcing

def parse():
 p=argparse.ArgumentParser();p.add_argument('--nojet',required=True);p.add_argument('--jet',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--hour',type=float,default=20)
 p.add_argument('--r-min',type=float,default=126);p.add_argument('--r-max',type=float,default=294);p.add_argument('--z-min',type=float,default=1);p.add_argument('--z-max',type=float,default=18);p.add_argument('--dr-km',type=float,default=12);p.add_argument('--f',type=float,default=5.464e-5);return p.parse_args()

def avg(path,a):
 cfg=PipelineConfig(input_file=path,output_dir=a.output_dir,target_time_hours=a.hour,max_r_km=300,dr_km=a.dr_km,max_z_km=20,coriolis_f=a.f,include_model_budget_terms=True,write_netcdf=False,write_ieee=False,plot_solution=False,sor_max_iter=60000,sor_omega=1.5,sor_tol=1.0e-16,sor_verbose_every=5000)
 return cfg,azimuthal_average_from_3d(cfg)

def subset_basic(b,iz,ir):
 out={}
 for k,v in b.items():
  out[k]=np.asarray(v)[np.ix_(iz,ir)] if isinstance(v,np.ndarray) and np.asarray(v).ndim==2 else v
 return out

def solve_converged(op,forcing,r,z):
 a={k:_to_solver_layout_zr_to_rz(op[k]) for k in ('A','B','C','D','E')}
 p=solve_se_sor(A=a['A'],B=a['B'],C=a['C'],D=a['D'],E=a['E'],
     F=_to_solver_layout_zr_to_rz(forcing),dr=float(np.mean(np.diff(r))),dz=float(np.mean(np.diff(z))),
     max_iter=80000,omega=1.5,tol=1.0e-14,verbose_every=10000)
 return p[:,1:-1].T

def main():
 a=parse();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 cfg,c=avg(a.nojet,a);_,j=avg(a.jet,a);r=np.asarray(c['r_km']);z=np.asarray(c['z_km']);ir=np.flatnonzero((r>=a.r_min)&(r<=a.r_max));iz=np.flatnonzero((z>=a.z_min)&(z<=a.z_max));rs=r[ir]*1000;zs=z[iz]*1000
 cases={}
 for name,x in [('CTRL',c),('JET',j)]:
  theta,tw=invert_balanced_theta(x['ut'],x['theta'],r*1000,z*1000,a.f)
  full=build_basic_state(x['ut'],theta,x['rho'],r*1000,z*1000,a.f,baroclinic_scale=1.)
  b=subset_basic(full,iz,ir);k1=b['K1_raw'];k2=b['K2_raw'];k3=b['K3_raw'];disc=k1*k3-k2*k2
  if np.any(~np.isfinite(disc)) or np.any(disc<=0):raise RuntimeError(f'{name} subdomain is not fully elliptic: bad={np.count_nonzero(disc<=0)}, minD={np.nanmin(disc):.3e}')
  cases[name]=dict(x=x,b=b,op=assemble_operator(b,k1,k2,k3,rs,zs),tw=tw,D=disc,K1=k1,K2=k2,K3=k3)
 sources={'CTRL':(c['Q'][np.ix_(iz,ir)],c['Fnu'][np.ix_(iz,ir)]),'JET':(j['Q'][np.ix_(iz,ir)],j['Fnu'][np.ix_(iz,ir)])}
 sol={};rhs={}
 for opn in ('CTRL','JET'):
  for src in ('CTRL','JET'):
   q,f=sources[src];rr=build_forcing(cases[opn]['b'],q,f,rs,zs);key=opn[0]+src[0];rhs[key]=rr['forcing_total'];sol[key]=solve_converged(cases[opn]['op'],rr['forcing_total'],rs,zs)
 p00,p01,p10,p11=sol['CC'],sol['CJ'],sol['JC'],sol['JJ'];effectF=p01-p00;effectL=p10-p00;interaction=p11-p10-p01+p00;total=p11-p00
 fields=[p00,p01,p10,p11,effectF,effectL,interaction,total];titles=['CTRL op / CTRL forcing','CTRL op / JET forcing','JET op / CTRL forcing','JET op / JET forcing','forcing effect','operator effect','interaction','total balanced difference']
 cm=LinearSegmentedColormap.from_list('nature',['#3C5488','#A8BFDC','#F7F7F4','#F1B39A','#B9363E']);v=np.nanpercentile(abs(np.concatenate([x.ravel() for x in fields]))*2*np.pi/1e9,98)
 fig,ax=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True,constrained_layout=True)
 for q,(x,t) in enumerate(zip(fields,titles)):
  im=ax.flat[q].pcolormesh(r[ir],z[iz],x*2*np.pi/1e9,cmap=cm,norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v),shading='auto');ax.flat[q].set_title(t);ax.flat[q].set_xlabel('Radius (km)');ax.flat[q].set_ylabel('Height (km)')
 fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.65,pad=.05,label=r'$2\pi\psi$ ($10^9$ kg s$^{-1}$)');fig.suptitle(f'Unregularized four-cell SE attribution at {a.hour:g} h\nfully elliptic balanced-projection subdomain',fontweight='bold');fig.savefig(out/'figure1_se_forcing_operator_factorial.png',dpi=260,bbox_inches='tight');plt.close(fig)
 def norm(x):return float(np.sqrt(np.nanmean(x*x)))
 summary={'hour':a.hour,'subdomain':dict(r_min=float(r[ir][0]),r_max=float(r[ir][-1]),z_min=float(z[iz][0]),z_max=float(z[iz][-1])),'min_D':{k:float(np.nanmin(cases[k]['D'])) for k in cases},'thermal_wind':{k:cases[k]['tw'] for k in cases},'rms_psi':{'forcing':norm(effectF),'operator':norm(effectL),'interaction':norm(interaction),'total':norm(total)},'fractional_rms':{'forcing':norm(effectF)/max(norm(total),1e-30),'operator':norm(effectL)/max(norm(total),1e-30),'interaction':norm(interaction)/max(norm(total),1e-30)}}
 np.savez_compressed(out/'se_factorial_products.npz',r_km=r[ir],z_km=z[iz],psi00=p00,psi01=p01,psi10=p10,psi11=p11,forcing_effect=effectF,operator_effect=effectL,interaction=interaction,total=total,**{f'{f}_{k}':cases[k][f] for k in cases for f in ('K1','K2','K3','D')})
 (out/'se_factorial_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
