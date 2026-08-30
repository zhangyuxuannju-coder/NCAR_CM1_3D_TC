#!/usr/bin/env python3
"""Screen CTRL/JET times for a common raw-elliptic SE analysis domain."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src._se_pipeline_single import PipelineConfig,azimuthal_average_from_3d
from src.se_applicability import compute_case_stability

def parse():
 p=argparse.ArgumentParser(); p.add_argument('--nojet',required=True);p.add_argument('--jet',required=True);p.add_argument('--output-dir',required=True)
 p.add_argument('--hours',type=float,nargs='+',default=[20,25,30,35,40,45,50,55,60,65,70])
 p.add_argument('--r-min',type=float,default=50);p.add_argument('--r-max',type=float,default=300);p.add_argument('--z-min',type=float,default=2);p.add_argument('--z-max',type=float,default=18)
 p.add_argument('--dr-km',type=float,default=12);p.add_argument('--f',type=float,default=5.464e-5);return p.parse_args()

def one(path,h,a):
 cfg=PipelineConfig(input_file=path,output_dir=a.output_dir,target_time_hours=h,max_r_km=a.r_max,dr_km=a.dr_km,max_z_km=20,coriolis_f=a.f,include_model_budget_terms=False,write_netcdf=False,write_ieee=False,plot_solution=False)
 avg=azimuthal_average_from_3d(cfg); r=np.asarray(avg['r_km']);z=np.asarray(avg['z_km'])
 f,_=compute_case_stability(avg['ut'],avg['theta'],avg['rho'],r*1000,z*1000,a.f)
 return avg,f,r,z

def main():
 a=parse();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True); rows=[]; cache={}
 for h in a.hours:
  cases={}
  for name,path in [('noJET',a.nojet),('JET',a.jet)]:
   avg,f,r,z=one(path,h,a);cases[name]=(avg,f);rr,zz=np.meshgrid(r,z);m=(rr>=a.r_min)&(rr<=a.r_max)&(zz>=a.z_min)&(zz<=a.z_max)
   cls=np.asarray(f['stability_class']); rows.append(dict(hour=h,case=name,elliptic_fraction=float(np.mean(cls[m]==0)),inertial_fraction=float(np.mean(cls[m]==1)),symmetric_fraction=float(np.mean(cls[m]==2)),static_fraction=float(np.mean(cls[m]==3)),D_min=float(np.nanmin(np.asarray(f['D_raw'])[m])),I2_min=float(np.nanmin(np.asarray(f['I2_raw'])[m]))))
  cache[h]=(cases,r,z,m)
  common=(cases['noJET'][1]['stability_class']==0)&(cases['JET'][1]['stability_class']==0)&m
  rows.append(dict(hour=h,case='COMMON',elliptic_fraction=float(np.sum(common)/np.sum(m)),inertial_fraction=np.nan,symmetric_fraction=np.nan,static_fraction=np.nan,D_min=np.nan,I2_min=np.nan))
 best=max(a.hours,key=lambda h:next(x['elliptic_fraction'] for x in rows if x['hour']==h and x['case']=='COMMON'))
 with (out/'ellipticity_screen.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(1,2,figsize=(10.8,4),constrained_layout=True);color={'noJET':'#3C5488','JET':'#B9363E','COMMON':'#009E73'}
 for c in color:
  q=[x for x in rows if x['case']==c];ax[0].plot([x['hour'] for x in q],[x['elliptic_fraction'] for x in q],'o-',color=color[c],label=c)
 ax[0].set(xlabel='Time (h)',ylabel='Elliptic grid fraction',ylim=(0,1.03),title=f'Raw ellipticity in r={a.r_min:g}–{a.r_max:g} km, z={a.z_min:g}–{a.z_max:g} km');ax[0].legend(frameon=False);ax[0].grid(alpha=.25)
 for c in ('noJET','JET'):
  q=[x for x in rows if x['case']==c];ax[1].plot([x['hour'] for x in q],[x['inertial_fraction'] for x in q],'o-',color=color[c],label=f'{c}: inertial');ax[1].plot([x['hour'] for x in q],[x['symmetric_fraction'] for x in q],'s--',color=color[c],alpha=.7,label=f'{c}: shear/symmetric')
 ax[1].set(xlabel='Time (h)',ylabel='Grid fraction',ylim=(0,1.03),title='Reasons for non-ellipticity');ax[1].legend(frameon=False,fontsize=8);ax[1].grid(alpha=.25)
 fig.savefig(out/'figure1_common_ellipticity_screen.png',dpi=260,bbox_inches='tight');plt.close(fig)
 cases,r,z,m=cache[best]; np.savez_compressed(out/'best_time_operator_fields.npz',hour=best,r_km=r,z_km=z,mask=m,**{f'{k}_{c}':np.asarray(v) for c in cases for k,v in cases[c][1].items() if isinstance(v,np.ndarray)})
 summary={'best_common_hour':float(best),'domain':dict(r_min=a.r_min,r_max=a.r_max,z_min=a.z_min,z_max=a.z_max),'rows':rows}
 (out/'ellipticity_screen_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
