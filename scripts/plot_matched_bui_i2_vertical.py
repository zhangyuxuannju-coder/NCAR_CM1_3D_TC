#!/usr/bin/env python3
"""Matched-pair vertical Bui I2 panels using the existing attribution code."""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
sys.path.insert(0,'.')
from scripts.analyze_inertial_stability_attribution import analyze_case, read_pmin, nature_diverging, style

def main():
 p=argparse.ArgumentParser(); p.add_argument('--matches-json',required=True); p.add_argument('--ctrl',required=True); p.add_argument('--jet',required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--label',required=True); a=p.parse_args()
 pairs=json.loads(Path(a.matches_json).read_text()); jh=np.array([x['jet_h'] for x in pairs]); ch=np.array([x['ctrl_h'] for x in pairs])
 cfg=SimpleNamespace(max_r_km=600.,dr_km=12.,max_z_km=20.,f=6.1636e-5,eddy_average='reynolds',radial_smooth_sigma=1.,center_window=21,center_method='min',output_dir=str(a.output_dir))
 a.output_dir.mkdir(parents=True,exist_ok=True)
 ctrl=analyze_case(a.ctrl,ch,read_pmin(a.ctrl,ch),cfg,'CTRL')
 jet=analyze_case(a.jet,jh,read_pmin(a.jet,jh),cfg,'JET')
 r,z=ctrl[0]['r_km'],ctrl[0]['z_km']; fields=[(j['I2'],c['I2'],j['I2']-c['I2']) for j,c in zip(jet,ctrl)]
 outmask=(r[None,:]>=50)&(r[None,:]<=500)&(z[:,None]>=8)&(z[:,None]<=18)
 pos=np.concatenate([abs(q[np.isfinite(q)&outmask]) for trio in fields for q in trio[:2]]); d=np.concatenate([abs(trio[2][np.isfinite(trio[2])&outmask]) for trio in fields]); vmax=np.percentile(pos,99); dmax=np.percentile(d,99)
 style(); fig,ax=plt.subplots(3,len(pairs),figsize=(3.0*len(pairs),8.2),sharex=True,sharey=True,constrained_layout=True)
 for k,(pair,(jj,cc,dd)) in enumerate(zip(pairs,fields)):
  for row,(q,name,lim) in enumerate([(jj,'JET raw Bui $I^2$',vmax),(cc,'CTRL raw Bui $I^2$',vmax),(dd,r'JET-CTRL $\Delta I^2$',dmax)]):
   im=ax[row,k].pcolormesh(r,z,q*1e12,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim*1e12,vcenter=0,vmax=lim*1e12),shading='auto')
   ax[row,k].set_xlim(0,600); ax[row,k].set_ylim(0,20)
   if k==0: ax[row,k].set_ylabel(name+'\nHeight (km)')
   if row==0: ax[row,k].set_title(f"J{pair['jet_h']:.0f}/C{pair['ctrl_h']:.0f}")
  ax[2,k].set_xlabel('Radius (km)')
 fig.colorbar(ax[0,0].collections[0],ax=ax[:2,:],pad=.01,label=r'raw Bui $I^2$ ($10^{-12}$)'); fig.colorbar(ax[2,0].collections[0],ax=ax[2,:],pad=.01,label=r'$\Delta I^2$ ($10^{-12}$)')
 fig.suptitle(a.label+' | azimuthal-mean raw Bui inertial stability (outflow-scaled colours)',fontweight='bold'); fig.savefig(a.output_dir/'vertical_raw_Bui_I2_JET_CTRL_difference.png',dpi=220,bbox_inches='tight'); plt.close(fig)
 (a.output_dir/'vertical_raw_Bui_I2_metadata.json').write_text(json.dumps({'definition':'existing analyze_inertial_stability_attribution.py: compute_case_stability I2_raw; unregularized','matches':pairs},indent=2))
if __name__=='__main__': main()
