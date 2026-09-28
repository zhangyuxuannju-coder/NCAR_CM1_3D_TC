#!/usr/bin/env python3
"""Matched-time PV panels reusing plot_pv_streamlines_comparison.diagnose_panel."""
import argparse,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0,'.')
from scripts.plot_pv_streamlines_comparison import diagnose_panel

def main():
 p=argparse.ArgumentParser(); p.add_argument('--matches-json',required=True); p.add_argument('--ctrl',required=True); p.add_argument('--jet',required=True); p.add_argument('--output',required=True); p.add_argument('--height-km',type=float,default=15.); p.add_argument('--half-width-km',type=float,default=500.); p.add_argument('--pv-limit',type=float,default=10.); a=p.parse_args()
 pairs=json.loads(Path(a.matches_json).read_text()); panels=[]
 for q in pairs:
  for name,path,h in [('CTRL',a.ctrl,q['ctrl_h']),('JET',a.jet,q['jet_h'])]:
   x=diagnose_panel(path,h,a.height_km,a.half_width_km,2.,.65); x.update(case=name,label=f"J{q['jet_h']:.0f}/C{q['ctrl_h']:.0f}"); panels.append(x)
 fig,axes=plt.subplots(len(pairs),2,figsize=(8.8,4.15*len(pairs)),sharex=True,sharey=True,constrained_layout=True); axes=np.atleast_2d(axes)
 for ax,x in zip(axes.flat,panels):
  xx,yy=np.meshgrid(x['x_km'],x['y_km']); ax.pcolormesh(x['x_km'],x['y_km'],x['pv_pvu'],cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-a.pv_limit,vcenter=0,vmax=a.pv_limit),shading='auto')
  # Existing script's streamplot needs a uniform grid.
  xs=np.linspace(x['x_km'][0],x['x_km'][-1],len(x['x_km'])); ys=np.linspace(x['y_km'][0],x['y_km'][-1],len(x['y_km'])); X,Y=np.meshgrid(xs,ys); pts=np.c_[Y.ravel(),X.ravel()]
  u=RegularGridInterpolator((x['y_km'],x['x_km']),x['u_m_s'],bounds_error=False,fill_value=np.nan)(pts).reshape(X.shape); v=RegularGridInterpolator((x['y_km'],x['x_km']),x['v_m_s'],bounds_error=False,fill_value=np.nan)(pts).reshape(X.shape); bad=np.hypot(X,Y)>a.half_width_km; u[bad]=np.nan;v[bad]=np.nan
  ax.streamplot(xs,ys,u,v,density=.8,color='#252a32',linewidth=.65,arrowsize=.75); ax.add_patch(plt.Circle((0,0),200,fill=False,color='0.4',ls='--',lw=.5)); ax.scatter(0,0,marker='x',c='#9D2933',s=28); ax.set_title(f"{x['label']} | {x['case']} {x['selected_hour']:.0f} h",fontsize=10,fontweight='bold'); ax.set_aspect('equal'); ax.set_xlim(-a.half_width_km,a.half_width_km);ax.set_ylim(-a.half_width_km,a.half_width_km)
 for ax in axes[-1]:ax.set_xlabel('Storm-relative x (km)')
 for ax in axes[:,0]:ax.set_ylabel('Storm-relative y (km)')
 cb=fig.colorbar(axes[0,0].collections[0],ax=axes,orientation='horizontal',shrink=.7,pad=.02);cb.set_label('Dry Ertel PV (PVU)')
 fig.suptitle(f'Strength/RMW-matched dry Ertel PV and horizontal flow, z={panels[0]["selected_height_km"]:.2f} km',fontweight='bold');out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=220,bbox_inches='tight')
 Path(str(out)+'.json').write_text(json.dumps({'definition':'dry Ertel PV from existing plot_pv_streamlines_comparison.py','matches':pairs,'height_km':panels[0]['selected_height_km']},indent=2))
if __name__=='__main__':main()
