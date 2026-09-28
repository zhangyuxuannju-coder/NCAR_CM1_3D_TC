#!/usr/bin/env python3
"""Physical I2 comparison for intensity/RMW-matched CM1 pairs.

I2=(f+2*vtheta/r)*(f+zeta).  Vertical fields use azimuthal vtheta;
horizontal fields use native u/v on TC-centred Cartesian planes.
"""
import argparse, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
from scripts.solve_matched_operator_forcing_outer100 import read_case

F=6.1636e-5

class A:
    max_r_km=600.; dr_km=12.; max_z_km=20.; f=F; output_dir="."

def i2_axisym(avg):
    r=np.asarray(avg['r_km'],float)*1000.; vt=np.asarray(avg['ut'],float)
    rs=np.maximum(r, max(np.diff(r).min()/2,1.))
    zeta=np.gradient(vt*rs[None,:],r,axis=1,edge_order=2)/rs[None,:]
    return (F+2*vt/rs[None,:])*(F+zeta)

def axisym(path,hour):
    x=read_case(path,hour,A()); return np.asarray(x['r_km']),np.asarray(x['z_km']),i2_axisym(x)

def coords_km(ds,name):
    q=np.asarray(ds[name][:],float)
    return q/1000 if np.nanmax(np.abs(q))>1.e4 else q

def horizontal(path,hour,zkm,target_x,target_y):
    with Dataset(path) as ds:
        time=np.asarray(ds['time'][:],float)/3600.; it=int(np.argmin(abs(time-hour)))
        x,y,z=coords_km(ds,'xh'),coords_km(ds,'yh'),coords_km(ds,'zh'); iz=int(np.argmin(abs(z-zkm)))
        ps=np.asarray(ds['psfc'][it],float); iy,ix=np.unravel_index(np.nanargmin(gaussian_filter(ps,2)),ps.shape)
        xc,yc=x[ix],y[iy]
        # CM1 Arakawa-C winds, destaggered to xh/yh scalar points.
        u=np.asarray(ds['u'][it,iz],float); v=np.asarray(ds['v'][it,iz],float)
    u=.5*(u[:,:-1]+u[:,1:]); v=.5*(v[:-1,:]+v[1:,:])
    xx,yy=np.meshgrid(x-xc,y-yc); rr=np.hypot(xx,yy)*1000.; rr=np.maximum(rr,6000.)
    vt=(-u*yy+v*xx)/np.maximum(np.hypot(xx,yy),6.)
    # Derivatives use physical coordinate spacings (m).
    dvdx=np.gradient(v,x*1000.,axis=1,edge_order=2); dudy=np.gradient(u,y*1000.,axis=0,edge_order=2)
    i2=(F+2*vt/rr)*(F+dvdx-dudy)
    interp=RegularGridInterpolator((y-yc,x-xc),i2,bounds_error=False,fill_value=np.nan)
    return interp(np.c_[target_y.ravel(),target_x.ravel()]).reshape(target_x.shape), (xc,yc), float(z[iz])

def symmetric(fields):
    return max(float(np.nanpercentile(np.concatenate([abs(q[np.isfinite(q)]) for q in fields]),99)),1e-12)

def positive(fields):
    return max(float(np.nanpercentile(np.concatenate([q[np.isfinite(q)] for q in fields]),99)),1e-12)

def labels(rec): return f"J{rec['jet_h']:.0f}/C{rec['ctrl_h']:.0f}"

def plot_vertical(records, jet_path, ctrl_path, out, title):
    arrays=[]
    for rec in records:
        r,z,j=axisym(jet_path,rec['jet_h']); _,_,c=axisym(ctrl_path,rec['ctrl_h']); arrays.append((r,z,j,c,j-c))
    vmax=positive([q for x in arrays for q in x[2:4]]); dmax=symmetric([x[4] for x in arrays])
    fig,ax=plt.subplots(3,len(records),figsize=(3.05*len(records),8.4),sharex=True,sharey=True,constrained_layout=True)
    for k,(rec,(r,z,j,c,d)) in enumerate(zip(records,arrays)):
        for row,q,name,cmap,vmin,vmax0 in [(0,j,'JET $I^2$','viridis',0,vmax),(1,c,'CTRL $I^2$','viridis',0,vmax),(2,d,'JET−CTRL $\\Delta I^2$','RdBu_r',-dmax,dmax)]:
            im=ax[row,k].contourf(r,z,q,levels=25,cmap=cmap,vmin=vmin,vmax=vmax0,extend='both')
            ax[row,k].set_title(labels(rec) if row==0 else '')
            if k==0: ax[row,k].set_ylabel(name+'\nHeight (km)')
            ax[row,k].set_xlim(0,600); ax[row,k].set_ylim(0,20)
        ax[2,k].set_xlabel('Radius (km)')
    fig.colorbar(ax[0,0].collections[0],ax=ax[:2,:],pad=.01,label='$I^2$ (s$^{-2}$)')
    fig.colorbar(ax[2,0].collections[0],ax=ax[2,:],pad=.01,label='$\\Delta I^2$ (s$^{-2}$)')
    fig.suptitle(title+' | azimuthal-mean physical inertial stability',fontweight='bold')
    fig.savefig(out/'I2_vertical_JET_CTRL_difference.png',dpi=220,bbox_inches='tight'); plt.close(fig)

def plot_horizontal(records, jet_path, ctrl_path, out, zreq, title):
    grid=np.arange(-360.,360.1,6.); xx,yy=np.meshgrid(grid,grid)
    arrays=[]; actual=[]
    for rec in records:
        j,cj,zj=horizontal(jet_path,rec['jet_h'],zreq,xx,yy); c,cc,zc=horizontal(ctrl_path,rec['ctrl_h'],zreq,xx,yy)
        arrays.append((j,c,j-c)); actual.append((zj,zc,cj,cc))
    vmax=positive([q for x in arrays for q in x[:2]]); dmax=symmetric([x[2] for x in arrays])
    fig,ax=plt.subplots(3,len(records),figsize=(3.05*len(records),8.4),sharex=True,sharey=True,constrained_layout=True)
    for k,(rec,(j,c,d)) in enumerate(zip(records,arrays)):
        for row,q,name,cmap,vmin,vmax0 in [(0,j,'JET $I^2$','viridis',0,vmax),(1,c,'CTRL $I^2$','viridis',0,vmax),(2,d,'JET−CTRL $\\Delta I^2$','RdBu_r',-dmax,dmax)]:
            im=ax[row,k].pcolormesh(grid,grid,q,cmap=cmap,vmin=vmin,vmax=vmax0,shading='auto')
            ax[row,k].contour(grid,grid,np.hypot(xx,yy),levels=[100,200],colors='w',linewidths=.45,alpha=.6)
            if row==0: ax[row,k].set_title(labels(rec))
            if k==0: ax[row,k].set_ylabel(name+'\ny (km)')
            ax[row,k].set_aspect('equal'); ax[row,k].set_xlim(-360,360); ax[row,k].set_ylim(-360,360)
        ax[2,k].set_xlabel('x from TC centre (km)')
    fig.colorbar(ax[0,0].collections[0],ax=ax[:2,:],pad=.01,label='$I^2$ (s$^{-2}$)')
    fig.colorbar(ax[2,0].collections[0],ax=ax[2,:],pad=.01,label='$\\Delta I^2$ (s$^{-2}$)')
    fig.suptitle(f'{title} | horizontal physical inertial stability near {zreq:g} km',fontweight='bold')
    fig.savefig(out/f'I2_horizontal_{zreq:g}km_JET_CTRL_difference.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    return actual

def main():
    p=argparse.ArgumentParser(); p.add_argument('--matches-json',required=True); p.add_argument('--jet',required=True); p.add_argument('--ctrl',required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--label',required=True); a=p.parse_args()
    records=json.loads(Path(a.matches_json).read_text()); a.output_dir.mkdir(parents=True,exist_ok=True)
    plot_vertical(records,a.jet,a.ctrl,a.output_dir,a.label)
    meta={'definition':'I2=(f+2*vtheta/r)*(f+zeta); f=6.1636e-5 s-1','vertical':'azimuthal mean vtheta','horizontal':'native u/v, TC-centred after each case is independently centred','horizontal_levels_requested_km':[12,14],'matches':records}
    for z in (12.,14.): plot_horizontal(records,a.jet,a.ctrl,a.output_dir,z,a.label)
    (a.output_dir/'I2_diagnostic_metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__': main()
