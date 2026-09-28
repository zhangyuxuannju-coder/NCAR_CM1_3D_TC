#!/usr/bin/env python3
"""Animate dry Ertel PV and horizontal flow with fixed plotting scales."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import netCDF4
import numpy as np
from scipy.ndimage import gaussian_filter

def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--height-km",type=float,default=15); p.add_argument("--start-hour",type=float,default=0)
    p.add_argument("--end-hour",type=float,default=None); p.add_argument("--interval-hours",type=float,default=1)
    p.add_argument("--fps",type=int,default=8); p.add_argument("--pv-limit",type=float,default=10)
    p.add_argument("--jet-offset-km",type=float,default=999); p.add_argument("--x-half-width-km",type=float,default=1800)
    p.add_argument("--y-south-km",type=float,default=1200); p.add_argument("--y-north-km",type=float,default=1800)
    p.add_argument("--quiver-stride",type=int,default=12); p.add_argument("--pv-smooth-sigma",type=float,default=.65)
    p.add_argument("--sample-only",action="store_true"); p.add_argument("--keep-frames",action="store_true")
    return p

def fields(ds,it,k,x,y,z,f):
    ks=slice(k-1,k+2); th=np.asarray(ds["th"][it,ks],float); rho=np.asarray(ds["rho"][it,ks],float)
    us=np.asarray(ds["u"][it,ks],float); vs=np.asarray(ds["v"][it,ks],float)
    u=.5*(us[...,:-1]+us[...,1:]); v=.5*(vs[...,:-1,:]+vs[...,1:,:])
    ws=np.asarray(ds["w"][it,k-1:k+3],float); w=.5*(ws[:-1]+ws[1:])
    zm=z[k-1:k+2]*1000; ym=y*1000; xm=x*1000
    tz,ty,tx=np.gradient(th,zm,ym,xm,edge_order=2); uz,uy,ux=np.gradient(u,zm,ym,xm,edge_order=2)
    vz,vy,vx=np.gradient(v,zm,ym,xm,edge_order=2); wz,wy,wx=np.gradient(w,zm,ym,xm,edge_order=2)
    pv=((wy-vz)*tx+(uz-wx)*ty+(vx-uy+f)*tz)/np.maximum(rho,1e-8)*1e6
    return pv[1],u[1],v[1]

def main():
    a=parser().parse_args(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with netCDF4.Dataset(a.input) as ds:
        th=np.asarray(ds["time"][:],float)/3600; x=np.asarray(ds["xh"][:],float); y=np.asarray(ds["yh"][:],float); z=np.asarray(ds["zh"][:],float); f=float(np.asarray(ds["f_cor"][:]).ravel()[0]); k=int(np.argmin(abs(z-a.height_km)))
        end=float(th[-1]) if a.end_hour is None else a.end_hour
        if a.sample_only:
            sample_hours=[a.start_hour,72,120,180,end]; vals=[]
            for target in sample_hours:
                it=int(np.argmin(abs(th-target))); pv,_,_=fields(ds,it,k,x,y,z,f); q=np.nanpercentile(pv,[.1,1,50,99,99.9]); vals.append(dict(time_hour=float(th[it]),percentiles_pvu=q.tolist(),min=float(np.nanmin(pv)),max=float(np.nanmax(pv)))); print(vals[-1],flush=True)
            out.with_suffix(".sample.json").write_text(json.dumps(dict(selected_height_km=float(z[k]),samples=vals),indent=2)); return
        indices=[]
        for target in np.arange(a.start_hour,end+a.interval_hours/4,a.interval_hours):
            it=int(np.argmin(abs(th-target)))
            if not indices or it!=indices[-1]: indices.append(it)
        fd=out.parent/(out.stem+"_frames"); fd.mkdir(exist_ok=True); rec=[]
        cmap=LinearSegmentedColormap.from_list("pv",["#2F4B7C","#6785B5","#C5D4E8","#F7F7F4","#F3C6B3","#E56B4E","#9D2933"],N=256); norm=TwoSlopeNorm(vmin=-a.pv_limit,vcenter=0,vmax=a.pv_limit)
        for n,it in enumerate(indices):
            ps=gaussian_filter(np.asarray(ds["psfc"][it],float),2,mode="nearest"); iyc,ixc=np.unravel_index(np.argmin(ps),ps.shape); xc,yc=float(x[ixc]),float(y[iyc]); pv,u,v=fields(ds,it,k,x,y,z,f); pv=gaussian_filter(pv,a.pv_smooth_sigma,mode="nearest")
            xr,yr=x-xc,y-yc; mx=abs(xr)<=a.x_half_width_km; my=(yr>=-a.y_south_km)&(yr<=a.y_north_km); xp,yp=xr[mx],yr[my]; pp=pv[np.ix_(my,mx)]; up=u[np.ix_(my,mx)]; vp=v[np.ix_(my,mx)]; px=np.nanmean(pp,axis=1)
            fig=plt.figure(figsize=(14,8.5),dpi=120); gs=fig.add_gridspec(1,2,width_ratios=(4.8,1.35),wspace=.12); ax=fig.add_subplot(gs[0]); ap=fig.add_subplot(gs[1],sharey=ax)
            m=ax.pcolormesh(xp,yp,pp,shading="auto",cmap=cmap,norm=norm,rasterized=True); cb=fig.colorbar(m,ax=ax,pad=.015,fraction=.04,extend="both"); cb.set_label("Dry Ertel potential vorticity (PVU)")
            xx,yy=np.meshgrid(xp,yp); s=max(1,a.quiver_stride); q=ax.quiver(xx[::s,::s],yy[::s,::s],up[::s,::s],vp[::s,::s],color="#20242C",alpha=.65,scale=500,width=.0022); ax.quiverkey(q,.89,.04,20,"20 m s$^{-1}$",coordinates="axes")
            ax.scatter(0,0,s=70,facecolor="white",edgecolor="black",linewidth=1.5,zorder=6); ax.axhline(a.jet_offset_km,color="magenta",ls="--",lw=2,label=f"Initial jet axis (+{a.jet_offset_km:.0f} km)"); ax.set(xlim=(-a.x_half_width_km,a.x_half_width_km),ylim=(-a.y_south_km,a.y_north_km),xlabel="Zonal distance from TC centre (km)",ylabel="Meridional distance from TC centre (km)"); ax.grid(alpha=.18,ls="--"); ax.legend(loc="lower left",fontsize=9,framealpha=.92); ax.set_title(f"Dry Ertel PV and horizontal flow: t={th[it]:.0f} h, z={z[k]:.2f} km\n{Path(a.input).name}",fontweight="bold")
            ap.plot(px,yp,color="navy",lw=2); ap.axhline(a.jet_offset_km,color="magenta",ls="--",lw=1.8); ap.axvline(0,color=".4",lw=.8); ap.set_xlim(-a.pv_limit,a.pv_limit); ap.set_xlabel("x-mean PV (PVU)"); ap.set_title("Zonal-mean\nPV profile"); ap.grid(alpha=.25,ls="--"); ap.tick_params(labelleft=False)
            frame=fd/f"frame_{n:04d}.png"; fig.savefig(frame,bbox_inches="tight"); plt.close(fig); qq=np.nanpercentile(pp,[1,50,99]); rec.append(dict(frame=n,time_hour=float(th[it]),tc_center_km=[xc,yc],pv_percentiles_1_50_99_pvu=qq.tolist())); print(f"frame {n+1}/{len(indices)}: {th[it]:.0f} h",flush=True)
    ff="/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/ffmpeg"; subprocess.run([ff,"-y","-framerate",str(a.fps),"-i",str(fd/"frame_%04d.png"),"-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2","-c:v","libx264","-pix_fmt","yuv420p","-crf","20",str(out)],check=True)
    meta=dict(definition="dry Ertel PV = rho^-1 absolute_vorticity dot grad(theta)",input=a.input,output=str(out),selected_height_km=float(z[k]),fixed_pv_colorbar_pvu=[-a.pv_limit,a.pv_limit],fps=a.fps,interval_hours=a.interval_hours,frames=rec); out.with_suffix(".json").write_text(json.dumps(meta,indent=2))
    if not a.keep_frames:
        for fpath in fd.glob("frame_*.png"): fpath.unlink()
        fd.rmdir()
if __name__=="__main__": main()
