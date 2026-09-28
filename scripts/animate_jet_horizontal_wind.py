#!/usr/bin/env python3
"""Animate storm-centred upper-level flow with a fixed colour scale."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy.ndimage import gaussian_filter

P=argparse.ArgumentParser()
for name,kw in [("input",dict(required=True)),("output",dict(required=True)),("height-km",dict(type=float,default=15.0)),("start-hour",dict(type=float,default=0)),("end-hour",dict(type=float,default=None)),("interval-hours",dict(type=float,default=1)),("fps",dict(type=int,default=8)),("speed-max",dict(type=float,default=40)),("jet-offset-km",dict(type=float,default=999)),("x-half-width-km",dict(type=float,default=1800)),("y-south-km",dict(type=float,default=1200)),("y-north-km",dict(type=float,default=1800)),("quiver-stride",dict(type=int,default=12))]: P.add_argument("--"+name,**kw)
P.add_argument("--keep-frames",action="store_true")

def main():
 a=P.parse_args(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fd=out.parent/(out.stem+"_frames"); fd.mkdir(exist_ok=True); rec=[]
 with netCDF4.Dataset(a.input) as ds:
  th=np.asarray(ds["time"][:],float)/3600; x=np.asarray(ds["xh"][:],float); y=np.asarray(ds["yh"][:],float); z=np.asarray(ds["zh"][:],float); iz=int(np.argmin(abs(z-a.height_km))); end=float(th[-1]) if a.end_hour is None else a.end_hour
  ii=[]
  for t in np.arange(a.start_hour,end+a.interval_hours/4,a.interval_hours):
   i=int(np.argmin(abs(th-t)))
   if not ii or i!=ii[-1]: ii.append(i)
  levels=np.linspace(0,a.speed_max,31)
  for n,it in enumerate(ii):
   ps=gaussian_filter(np.asarray(ds["psfc"][it],float),2,mode="nearest"); iyc,ixc=np.unravel_index(np.argmin(ps),ps.shape); xc,yc=float(x[ixc]),float(y[iyc])
   us=np.asarray(ds["u"][it,iz],float); vs=np.asarray(ds["v"][it,iz],float); u=.5*(us[:,:-1]+us[:,1:]); v=.5*(vs[:-1]+vs[1:]); xr,yr=x-xc,y-yc; mx=abs(xr)<=a.x_half_width_km; my=(yr>=-a.y_south_km)&(yr<=a.y_north_km); xp,yp=xr[mx],yr[my]; up,vp=u[np.ix_(my,mx)],v[np.ix_(my,mx)]; sp=np.hypot(up,vp); ux=np.nanmean(up,axis=1); j=int(np.nanargmax(ux)); jy,ju=float(yp[j]),float(ux[j])
   fig=plt.figure(figsize=(14,8.5),dpi=120); gs=fig.add_gridspec(1,2,width_ratios=(4.8,1.35),wspace=.12); ax=fig.add_subplot(gs[0]); ap=fig.add_subplot(gs[1],sharey=ax); cf=ax.contourf(xp,yp,sp,levels=levels,cmap="turbo",extend="max"); cb=fig.colorbar(cf,ax=ax,pad=.015,fraction=.04); cb.set_label("Horizontal wind speed (m s$^{-1}$)"); xx,yy=np.meshgrid(xp,yp); s=max(1,a.quiver_stride); q=ax.quiver(xx[::s,::s],yy[::s,::s],up[::s,::s],vp[::s,::s],color="black",alpha=.65,scale=500,width=.0022); ax.quiverkey(q,.89,.04,20,"20 m s$^{-1}$",coordinates="axes"); ax.scatter(0,0,s=70,facecolor="white",edgecolor="black",linewidth=1.5,zorder=6); ax.axhline(a.jet_offset_km,color="magenta",ls="--",lw=2,label=f"Initial jet axis (+{a.jet_offset_km:.0f} km)"); ax.axhline(jy,color="cyan",ls="-.",lw=2,label=f"Diagnosed x-mean u max ({jy:.0f} km)"); ax.set(xlim=(-a.x_half_width_km,a.x_half_width_km),ylim=(-a.y_south_km,a.y_north_km),xlabel="Zonal distance from TC centre (km)",ylabel="Meridional distance from TC centre (km)"); ax.grid(alpha=.18,ls="--"); ax.legend(loc="lower left",fontsize=9,framealpha=.92); ax.set_title(f"Upper-level horizontal wind: t={th[it]:.0f} h, z={z[iz]:.2f} km\n{Path(a.input).name}",fontweight="bold"); ap.plot(ux,yp,color="navy",lw=2.2); ap.axhline(a.jet_offset_km,color="magenta",ls="--",lw=1.8); ap.axhline(jy,color="cyan",ls="-.",lw=1.8); ap.axvline(0,color=".4",lw=.8); ap.scatter([ju],[jy],color="cyan",edgecolor="black",zorder=5); ap.set_xlim(-15,a.speed_max); ap.set_xlabel("x-mean u (m s$^{-1}$)"); ap.set_title("Zonal-mean\nwesterly profile"); ap.grid(alpha=.25,ls="--"); ap.tick_params(labelleft=False); frame=fd/f"frame_{n:04d}.png"; fig.savefig(frame,bbox_inches="tight"); plt.close(fig); rec.append(dict(frame=n,time_hour=float(th[it]),tc_center_km=[xc,yc],jet_axis_relative_y_km=jy,xmean_u_max_m_s=ju,domain_wind_max_m_s=float(np.nanmax(sp)))); print(f"frame {n+1}/{len(ii)}: {th[it]:.0f} h",flush=True)
 try:
  import imageio_ffmpeg; ff=imageio_ffmpeg.get_ffmpeg_exe()
 except Exception: ff="ffmpeg"
 subprocess.run([ff,"-y","-framerate",str(a.fps),"-i",str(fd/"frame_%04d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-crf","20",str(out)],check=True)
 meta=dict(input=a.input,output=str(out),selected_height_km=float(z[iz]),fixed_colorbar_m_s=[0,a.speed_max],fps=a.fps,interval_hours=a.interval_hours,frames=rec); out.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
 if not a.keep_frames:
  for f in fd.glob("frame_*.png"): f.unlink()
  fd.rmdir()
 print(json.dumps({k:v for k,v in meta.items() if k!="frames"},indent=2))
if __name__=="__main__": main()
