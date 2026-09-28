#!/usr/bin/env python3
from pathlib import Path
import csv, json
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

SRC = "/data/zhangyx/DATA/cm1out_25N_nojet.nc"
OUT = Path("output/ctrl_hourly_cref_25N_70_82h_r200km")
START_HOUR, END_HOUR, RMAX_KM = 70, 82, 200.0
LEVELS = np.arange(0.0, 65.0, 5.0)

def draw_panel(ax, xrel, yrel, field, hour, pmin, labels=True):
    cmap = plt.get_cmap("turbo", len(LEVELS)-1)
    m = ax.contourf(xrel, yrel, field, levels=LEVELS, cmap=cmap,
                    norm=BoundaryNorm(LEVELS, cmap.N), extend="max")
    ax.contour(xrel, yrel, field, levels=[20,30,40,50], colors="black",
               linewidths=.45, alpha=.65)
    ax.plot(0, 0, "+", color="white", ms=8, mew=1.4)
    for radius in (50,100,150,200):
        ax.add_patch(plt.Circle((0,0), radius, fill=False, color="white", lw=.55, alpha=.6))
    ax.set(xlim=(-RMAX_KM,RMAX_KM), ylim=(-RMAX_KM,RMAX_KM)); ax.set_aspect("equal")
    ax.set_title(f"CTRL  t = {hour:.0f} h   Pmin = {pmin:.1f} hPa")
    if labels:
        ax.set_xlabel("Storm-relative x (km)"); ax.set_ylabel("Storm-relative y (km)")
    return m

def main():
    frames = OUT/"frames"; frames.mkdir(parents=True, exist_ok=True)
    records, panels = [], []
    with Dataset(SRC) as ds:
        hours=np.asarray(ds["time"][:],float)/3600; x=np.asarray(ds["xh"][:],float); y=np.asarray(ds["yh"][:],float)
        indices=[int(np.argmin(abs(hours-h))) for h in range(START_HOUR,END_HOUR+1)]
        if any(abs(hours[it]-h)>.05 for it,h in zip(indices,range(START_HOUR,END_HOUR+1))):
            raise ValueError("Dataset does not contain every requested integer hour")
        for it in indices:
            hour=float(hours[it]); ps=np.asarray(ds["psfc"][it],float)
            iy,ix=np.unravel_index(np.nanargmin(gaussian_filter(ps,2,mode="nearest")),ps.shape)
            xc,yc=float(x[ix]),float(y[iy]); pmin=float(np.nanmin(ps)/100)
            xsel=np.flatnonzero((x>=xc-RMAX_KM-1)&(x<=xc+RMAX_KM+1))
            ysel=np.flatnonzero((y>=yc-RMAX_KM-1)&(y<=yc+RMAX_KM+1))
            x0,x1=int(xsel[0]),int(xsel[-1])+1; y0,y1=int(ysel[0]),int(ysel[-1])+1
            xx,yy=np.meshgrid(x[x0:x1]-xc,y[y0:y1]-yc); cref=np.asarray(ds["cref"][it,y0:y1,x0:x1],float)
            cref=np.ma.masked_where((np.hypot(xx,yy)>RMAX_KM)|~np.isfinite(cref),cref)
            fig,ax=plt.subplots(figsize=(7.2,6.4),constrained_layout=True); m=draw_panel(ax,xx,yy,cref,hour,pmin)
            fig.colorbar(m,ax=ax,pad=.02,label="Composite radar reflectivity (dBZ)")
            fig.savefig(frames/f"ctrl_cref_t{hour:03.0f}h.png",dpi=220); plt.close(fig)
            panels.append((hour,pmin,xx,yy,cref)); records.append((hour,pmin,xc,yc)); print(f"Finished {hour:.0f} h",flush=True)
    fig,axes=plt.subplots(4,4,figsize=(15,14.3),constrained_layout=True); axes=axes.ravel(); m=None
    for ax,(hour,pmin,xx,yy,cref) in zip(axes,panels):
        m=draw_panel(ax,xx,yy,cref,hour,pmin,False); ax.set_xticks([-200,-100,0,100,200]); ax.set_yticks([-200,-100,0,100,200]); ax.tick_params(labelsize=7)
    for ax in axes[len(panels):]: ax.axis("off")
    fig.colorbar(m,ax=axes.tolist(),shrink=.78,pad=.015,label="Composite radar reflectivity (dBZ)")
    fig.suptitle("25N CTRL hourly storm-centered composite reflectivity (70–82 h)",fontsize=16,fontweight="bold")
    fig.savefig(OUT/"ctrl_cref_70_82h_montage.png",dpi=220); plt.close(fig)
    with (OUT/"centers_and_pressure.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["time_h","pmin_hpa","center_x_km","center_y_km"]); w.writerows(records)
    (OUT/"manifest.json").write_text(json.dumps({"source":SRC,"field":"cref","hours":[70,82],"interval_hours":1,"radius_km":200,"center_method":"smoothed psfc minimum","reflectivity_levels_dbz":LEVELS.tolist()},indent=2),encoding="utf-8")
    print(OUT,flush=True)

if __name__ == "__main__": main()
