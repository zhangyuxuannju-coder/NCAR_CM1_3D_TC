#!/usr/bin/env python3
"""Publication figures for CTRL/JET direct CM1 diabatic-heating structure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from skimage.measure import marching_cubes
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src._se_pipeline_single import PipelineConfig, _find_center

CP, RD, P0 = 1004.0, 287.0, 100000.0


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True); p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--same-times", nargs="+", type=float, default=[55,65,70,72,80])
    p.add_argument("--pairs", nargs="+", default=["55:59","65:64","70:66","72:67","80:92"])
    p.add_argument("--max-xy-km", type=float, default=200); p.add_argument("--max-z-km", type=float, default=18)
    p.add_argument("--dr-km", type=float, default=6); p.add_argument("--f", type=float, default=6.2e-5)
    return p.parse_args()


def destagger(arr, axis):
    a=[slice(None)]*arr.ndim; b=[slice(None)]*arr.ndim; a[axis]=slice(0,-1);b[axis]=slice(1,None)
    return .5*(arr[tuple(a)]+arr[tuple(b)])


def radial_mean(field, radius, dr, maxr):
    edges=np.arange(0,maxr+dr,dr); rc=.5*(edges[:-1]+edges[1:]); idx=np.digitize(radius.ravel(),edges)-1
    valid=(idx>=0)&(idx<rc.size); count=np.bincount(idx[valid],minlength=rc.size).astype(float)
    out=np.full((field.shape[0],rc.size),np.nan)
    for k in range(field.shape[0]):
        f=field[k].ravel(); use=valid&np.isfinite(f); s=np.bincount(idx[use],weights=f[use],minlength=rc.size)
        out[k]=np.divide(s,count,out=np.full(rc.size,np.nan),where=count>0)
    return rc,out


def regrid3(field, z, y, x, target_y, target_x):
    interp = RegularGridInterpolator((z, y, x), field, bounds_error=False, fill_value=np.nan)
    zz, yy, xx = np.meshgrid(z, target_y, target_x, indexing="ij")
    points = np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))
    return interp(points).reshape(zz.shape).astype(np.float32)


def load_snapshot(path,hour,a):
    with Dataset(path) as nc:
        time=np.asarray(nc["time"][:],float); it=int(np.argmin(abs(time-hour*3600)))
        x=np.asarray(nc["xh"][:],float);y=np.asarray(nc["yh"][:],float);zall=np.asarray(nc["zh"][:],float)
        keep=np.flatnonzero(zall<=a.max_z_km);z=zall[keep]; nk=keep.size
        ps=np.asarray(nc["psfc"][it],float)
        cfg=PipelineConfig(input_file=path,output_dir=a.output_dir,target_time_hours=hour,coriolis_f=a.f)
        xc,yc=_find_center(ps,x,y,cfg)
        xm=(x>=xc-a.max_xy_km)&(x<=xc+a.max_xy_km);ym=(y>=yc-a.max_xy_km)&(y<=yc+a.max_xy_km)
        q=(np.asarray(nc["ptb_mp"][it,keep][:,ym][:,:,xm],np.float32)+
           np.asarray(nc["ptb_rad"][it,keep][:,ym][:,:,xm],np.float32))
        rho=np.asarray(nc["rho"][it,keep][:,ym][:,:,xm],np.float32)
        prs=np.asarray(nc["prs"][it,keep][:,ym][:,:,xm],np.float32)
        u=destagger(np.asarray(nc["u"][it,keep][:,ym],np.float32),2)[:,:,xm]
        v=destagger(np.asarray(nc["v"][it,keep][:,:,xm],np.float32),1)[:,ym]
        wf=np.asarray(nc["w"][it,:nk+1][:,ym][:,:,xm],np.float32); w=destagger(wf,0)
    xx,yy=np.meshgrid(x[xm]-xc,y[ym]-yc);radius=np.hypot(xx,yy);ang=np.arctan2(yy,xx)
    ut=-u*np.sin(ang)[None]+v*np.cos(ang)[None]
    rc,ut0=radial_mean(ut,radius,a.dr_km,a.max_xy_km)
    iz=int(np.argmin(abs(z-1.0)));rmw=float(rc[np.nanargmax(ut0[iz])])
    _,q0=radial_mean(q,radius,a.dr_km,a.max_xy_km);_,w0=radial_mean(w,radius,a.dr_km,a.max_xy_km)
    exner=np.maximum(prs/P0,1e-6)**(RD/CP); heat=np.maximum(q,0)*rho*CP*exner
    dz=np.gradient(z)*1000; weight=heat*dz[:,None,None]
    total=float(np.nansum(weight)); cr=float(np.nansum(weight*radius[None])/total);cz=float(np.nansum(weight*z[:,None,None])/total)
    zin=[];rin=[]
    for k in range(z.size):
        wk=weight[k]; s=float(np.nansum(wk));zin.append(float(z[k]));rin.append(float(np.nansum(wk*radius)/s) if s>0 else np.nan)
    inside=float(np.nansum(weight[:,radius<=rmw])/total)
    xrel=x[xm]-xc; yrel=y[ym]-yc
    dxy=float(min(np.median(np.diff(x)),np.median(np.diff(y))))
    grid=np.arange(-a.max_xy_km,a.max_xy_km+.25*dxy,dxy)
    q_plot=regrid3(q,z,yrel,xrel,grid,grid); w_plot=regrid3(w,z,yrel,xrel,grid,grid)
    return dict(hour=float(time[it]/3600),x=grid,y=grid,z=z,q=q_plot,w=w_plot,rho=rho,prs=prs,
                r2d=radius,rc=rc,q0=q0,w0=w0,rmw=rmw,xc=xc,yc=yc,centroid_r=cr,centroid_z=cz,
                axis_r=np.asarray(rin),axis_z=np.asarray(zin),inside_fraction=inside)


def symmetric_limit(arrs,p=99): return max(float(np.nanpercentile(abs(x),p)) for x in arrs)


def plot_rz(c,j,label,out):
    fields=[c["q0"]*3600,j["q0"]*3600,(j["q0"]-c["q0"])*3600]; vmax=symmetric_limit(fields)
    fig,ax=plt.subplots(1,3,figsize=(13.5,4.3),sharex=True,sharey=True,constrained_layout=True)
    for k,(a0,f,title) in enumerate(zip(ax,fields,[f"CTRL {c['hour']:.0f} h",f"JET {j['hour']:.0f} h","JET - CTRL"])):
        im=a0.pcolormesh(c["rc"],c["z"],f,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),shading="auto")
        ww=[c["w0"],j["w0"],j["w0"]-c["w0"]][k]; levels=[-.5,-.2,.2,.5,1,2]
        use=[x for x in levels if np.nanmin(ww)<=x<=np.nanmax(ww)]
        if use:a0.contour(c["rc"],c["z"],ww,levels=use,colors="k",linewidths=.65)
        if k<2:
            rec=[c,j][k];a0.axvline(rec["rmw"],color="#009E73",ls="--",lw=1.3);a0.plot(rec["axis_r"],rec["axis_z"],color="#F0E442",lw=2)
        a0.set(title=title,xlabel="Radius (km)");a0.set_xlim(0,200);a0.set_ylim(0,18)
    ax[0].set_ylabel("Height (km)");fig.colorbar(im,ax=ax,label=r"$ptb_{mp}+ptb_{rad}$ (K h$^{-1}$)",shrink=.82)
    fig.suptitle(label+" | black: w; green: RMW; yellow: heating axis",fontweight="bold")
    fig.savefig(out.with_suffix(".png"),dpi=260,bbox_inches="tight");fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)


def plot_xy(c,j,label,out):
    heights=[5,8,12];fig,ax=plt.subplots(3,3,figsize=(11,10),sharex=True,sharey=True,constrained_layout=True)
    allf=[]
    for h in heights:
        iz=int(np.argmin(abs(c["z"]-h)));allf += [c["q"][iz]*3600,j["q"][iz]*3600,(j["q"][iz]-c["q"][iz])*3600]
    vmax=symmetric_limit(allf)
    n=0
    for row,h in enumerate(heights):
        iz=int(np.argmin(abs(c["z"]-h)))
        for col,(rec,f,title) in enumerate([(c,c["q"][iz]*3600,"CTRL"),(j,j["q"][iz]*3600,"JET"),(c,(j["q"][iz]-c["q"][iz])*3600,"JET - CTRL")]):
            a0=ax[row,col];im=a0.pcolormesh(rec["x"],rec["y"],f,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),shading="auto")
            ww=[c["w"][iz],j["w"][iz],j["w"][iz]-c["w"][iz]][col];use=[x for x in [.5,1,2,4] if np.nanmax(ww)>=x]
            if use:a0.contour(rec["x"],rec["y"],ww,levels=use,colors="k",linewidths=.65)
            if col<2:a0.add_patch(plt.Circle((0,0),rec["rmw"],fill=False,color="#009E73",ls="--",lw=1.2))
            if row==0:a0.set_title(title)
            if col==0:a0.set_ylabel(f"{h} km\ny (km)")
            if row==2:a0.set_xlabel("x (km)")
            a0.set_aspect("equal");a0.set_xlim(-150,150);a0.set_ylim(-150,150);n+=1
    fig.colorbar(im,ax=ax,label=r"$Q_\theta$ (K h$^{-1}$)",shrink=.75);fig.suptitle(label+" | black: upward motion; green: RMW",fontweight="bold")
    fig.savefig(out.with_suffix(".png"),dpi=260,bbox_inches="tight");fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)


def add_surface(ax,vol,x,y,z,level,color,alpha):
    if not(np.nanmin(vol)<level<np.nanmax(vol)):return
    v,f,_,_=marching_cubes(np.nan_to_num(vol,nan=np.nanmin(vol)),level=level)
    zz=np.interp(v[:,0],np.arange(len(z)),z);yy=np.interp(v[:,1],np.arange(len(y)),y);xx=np.interp(v[:,2],np.arange(len(x)),x)
    mesh=Poly3DCollection(np.c_[xx,yy,zz][f],alpha=alpha,facecolor=color,edgecolor="none");ax.add_collection3d(mesh)


def plot_3d(c,j,label,out):
    qlev=float(np.nanpercentile(np.r_[np.maximum(c["q"],0).ravel(),np.maximum(j["q"],0).ravel()],95))*3600
    d=(j["q"]-c["q"])*3600;dlev=float(np.nanpercentile(abs(d),95))
    fig=plt.figure(figsize=(15,5)); axes=[fig.add_subplot(1,3,k+1,projection="3d") for k in range(3)]
    for ax,rec,title in zip(axes[:2],[c,j],["CTRL","JET"]):
        add_surface(ax,rec["q"]*3600,rec["x"],rec["y"],rec["z"],qlev,"#B9363E",.38)
        add_surface(ax,rec["w"],rec["x"],rec["y"],rec["z"],1.0,"#3C5488",.16)
        th=np.linspace(0,2*np.pi,181);ax.plot(rec["rmw"]*np.cos(th),rec["rmw"]*np.sin(th),np.zeros_like(th)+.5,"k--",lw=1)
        ax.set_title(title)
    add_surface(axes[2],d,c["x"],c["y"],c["z"],dlev,"#B9363E",.40);add_surface(axes[2],-d,c["x"],c["y"],c["z"],dlev,"#3C5488",.35);axes[2].set_title("JET - CTRL (+red / -blue)")
    for ax in axes:
        ax.set(xlim=(-180,180),ylim=(-180,180),zlim=(0,18),xlabel="x (km)",ylabel="y (km)",zlabel="z (km)");ax.view_init(25,-58);ax.set_box_aspect((1,1,.35))
    fig.suptitle(f"{label} | Q95={qlev:.2f} K h-1; translucent blue in CTRL/JET: w=1 m s-1",fontweight="bold")
    fig.tight_layout();fig.savefig(out.with_suffix(".png"),dpi=260,bbox_inches="tight");fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)


def main():
    a=args_parser();out=Path(a.output_dir);(out/"figures").mkdir(parents=True,exist_ok=True);(out/"products").mkdir(exist_ok=True)
    rows=[]; cache={}
    def get(case,h):
        key=(case,float(h));
        if key not in cache:cache[key]=load_snapshot(a.jet if case=="JET" else a.nojet,h,a)
        return cache[key]
    jobs=[]
    for h in a.same_times:jobs.append((f"same_t{h:03.0f}",get("CTRL",h),get("JET",h),f"Same time {h:g} h"))
    for item in a.pairs:
        jh,ch=map(float,item.split(":"));jobs.append((f"matched_J{jh:03.0f}_C{ch:03.0f}",get("CTRL",ch),get("JET",jh),f"Strength matched J{jh:g}/C{ch:g}"))
    for tag,c,j,label in jobs:
        plot_rz(c,j,label,out/"figures"/(tag+"_rz"));plot_xy(c,j,label,out/"figures"/(tag+"_xy"));plot_3d(c,j,label,out/"figures"/(tag+"_3d"))
        for case,rec in [("CTRL",c),("JET",j)]:rows.append(dict(comparison=tag,case=case,hour=rec["hour"],rmw_km=rec["rmw"],centroid_r_km=rec["centroid_r"],centroid_z_km=rec["centroid_z"],inside_rmw_fraction=rec["inside_fraction"]))
    pd.DataFrame(rows).to_csv(out/"products"/"heating_position_metrics.csv",index=False)
    (out/"manifest.json").write_text(json.dumps({"inputs":{"CTRL":a.nojet,"JET":a.jet},"same_times":a.same_times,"pairs":a.pairs,"Q":"ptb_mp+ptb_rad","figures":"PNG+PDF","three_d":"marching cubes, common paired Q95"},indent=2),encoding="utf-8")

if __name__=="__main__":main()
