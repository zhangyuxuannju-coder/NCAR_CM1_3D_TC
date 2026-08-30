#!/usr/bin/env python3
"""CM1 secondary-circulation, closure, and wavenumber-1 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from netCDF4 import Dataset
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src._se_pipeline_single import PipelineConfig, azimuthal_average_from_3d


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True); p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--times-hours", type=float, nargs="+", default=[25, 55, 80, 110])
    p.add_argument("--max-r-km", type=float, default=300.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--f", type=float, default=5.464e-5)
    return p.parse_args()


def bin_mean(values, idx, valid, nr):
    out = np.full((values.shape[0], nr), np.nan)
    count = np.bincount(idx[valid], minlength=nr).astype(float)
    for k in range(values.shape[0]):
        flat = np.asarray(values[k], float).ravel(); use = valid & np.isfinite(flat)
        total = np.bincount(idx[use], weights=flat[use], minlength=nr)
        out[k] = np.divide(total, count, out=np.full(nr, np.nan), where=count > 0)
    return out


def bin_k1(values, phase, idx, valid, nr):
    out = np.full((values.shape[0], nr), np.nan + 1j*np.nan)
    count = np.bincount(idx[valid], minlength=nr).astype(float)
    for k in range(values.shape[0]):
        flat = np.asarray(values[k], float).ravel(); use = valid & np.isfinite(flat)
        x = flat[use] * phase[use]
        re = np.bincount(idx[use], weights=x.real, minlength=nr)
        im = np.bincount(idx[use], weights=x.imag, minlength=nr)
        out[k] = np.divide(re+1j*im, count,
                           out=np.full(nr, np.nan+1j*np.nan), where=count > 0)
    return out


def load_modes(path, hour, avg, max_z):
    with Dataset(path) as nc:
        time = np.asarray(nc["time"][:], float); it = int(np.argmin(abs(time-hour*3600)))
        xh = np.asarray(nc["xh"][:], float); yh = np.asarray(nc["yh"][:], float)
        zh = np.asarray(nc["zh"][:], float); keep = np.flatnonzero(zh <= max_z); nk = keep.size
        u0 = np.asarray(nc["u"][it, keep], np.float32)
        v0 = np.asarray(nc["v"][it, keep], np.float32)
        w0 = np.asarray(nc["w"][it, :nk+1], np.float32)
        rho = np.asarray(nc["rho"][it, keep], np.float32)
    u = .5*(u0[:, :, :-1]+u0[:, :, 1:]); v = .5*(v0[:, :-1]+v0[:, 1:])
    w = .5*(w0[:-1]+w0[1:]); del u0, v0, w0
    xx, yy = np.meshgrid(xh, yh)
    dx = xx-float(avg["center_x_km"][0]); dy = yy-float(avg["center_y_km"][0])
    radius = np.hypot(dx, dy); angle = np.arctan2(dy, dx)
    ur = u*np.cos(angle)[None]+v*np.sin(angle)[None]
    r = np.asarray(avg["r_km"], float); dr = float(np.median(np.diff(r)))
    edges = np.r_[max(0., r[0]-.5*dr), r+.5*dr]
    idx = np.digitize(radius.ravel(), edges)-1; valid = (idx >= 0) & (idx < r.size)
    phase = np.exp(-1j*angle.ravel()); fr = rho*ur; fz = rho*w
    return dict(hour=float(time[it]/3600), rho=bin_mean(rho,idx,valid,r.size),
                fr0=bin_mean(fr,idx,valid,r.size), fz0=bin_mean(fz,idx,valid,r.size),
                fr1=2*abs(bin_k1(fr,phase,idx,valid,r.size)),
                fz1=2*abs(bin_k1(fz,phase,idx,valid,r.size)))


def fit_psi(fr, fz, r, z):
    nz,nr=fr.shape; rows=[]; cols=[]; vals=[]; rhs=[]; q=0
    for k in range(nz-1):
        dz=z[k+1]-z[k]; target=-r*.5*(fr[k]+fr[k+1])
        for j in range(nr):
            if np.isfinite(target[j]):
                rows += [q,q]; cols += [k*nr+j,(k+1)*nr+j]
                vals += [-1/dz,1/dz]; rhs.append(target[j]); q+=1
    for k in range(nz):
        for j in range(nr-1):
            dr=r[j+1]-r[j]; target=.5*(r[j]*fz[k,j]+r[j+1]*fz[k,j+1])
            if np.isfinite(target):
                rows += [q,q]; cols += [k*nr+j,k*nr+j+1]
                vals += [-1/dr,1/dr]; rhs.append(target); q+=1
    rows.append(q); cols.append(nz*nr-1); vals.append(1.); rhs.append(0.); q+=1
    A=sparse.coo_matrix((vals,(rows,cols)),shape=(q,nz*nr)).tocsr()
    sol=lsqr(A,np.asarray(rhs),atol=1e-11,btol=1e-11,iter_lim=20000)
    psi=sol[0].reshape(nz,nr); psi-=psi[-1,-1]
    pz=np.gradient(psi,z,axis=0); pr=np.gradient(psi,r,axis=1)
    tz=-fr*r[None]; tr=fz*r[None]; mask=np.ones_like(psi,bool)
    mask[[0,-1]]=False; mask[:,[0,-1]]=False
    nrmse=np.sqrt(np.nanmean(((pz-tz)**2+(pr-tr)**2)[mask]))/max(
        np.sqrt(np.nanmean((tz**2+tr**2)[mask])),1e-12)
    return psi,float(nrmse),dict(stop=int(sol[1]),iterations=int(sol[2]))


def rms(x,mask):
    a=np.asarray(x)[mask & np.isfinite(x)]
    return float(np.sqrt(np.mean(a*a))) if a.size else np.nan


def analyze(case,path,hour,a):
    cfg=PipelineConfig(input_file=path,output_dir=a.output_dir,target_time_hours=hour,
        max_r_km=a.max_r_km,dr_km=a.dr_km,max_z_km=a.max_z_km,coriolis_f=a.f,
        include_model_budget_terms=False,write_netcdf=False,write_ieee=False,plot_solution=False)
    avg=azimuthal_average_from_3d(cfg); m=load_modes(path,hour,avg,a.max_z_km)
    rk=np.asarray(avg["r_km"],float); zk=np.asarray(avg["z_km"],float)
    r=rk*1000; z=zk*1000; psi,closure,solver=fit_psi(m["fr0"],m["fz0"],r,z)
    rho=np.maximum(m["rho"],1e-8); ur=m["fr0"]/rho; w=m["fz0"]/rho
    rr,zz=np.meshgrid(rk,zk); upper=(rr>=50)&(rr<=a.max_r_km)&(zz>=10)&(zz<=16)
    zmask=(zk>=10)&(zk<=16); qout=2*np.pi*r*np.trapezoid(m["fr0"][zmask],z[zmask],axis=0)
    rmask=(rk>=50)&(rk<=a.max_r_km); jj=np.flatnonzero(rmask)[np.nanargmax(qout[rmask])]
    return dict(case=case,hour=m["hour"],r_km=rk,z_km=zk,psi=psi,ur=ur,w=w,rho=rho,
        k1_ur=m["fr1"]/rho,k1_w=m["fz1"]/rho,closure_nrmse=closure,solver=solver,
        k1_k0_ur=rms(m["fr1"],upper)/max(rms(m["fr0"],upper),1e-12),
        k1_k0_w=rms(m["fz1"],upper)/max(rms(m["fz0"],upper),1e-12),
        qout=float(qout[jj]),qout_radius=float(rk[jj]),
        center_x=float(avg["center_x_km"][0]),center_y=float(avg["center_y_km"][0]))


def cmap():
    return LinearSegmentedColormap.from_list("nature",["#3C5488","#A8BFDC","#F7F7F4","#F1B39A","#B9363E"])


def contours(ax,r,z,u):
    for lev,col,ls in (([-10,-5,-2],"#3C5488","--"),([2,5,10],"#009E73","-")):
        use=[x for x in lev if np.nanmin(u)<=x<=np.nanmax(u)]
        if use:
            cs=ax.contour(r,z,u,levels=use,colors=col,linestyles=ls,linewidths=.8)
            ax.clabel(cs,fmt="%g",fontsize=6)


def plot_sections(panels,times,path):
    fig,ax=plt.subplots(len(times),3,figsize=(13,3.05*len(times)),sharex=True,sharey=True,
                       constrained_layout=True,squeeze=False)
    vmax=np.nanpercentile(abs(np.concatenate([(x["psi"]*2*np.pi/1e9).ravel() for x in panels])),98)
    diffs=[(panels[2*i+1]["psi"]-panels[2*i]["psi"])*2*np.pi/1e9 for i in range(len(times))]
    dmax=np.nanpercentile(abs(np.concatenate([x.ravel() for x in diffs])),98); cm=cmap(); letters="abcdefghijklmnopqrstuvwxyz"
    for i,h in enumerate(times):
        for j,rec in enumerate(panels[2*i:2*i+2]):
            mc=ax[i,j].pcolormesh(rec["r_km"],rec["z_km"],rec["psi"]*2*np.pi/1e9,
                cmap=cm,norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),shading="auto")
            contours(ax[i,j],rec["r_km"],rec["z_km"],rec["ur"])
            ax[i,j].set_title(f"{rec['case']} · {rec['hour']:.0f} h | closure={rec['closure_nrmse']:.2f}")
        d=panels[2*i+1]["psi"]-panels[2*i]["psi"]; du=panels[2*i+1]["ur"]-panels[2*i]["ur"]
        md=ax[i,2].pcolormesh(panels[2*i]["r_km"],panels[2*i]["z_km"],d*2*np.pi/1e9,
            cmap=cm,norm=TwoSlopeNorm(vmin=-dmax,vcenter=0,vmax=dmax),shading="auto")
        contours(ax[i,2],panels[2*i]["r_km"],panels[2*i]["z_km"],du); ax[i,2].set_title(f"JET − noJET · {h:.0f} h")
        for j in range(3):
            ax[i,j].set_xlim(0,300); ax[i,j].set_ylim(0,20); ax[i,j].text(-.08,1.02,f"({letters[3*i+j]})",transform=ax[i,j].transAxes,fontweight="bold")
    for x in ax[-1]: x.set_xlabel("Radius (km)")
    for x in ax[:,0]: x.set_ylabel("Height (km)")
    fig.colorbar(mc,ax=ax[:,:2],orientation="horizontal",shrink=.72,pad=.03,label=r"Best-fit $2\pi\psi$ ($10^9$ kg s$^{-1}$)")
    fig.colorbar(md,ax=ax[:,2],orientation="horizontal",shrink=.9,pad=.03,label=r"JET−noJET $2\pi\Delta\psi$ ($10^9$ kg s$^{-1}$)")
    fig.suptitle("CM1 secondary circulation (green/blue: radial outflow/inflow, m s$^{-1}$)",fontweight="bold")
    fig.savefig(path,dpi=260,bbox_inches="tight"); plt.close(fig)


def plot_metrics(panels,path):
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True); color={"noJET":"#3C5488","JET":"#B9363E"}
    for c in color:
        x=[p for p in panels if p["case"]==c]; h=[p["hour"] for p in x]
        ax[0].plot(h,[p["k1_k0_ur"] for p in x],"o-",color=color[c],label=c)
        ax[1].plot(h,[p["k1_k0_w"] for p in x],"o-",color=color[c],label=c)
        ax[2].plot(h,[p["qout"]/1e9 for p in x],"o-",color=color[c],label=c)
    titles=["Upper-outflow radial mass flux","Upper-outflow vertical mass flux","Maximum signed upper-outflow transport"]
    yl=["k=1 amplitude / k=0 RMS","k=1 amplitude / k=0 RMS",r"$10^9$ kg s$^{-1}$"]
    for q in range(3): ax[q].set(title=titles[q],xlabel="Time (h)",ylabel=yl[q]); ax[q].grid(alpha=.25); ax[q].legend(frameon=False)
    fig.savefig(path,dpi=260,bbox_inches="tight"); plt.close(fig)


def main():
    a=args_parser(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); panels=[]
    for h in a.times_hours:
        panels += [analyze("noJET",a.nojet,h,a),analyze("JET",a.jet,h,a)]
    plot_sections(panels,a.times_hours,out/"figure1_cm1_mass_streamfunction.png")
    plot_metrics(panels,out/"figure2_wavenumber1_and_outflow_metrics.png")
    arrays={"r_km":panels[0]["r_km"],"z_km":panels[0]["z_km"]}; summary={"panels":[]}
    for p in panels:
        tag=f"{p['case'].lower()}_{p['hour']:.0f}h"
        for n in ("psi","ur","w","rho","k1_ur","k1_w"): arrays[f"{n}_{tag}"]=p[n]
        summary["panels"].append({k:v for k,v in p.items() if k not in ("r_km","z_km","psi","ur","w","rho","k1_ur","k1_w")})
    np.savez_compressed(out/"secondary_circulation_products.npz",**arrays)
    (out/"secondary_circulation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
