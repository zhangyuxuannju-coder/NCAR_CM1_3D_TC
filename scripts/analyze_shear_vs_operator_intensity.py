#!/usr/bin/env python3
"""Joint intensity, environmental shear, and SE operator-effect diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from netCDF4 import Dataset
import numpy as np
from scipy.ndimage import gaussian_filter


COLORS = {"CTRL": "#3C5488", "JET15": "#009E73", "JET30": "#B9363E"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrl", required=True)
    p.add_argument("--jet15", required=True)
    p.add_argument("--jet30", required=True)
    p.add_argument("--factorial-jet15", required=True)
    p.add_argument("--factorial-jet30", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hours", type=float, nargs="+", default=[24, 48, 72, 96, 120])
    p.add_argument("--lower-z-km", type=float, default=2.0)
    p.add_argument("--upper-z-km", type=float, default=12.0)
    p.add_argument("--shear-r-inner-km", type=float, default=200.0)
    p.add_argument("--shear-r-outer-km", type=float, default=1000.0)
    p.add_argument("--max-time-hours", type=float, default=120.0)
    return p.parse_args()


def intensity(path: str, max_hour: float):
    with Dataset(path) as ds:
        t = np.asarray(ds["time"][:], float) / 3600.0
        keep = np.flatnonzero(t <= max_hour + 1e-6)
        p = np.full(keep.size, np.nan)
        for n, it in enumerate(keep):
            f = np.asarray(ds["psfc"][it], float)
            p[n] = np.nanmin(gaussian_filter(f, sigma=2.0, mode="nearest")) / 100.0
    tt = t[keep]
    rate = -np.gradient(p, tt) * 24.0
    return tt, p, rate


def destagger(u_stag, v_stag):
    return .5*(u_stag[..., :-1]+u_stag[..., 1:]), .5*(v_stag[..., :-1, :]+v_stag[..., 1:, :])


def shear_at_hours(path: str, hours, zl, zu, ri, ro):
    rows = []
    with Dataset(path) as ds:
        t = np.asarray(ds["time"][:], float)/3600
        x = np.asarray(ds["xh"][:], float); y = np.asarray(ds["yh"][:], float)
        z = np.asarray(ds["zh"][:], float)
        il = int(np.argmin(abs(z-zl))); iu = int(np.argmin(abs(z-zu)))
        for hour in hours:
            it = int(np.argmin(abs(t-hour)))
            ps = gaussian_filter(np.asarray(ds["psfc"][it], float), 2.0, mode="nearest")
            iyc, ixc = np.unravel_index(np.nanargmin(ps), ps.shape)
            xc, yc = float(x[ixc]), float(y[iyc])
            ix = np.flatnonzero(abs(x-xc) <= ro); iy = np.flatnonzero(abs(y-yc) <= ro)
            i0,i1 = int(ix[0]),int(ix[-1]+1); j0,j1 = int(iy[0]),int(iy[-1]+1)
            levels = []
            for iz in (il,iu):
                us = np.asarray(ds["u"][it,iz,j0:j1,i0:i1+1],float)
                vs = np.asarray(ds["v"][it,iz,j0:j1+1,i0:i1],float)
                u,v = destagger(us,vs)
                xx,yy = np.meshgrid(x[i0:i1]-xc,y[j0:j1]-yc)
                rr = np.hypot(xx,yy); m=(rr>=ri)&(rr<=ro)
                levels.append((float(np.nanmean(u[m])),float(np.nanmean(v[m]))))
            du=levels[1][0]-levels[0][0]; dv=levels[1][1]-levels[0][1]
            rows.append({"hour":float(t[it]),"shear_m_s":float(np.hypot(du,dv)),
                         "shear_u_m_s":du,"shear_v_m_s":dv,
                         "lower_u_m_s":levels[0][0],"lower_v_m_s":levels[0][1],
                         "upper_u_m_s":levels[1][0],"upper_v_m_s":levels[1][1],
                         "center_x_km":xc,"center_y_km":yc,
                         "selected_lower_z_km":float(z[il]),"selected_upper_z_km":float(z[iu])})
    return rows


def wmean(a,w,m):
    good=m&np.isfinite(a)&np.isfinite(w); den=np.sum(w[good])
    return float(np.sum(w[good]*a[good])/den) if den>0 else np.nan


def wrms(a,w,m):
    good=m&np.isfinite(a)&np.isfinite(w); den=np.sum(w[good])
    return float(np.sqrt(np.sum(w[good]*a[good]**2)/den)) if den>0 else np.nan


def factorial_records(folder: str):
    recs=[]
    for npz_path in sorted(Path(folder).glob("full_domain_factorial_*h.npz")):
        json_path=npz_path.with_suffix(".json")
        meta=json.loads(json_path.read_text(encoding="utf-8"))
        d=np.load(npz_path); r=np.asarray(d["r_km"],float); z=np.asarray(d["z_km"],float)
        rr,zz=np.meshgrid(r,z); weight=np.broadcast_to(np.maximum(rr,1.0),rr.shape)
        u=np.asarray(d["u_operator_effect"],float); w=np.asarray(d["w_operator_effect"],float)
        urc=np.asarray(d["ur_ctrl_cm1"],float); urj=np.asarray(d["ur_jet_cm1"],float)
        upper=(rr>=50)&(rr<=300)&(zz>=10)&(zz<=16)&((urc>=2)|(urj>=2))
        low=(rr>=50)&(rr<=300)&(zz>=.5)&(zz<=3)&((urc<=-1)|(urj<=-1))
        if not np.any(upper): upper=(rr>=50)&(rr<=300)&(zz>=10)&(zz<=16)
        if not np.any(low): low=(rr>=50)&(rr<=300)&(zz>=.5)&(zz<=3)
        ascent=(rr>=20)&(rr<=150)&(zz>=2)&(zz<=14)
        block=meta["radial_wind_attribution"]["upper_outflow_union"]
        recs.append({
            "hour":float(meta["hour"]),"r_km":r,"z_km":z,"u_operator":u,"w_operator":w,
            "upper_outflow_u_mean_m_s":wmean(u,weight,upper),
            "upper_outflow_u_rms_m_s":wrms(u,weight,upper),
            "low_inflow_strength_change_m_s":-wmean(u,weight,low),
            "low_inflow_u_rms_m_s":wrms(u,weight,low),
            "core_ascent_w_mean_m_s":wmean(w,weight,ascent),
            "core_ascent_w_rms_m_s":wrms(w,weight,ascent),
            "operator_projection_share_on_total":float(block["projection_share_on_total"]["operator"]),
            "forcing_projection_share_on_total":float(block["projection_share_on_total"]["forcing"]),
            "interaction_projection_share_on_total":float(block["projection_share_on_total"]["interaction"]),
            "forcing_to_operator_rms_ratio":float(block["forcing_to_operator_rms_ratio"]),
            "ctrl_changed_fraction":float(meta["regularization"]["CTRL"]["changed_coefficient_fraction"]),
            "jet_changed_fraction":float(meta["regularization"]["JET"]["changed_coefficient_fraction"]),
        })
    return recs


def interp(t,x,h): return float(np.interp(h,t,x))


def plot_overview(series,shears,fac,out):
    fig,ax=plt.subplots(2,2,figsize=(13.2,8.4),constrained_layout=True)
    for name,(t,p,rate) in series.items():
        ax[0,0].plot(t,p,color=COLORS[name],lw=1.8,label=name)
        ax[0,1].plot(t,rate,color=COLORS[name],lw=1.4,label=name)
    ax[0,0].invert_yaxis(); ax[0,0].set(title="Minimum surface pressure",xlabel="Time (h)",ylabel="Pmin (hPa)")
    ax[0,1].axhline(0,color=".4",lw=.7); ax[0,1].set(title="24-h-equivalent pressure-fall rate",xlabel="Time (h)",ylabel="Intensification (hPa day$^{-1}$)")
    for name,rows in shears.items():
        ax[1,0].plot([x["hour"] for x in rows],[x["shear_m_s"] for x in rows],"o-",color=COLORS[name],lw=1.7,label=name)
    ax[1,0].set(title="Environmental deep-layer vector shear",xlabel="Time (h)",ylabel="2–12 km shear (m s$^{-1}$)")
    for name,recs in fac.items():
        h=[x["hour"] for x in recs]; c=COLORS[name]
        ax[1,1].plot(h,[x["upper_outflow_u_mean_m_s"] for x in recs],"o-",color=c,lw=1.8,label=f"{name}: upper outflow")
        ax[1,1].plot(h,[x["low_inflow_strength_change_m_s"] for x in recs],"s--",color=c,lw=1.3,label=f"{name}: low inflow")
    ax[1,1].axhline(0,color=".35",lw=.8)
    ax[1,1].set(title="Signed operator-only secondary-circulation effect",xlabel="Time (h)",ylabel="Favourable-direction change (m s$^{-1}$)")
    for a in ax.flat: a.grid(alpha=.22); a.legend(frameon=False,fontsize=8)
    fig.suptitle("Intensity, environmental shear, and regularized SE operator effect",fontweight="bold")
    fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig)


def plot_sections(fac,out):
    names=["JET15","JET30"]; hours=[x["hour"] for x in fac["JET15"]]
    allf=[x["u_operator"] for n in names for x in fac[n]]
    vmax=max(float(np.nanpercentile(np.abs(np.concatenate([x.ravel() for x in allf])),98.5)),1e-3)
    cm=LinearSegmentedColormap.from_list("div",["#3C5488","#A8BFDC","#F7F7F4","#F1B39A","#B9363E"])
    fig,axes=plt.subplots(2,len(hours),figsize=(3.15*len(hours),6.2),sharex=True,sharey=True,constrained_layout=True,squeeze=False)
    for i,n in enumerate(names):
        for j,x in enumerate(fac[n]):
            a=axes[i,j]; m=a.pcolormesh(x["r_km"],x["z_km"],x["u_operator"],cmap=cm,norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),shading="auto",rasterized=True)
            use=[q for q in (-.1,-.05,.05,.1) if np.nanmin(x["w_operator"])<=q<=np.nanmax(x["w_operator"])]
            if use:
                cs=a.contour(x["r_km"],x["z_km"],x["w_operator"],levels=use,colors=["#3C5488" if q<0 else "#009E73" for q in use],linewidths=.65)
            a.axhline(10,color=".4",ls=":",lw=.5); a.axhline(16,color=".4",ls=":",lw=.5)
            a.set_title(f"{n} · {x['hour']:.0f} h"); a.set_xlim(0,300); a.set_ylim(0,20)
            if j==0:a.set_ylabel("Height (km)")
            if i==1:a.set_xlabel("Radius (km)")
    cb=fig.colorbar(m,ax=axes,orientation="horizontal",shrink=.72,pad=.035)
    cb.set_label("Operator-only radial-wind effect (m s$^{-1}$); red=more outward, blue=more inward")
    fig.suptitle("Sign of the regularized SE operator effect on secondary circulation\ncontours: operator-only vertical-wind effect",fontweight="bold")
    fig.savefig(out,dpi=280,bbox_inches="tight"); plt.close(fig)


def main():
    a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    paths={"CTRL":a.ctrl,"JET15":a.jet15,"JET30":a.jet30}
    series={k:intensity(v,a.max_time_hours) for k,v in paths.items()}
    shears={k:shear_at_hours(v,a.hours,a.lower_z_km,a.upper_z_km,a.shear_r_inner_km,a.shear_r_outer_km) for k,v in paths.items()}
    fac={"JET15":factorial_records(a.factorial_jet15),"JET30":factorial_records(a.factorial_jet30)}
    rows=[]
    for name in ("JET15","JET30"):
        t,p,rate=series[name]; tc,pc,rc=series["CTRL"]
        smap={round(x["hour"],3):x for x in shears[name]}; scmap={round(x["hour"],3):x for x in shears["CTRL"]}
        for rec in fac[name]:
            h=rec["hour"]; key=round(h,3)
            row={"case":name,"hour":h,"pmin_hpa":interp(t,p,h),"pmin_minus_ctrl_hpa":interp(t,p,h)-interp(tc,pc,h),
                 "intensification_hpa_day":interp(t,rate,h),"intensification_minus_ctrl_hpa_day":interp(t,rate,h)-interp(tc,rc,h),
                 "shear_m_s":smap[key]["shear_m_s"],"shear_minus_ctrl_m_s":smap[key]["shear_m_s"]-scmap[key]["shear_m_s"]}
            row.update({k:v for k,v in rec.items() if k not in ("r_km","z_km","u_operator","w_operator")}); rows.append(row)
    with (out/"shear_operator_intensity_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"definitions":{"positive_upper_outflow":"operator effect makes radial wind more outward",
              "positive_low_inflow":"operator effect makes radial wind more inward",
              "positive_ascent":"operator effect makes vertical wind more upward",
              "shear":"vector difference of annulus-mean Earth-relative winds at nearest 2 and 12 km; r=200-1000 km",
              "causality_warning":"single deterministic members; diagnostic consistency, not independent causal identification"},"records":rows}
    (out/"shear_operator_intensity_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    plot_overview(series,shears,fac,out/"figure1_intensity_shear_operator.png")
    plot_sections(fac,out/"figure2_operator_effect_sections.png")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
