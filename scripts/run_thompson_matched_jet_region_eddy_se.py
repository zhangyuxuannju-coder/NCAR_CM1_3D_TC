#!/usr/bin/env python3
"""Locate matched-state eddy SE responses by RHS source region.

The outer-upper box is a *source-support diagnostic*: it selects the already
assembled SE RHS, so its response adds exactly with the other RHS regions.
It is termed JET-run-induced rather than a pure imposed-jet effect, because
JET-minus-CTRL eddy fluxes include jet--vortex interaction and vortex response.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.se_bui import build_basic_state, build_forcing, invert_balanced_theta, regularize_ellipticity
from src.se_nonuniform import solve_flux_form_dirichlet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", default="output/thompson_240h_se_attribution/matching/strength_matched_pairs.csv")
    p.add_argument("--output-dir", default="output/thompson_240h_se_attribution/se/matched_jet_region_eddy")
    p.add_argument("--eps", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    p.add_argument("--f", type=float, default=6.2e-5)
    p.add_argument("--jet-r-min-km", type=float, default=600.0)
    p.add_argument("--jet-z-min-km", type=float, default=8.0)
    p.add_argument("--jet-z-max-km", type=float, default=18.0)
    return p.parse_args()


def tag(hour: float) -> str:
    return f"t{hour:05.1f}h".replace(".", "p")


def load(case: str, hour: float) -> dict[str, np.ndarray]:
    p = ROOT / "output/thompson_240h_se_attribution/cache" / f"{case}_{tag(hour)}.npz"
    with np.load(p) as a:
        required = ("r_m", "z_m", "vt", "theta", "rho", "F_lambda_eddy", "eddy_kinetic_energy")
        return {k: np.asarray(a[k], float) for k in required}


def operator(state: dict[str, np.ndarray], fcor: float, eps: float) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    r, z = state["r_m"], state["z_m"]
    theta, tw = invert_balanced_theta(state["vt"], state["theta"], r, z, fcor)
    basic = build_basic_state(state["vt"], theta, state["rho"], r, z, fcor)
    k1, k2, k3, reg = regularize_ellipticity(basic["K1_raw"], basic["K2_raw"], basic["K3_raw"], eps_ratio=eps)
    metric = 1.0/(np.maximum(state["rho"], 1e-10)*np.maximum(r[None], .5*np.min(np.diff(r))))
    return {"basic":basic, "a":k1*metric, "b":k2*metric, "c":k3*metric}, {"thermal_wind":tw,"regularization":reg}


def winds(psi: np.ndarray, rho: np.ndarray, r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    den = np.maximum(rho*np.maximum(r[None], .5*np.min(np.diff(r))),1e-10)
    u = -np.gradient(psi,z,axis=0,edge_order=2)/den
    w = np.gradient(psi,r,axis=1,edge_order=2)/den
    u[[0,-1],:]=np.nan;u[:,[0,-1]]=np.nan;w[[0,-1],:]=np.nan;w[:,[0,-1]]=np.nan
    return u,w


def rms(a: np.ndarray, mask: np.ndarray) -> float:
    x=a[mask & np.isfinite(a)]
    return float(np.sqrt(np.mean(x*x))) if x.size else float("nan")


def project(a: np.ndarray, total: np.ndarray, mask: np.ndarray) -> float:
    use=mask & np.isfinite(a) & np.isfinite(total)
    return float(np.sum(a[use]*total[use])/max(np.sum(total[use]**2),1e-30)) if np.any(use) else float("nan")


def metrics(responses: dict[str, tuple[np.ndarray,np.ndarray]], r: np.ndarray, z: np.ndarray) -> dict[str, float]:
    rr,zz=np.meshgrid(r/1000,z/1000)
    masks={"low_inflow":(rr>=20)&(rr<=150)&(zz>=.5)&(zz<=2),"inner_ascent":(rr>=20)&(rr<=150)&(zz>=2)&(zz<=12)}
    out={}
    for domain,mask in masks.items():
        var=1 if domain=="inner_ascent" else 0
        total=responses["full"][var]
        for name,fields in responses.items():
            x=fields[var]
            out[f"{domain}_{name}_rms"]=rms(x,mask)
            if name!="full": out[f"{domain}_{name}_projection_on_full"]=project(x,total,mask)
    return out


def plot(jh: float,ch: float,r:np.ndarray,z:np.ndarray,fenv:np.ndarray,rhs:np.ndarray,masks:dict[str,np.ndarray],resp:dict[str,tuple[np.ndarray,np.ndarray]],path:Path)->None:
    fields=(fenv,rhs,np.where(masks["jet_upper"],rhs,np.nan),resp["full"][0],resp["jet_upper"][0],resp["core"][0],resp["full"][1],resp["jet_upper"][1],resp["core"][1])
    titles=(r"matched $F_{\lambda,eddy}^{JET}-F_{\lambda,eddy}^{CTRL}$",r"all eddy RHS",r"outer-upper RHS support",r"full $\Delta u_r$",r"outer-upper contribution",r"core contribution",r"full $\Delta w$",r"outer-upper contribution",r"core contribution")
    fig,axs=plt.subplots(3,3,figsize=(14,11),constrained_layout=True,sharex=True,sharey=True)
    for ax,x,title in zip(axs.flat,fields,titles):
        finite=np.abs(x[np.isfinite(x)]); vmax=max(float(np.percentile(finite,99)) if finite.size else 0.,1e-30)
        im=ax.contourf(r/1000,z/1000,x,levels=np.linspace(-vmax,vmax,25),cmap="RdBu_r",extend="both")
        ax.set(title=title,xlim=(0,1200),ylim=(0,24),xlabel="Radius (km)",ylabel="Height (km)")
        fig.colorbar(im,ax=ax,pad=.02)
    fig.suptitle(f"Strength-matched eddy forcing: J{jh:g} h minus C{ch:g} h",fontweight="bold")
    fig.savefig(path,dpi=180);plt.close(fig)


def main() -> None:
    a=parse_args();out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True)
    with (ROOT/a.matches).open() as h: pairs=list(csv.DictReader(h))
    records=[]
    for pair in pairs:
        jh,ch=float(pair["jet_hour"]),float(pair["ctrl_hour"])
        c,j=load("CTRL",ch),load("JET",jh)
        if not (np.array_equal(c["r_m"],j["r_m"]) and np.array_equal(c["z_m"],j["z_m"])):raise ValueError("cache grids differ")
        r,z=c["r_m"],c["z_m"];rr,zz=np.meshgrid(r/1000,z/1000)
        masks={"core":rr<300,"jet_upper":(rr>=a.jet_r_min_km)&(zz>=a.jet_z_min_km)&(zz<=a.jet_z_max_km)}
        for eps in a.eps:
            op,info=operator(c,a.f,eps);zero=np.zeros_like(c["F_lambda_eddy"])
            rhs=build_forcing(op["basic"],zero,j["F_lambda_eddy"]-c["F_lambda_eddy"],r,z)["forcing_total"]
            rhs_parts={"core":rhs*masks["core"],"jet_upper":rhs*masks["jet_upper"]}
            rhs_parts["remaining"]=rhs-rhs_parts["core"]-rhs_parts["jet_upper"]
            solve=lambda x:solve_flux_form_dirichlet(op["a"],op["b"],op["c"],x,r,z)
            sol={name:solve(x) for name,x in {"full":rhs,**rhs_parts}.items()}
            resp={name:winds(x.psi,c["rho"],r,z) for name,x in sol.items()}
            closure=sol["full"].psi-sum((x.psi for n,x in sol.items() if n!="full"),np.zeros_like(sol["full"].psi))
            rec={"jet_hour":jh,"ctrl_hour":ch,"eps_ratio":eps,"file":f"J{jh:g}_C{ch:g}_eps{eps:.0e}.npz","matrix_relative_residual":sol["full"].relative_residual,"rhs_jet_upper_fraction_l2":float(np.linalg.norm(rhs_parts["jet_upper"])/max(np.linalg.norm(rhs),1e-30)),"source_partition_relative_error":float(np.linalg.norm(closure)/max(np.linalg.norm(sol["full"].psi),1e-30)),"regularization":info["regularization"],**metrics(resp,r,z)}
            jet_box = masks["jet_upper"]
            rec["jet_upper_delta_eddy_speed_mean_ms"] = float(np.mean(
                np.sqrt(2*np.maximum(j["eddy_kinetic_energy"], 0))[jet_box]
                - np.sqrt(2*np.maximum(c["eddy_kinetic_energy"], 0))[jet_box]
            ))
            rec["jet_upper_delta_eddy_speed_rms_ms"] = rms(
                np.sqrt(2*np.maximum(j["eddy_kinetic_energy"], 0))
                - np.sqrt(2*np.maximum(c["eddy_kinetic_energy"], 0)), jet_box)
            records.append(rec)
            stem=f"J{jh:05.1f}_C{ch:05.1f}_eps{eps:.0e}".replace(".","p")
            np.savez_compressed(out/f"{stem}.npz",r_m=r,z_m=z,F_lambda_env=j["F_lambda_eddy"]-c["F_lambda_eddy"],rhs_full=rhs,rhs_core=rhs_parts["core"],rhs_jet_upper=rhs_parts["jet_upper"],psi_full=sol["full"].psi,psi_jet_upper=sol["jet_upper"].psi,delta_ur_full=resp["full"][0],delta_ur_jet_upper=resp["jet_upper"][0],delta_w_full=resp["full"][1],delta_w_jet_upper=resp["jet_upper"][1])
            if np.isclose(eps,1e-4):plot(jh,ch,r,z,j["F_lambda_eddy"]-c["F_lambda_eddy"],rhs,masks,resp,out/f"{stem}.png")
            print(json.dumps({"JET":jh,"CTRL":ch,"eps":eps,"outer_rhs_fraction":rec["rhs_jet_upper_fraction_l2"]}),flush=True)
    (out/"summary.json").write_text(json.dumps({"status":"BALANCED_SOURCE_SUPPORT_DIAGNOSTIC","definition":"outer-upper source support is the assembled JET-minus-CTRL eddy SE RHS within r>=600 km and 8<=z<=18 km","causal_scope":"JET-run-induced, includes imposed jet, jet-vortex interaction, and vortex response; not a pure imposed-jet attribution","matches_file":str((ROOT/a.matches).as_posix()),"records":records},indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":main()
