#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import run_se_forcing_operator_factorial_full as base


def args_parse():
    p=argparse.ArgumentParser(description="Strength-matched CTRL/JET full-operator SE factorial")
    p.add_argument("--nojet",required=True); p.add_argument("--jet",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--pairs",nargs="+",required=True,help="JET:CTRL hour pairs")
    p.add_argument("--eps",nargs="+",type=float,default=[1e-5,1e-4,1e-3])
    p.add_argument("--max-r-km",type=float,default=300.0)
    p.add_argument("--max-z-km",type=float,default=20.0)
    p.add_argument("--dr-km",type=float,default=12.0)
    p.add_argument("--f",type=float,default=6.2e-5)
    p.add_argument("--elliptic-margin",type=float,default=0.0)
    p.add_argument("--baroclinic-scale",type=float,default=1.0)
    return p.parse_args()


def region_metrics(effects,r,z):
    rr=np.broadcast_to(np.maximum(r[None,:],0.5),effects["u"]["total"].shape)
    zz=np.broadcast_to(z[:,None],effects["u"]["total"].shape)
    masks={
      "bl_inflow":(rr>=50)&(rr<=300)&(zz>=0.5)&(zz<=2.0),
      "inner_updraft":(rr>=20)&(rr<=150)&(zz>=2)&(zz<=12),
      "upper_outflow":(rr>=100)&(rr<=300)&(zz>=10)&(zz<=17),
    }
    out={}
    for reg,mask in masks.items():
      var="w" if reg=="inner_updraft" else "u"
      total=effects[var]["total"]
      for comp in ("forcing","operator","interaction","total"):
        arr=effects[var][comp]
        out[f"{reg}_{comp}_mean"]=float(np.nanmean(arr[mask]))
        out[f"{reg}_{comp}_rms"]=base.weighted_rms(arr,rr,mask)
        if comp!="total":
          out[f"{reg}_{comp}_share"]=base.projection_share(arr,total,rr,mask)
      out[f"{reg}_forcing_operator_ratio"]=out[f"{reg}_forcing_rms"]/max(out[f"{reg}_operator_rms"],1e-30)
    return out


def solve_pair(jh,ch,args,outdir):
    print(f"PAIR JET {jh:g} h / CTRL {ch:g} h",flush=True)
    ctrl=base.azimuthal_average(args.nojet,ch,args)
    jet=base.azimuthal_average(args.jet,jh,args)
    base.check_grids(ctrl,jet)
    r=np.asarray(ctrl["r_km"],float); z=np.asarray(ctrl["z_km"],float)
    rm=r*1000.; zm=z*1000.
    pair_rows=[]
    for eps in args.eps:
      args.eps_ratio=float(eps)
      cases={"CTRL":base.build_case(ctrl,rm,zm,args),"JET":base.build_case(jet,rm,zm,args)}
      sources={"CTRL":(np.asarray(ctrl["Q"]),np.asarray(ctrl["Fnu"])),"JET":(np.asarray(jet["Q"]),np.asarray(jet["Fnu"]))}
      cells={}
      for op in ("CTRL","JET"):
        for forc in ("CTRL","JET"):
          q,fl=sources[forc]
          cells[op[0]+forc[0]]=base.solve_one(cases[op],q,fl,rm,zm)
      effects={}
      for var in ("psi","u","w"):
        cc,cj,jc,jj=(cells[k][var] for k in ("CC","CJ","JC","JJ"))
        effects[var]={
          "forcing":cj-cc,
          "operator":jc-cc,
          "interaction":jj-jc-cj+cc,
          "total":jj-cc,
        }
      row={"jet_hour":jh,"ctrl_hour":ch,"eps":eps}
      row.update(region_metrics(effects,r,z))
      mb=base.metric_block(effects["u"],ctrl,jet,r,z)
      for domain,val in mb.items():
        row[f"{domain}_forcing_operator_ratio"]=val["forcing_to_operator_rms_ratio"]
        for comp,share in val["projection_share_on_total"].items():
          row[f"{domain}_{comp}_share"]=share
      for name in ("CTRL","JET"):
        row[f"{name.lower()}_nonelliptic_fraction"]=float(np.mean(cases[name]["raw_nonelliptic"]))
        row[f"{name.lower()}_changed_fraction"]=float(np.mean(cases[name]["changed"]))
      pair_rows.append(row)
      tag=f"J{jh:05.1f}_C{ch:05.1f}_eps{eps:.0e}".replace(".","p").replace("-","m")
      arrays={"r_km":r,"z_km":z}
      for k,v in cells.items():
        for var in ("psi","u","w"): arrays[f"{var}_{k}"]=v[var]
      for var,parts in effects.items():
        for k,v in parts.items(): arrays[f"{var}_{k}_effect"]=v
      np.savez_compressed(outdir/f"{tag}.npz",**arrays)
      (outdir/f"{tag}.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
      if abs(eps-1e-4)<1e-12:
        plot_maps(effects,r,z,jh,ch,outdir/f"{tag}_responses.png")
    return pair_rows


def plot_maps(effects,r,z,jh,ch,path):
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True,constrained_layout=True)
    parts=("forcing","operator","interaction","total")
    for row,var in enumerate(("u","w")):
      vmax=max(np.nanpercentile(np.abs(effects[var][p]),98.5) for p in parts)
      norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
      for col,p in enumerate(parts):
        im=axes[row,col].pcolormesh(r,z,effects[var][p],cmap="RdBu_r",norm=norm,shading="auto")
        axes[row,col].set_title(p.capitalize())
        axes[row,col].set_xlabel("Radius (km)")
        if col==0: axes[row,col].set_ylabel(("Radial wind" if var=="u" else "Vertical velocity")+"\nHeight (km)")
      fig.colorbar(im,ax=axes[row,:],label=("Delta u" if var=="u" else "Delta w")+" (m s-1)")
    fig.suptitle(f"Strength-matched SE: JET {jh:g} h versus CTRL {ch:g} h; eps=1e-4")
    fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)


def summary_plots(df,outdir):
    unique_pairs=df[["jet_hour","ctrl_hour"]].drop_duplicates()
    pairs=[f"J{j:g}/C{c:g}" for j,c in zip(unique_pairs.jet_hour,unique_pairs.ctrl_hour)]
    epsvals=sorted(df.eps.unique())
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    regions=("bl_inflow","inner_updraft","upper_outflow")
    titles=("Boundary-layer inflow","Inner-core ascent","Upper outflow")
    for ax,reg,title in zip(axes,regions,titles):
      for eps in epsvals:
        q=df[df.eps==eps]
        ax.plot(range(len(q)),q[f"{reg}_forcing_operator_ratio"],marker="o",label=f"eps={eps:g}")
      ax.axhline(1.2,color="0.4",ls="--",lw=.8); ax.axhline(1/1.2,color="0.4",ls=":",lw=.8)
      ax.set_xticks(range(len(pairs)),pairs,rotation=35,ha="right"); ax.set_yscale("log")
      ax.set_title(title); ax.set_ylabel("Forcing / operator RMS"); ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Strength-matched SE dominance (>1 forcing; <1 operator)")
    fig.savefig(outdir/"strength_matched_forcing_operator_ratio.png",dpi=230,bbox_inches="tight"); plt.close(fig)

    q=df[np.isclose(df.eps,1e-4)]
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    x=np.arange(len(q)); width=.25
    for ax,reg,title in zip(axes,regions,titles):
      for off,comp,color in ((-width,"forcing","#D55E00"),(0,"operator","#0072B2"),(width,"interaction","#009E73")):
        ax.bar(x+off,q[f"{reg}_{comp}_share"],width,label=comp,color=color)
      ax.axhline(0,color=".25",lw=.7); ax.axhline(1,color=".5",lw=.7,ls=":")
      ax.set_xticks(x,pairs,rotation=35,ha="right"); ax.set_title(title)
      ax.set_ylabel("Signed projection on total"); ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Strength-matched signed attribution, eps=1e-4 (terms sum to 1)")
    fig.savefig(outdir/"strength_matched_signed_attribution.png",dpi=230,bbox_inches="tight"); plt.close(fig)


def main():
    args=args_parse(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    pairs=[]
    for item in args.pairs:
      j,c=item.split(":"); pairs.append((float(j),float(c)))
    rows=[]
    for j,c in pairs: rows.extend(solve_pair(j,c,args,out))
    df=pd.DataFrame(rows); df.to_csv(out/"strength_matched_se_attribution.csv",index=False)
    summary_plots(df,out)
    manifest={"definition":{"forcing":"CJ-CC","operator":"JC-CC","interaction":"JJ-JC-CJ+CC","total":"JJ-CC"},"pairs":[{"jet_hour":j,"ctrl_hour":c} for j,c in pairs],"eps":args.eps,"interpretation":"regularized balanced projection; strength matching does not itself identify exclusive causality"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(df.to_json(orient="records",indent=2))


if __name__=="__main__": main()
