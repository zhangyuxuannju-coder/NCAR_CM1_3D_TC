#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_se_forcing_operator_factorial_full as base


def parse_args():
 p=argparse.ArgumentParser(description="Strength-matched decomposition of JET-minus-CTRL SE forcing")
 p.add_argument("--nojet",required=True);p.add_argument("--jet",required=True);p.add_argument("--output-dir",required=True)
 p.add_argument("--pairs",nargs="+",required=True,help="JET:CTRL")
 p.add_argument("--eps-ratio",type=float,default=1e-4);p.add_argument("--max-r-km",type=float,default=300)
 p.add_argument("--max-z-km",type=float,default=20);p.add_argument("--dr-km",type=float,default=12)
 p.add_argument("--f",type=float,default=6.2e-5);p.add_argument("--elliptic-margin",type=float,default=0)
 p.add_argument("--baroclinic-scale",type=float,default=1)
 return p.parse_args()


def sources(ctrl,jet):
 zero=np.zeros_like(ctrl["Q"])
 additive={
 "Q_eddy_radial":(jet["Q_eddy_radial"]-ctrl["Q_eddy_radial"],zero),
 "Q_eddy_vertical":(jet["Q_eddy_vertical"]-ctrl["Q_eddy_vertical"],zero),
 "Q_diffusion":(jet["Q_diffusion"]-ctrl["Q_diffusion"],zero),
 "Q_diabatic":(jet["Q_diabatic"]-ctrl["Q_diabatic"],zero),
 "Q_other":(jet["Q_other_model"]-ctrl["Q_other_model"],zero),
 "M_eddy_radial":(zero,jet["F_lambda_eddy_radial"]-ctrl["F_lambda_eddy_radial"]),
 "M_eddy_vertical":(zero,jet["F_lambda_eddy_vertical"]-ctrl["F_lambda_eddy_vertical"]),
 "M_diffusion":(zero,jet["F_lambda_diffusion"]-ctrl["F_lambda_diffusion"]),
 "M_other":(zero,jet["F_lambda_other_model"]-ctrl["F_lambda_other_model"]),
 }
 total=(jet["Q"]-ctrl["Q"],jet["Fnu"]-ctrl["Fnu"])
 return additive,total


def stats(field,total,r,z,region,var):
 rr=np.broadcast_to(np.maximum(r[None,:],.5),field.shape);zz=np.broadcast_to(z[:,None],field.shape)
 if region=="bl_inflow": mask=(rr>=50)&(rr<=300)&(zz>=.5)&(zz<=2)
 elif region=="inner_updraft": mask=(rr>=20)&(rr<=150)&(zz>=2)&(zz<=12)
 else: mask=(rr>=100)&(rr<=300)&(zz>=10)&(zz<=17)
 return {
  "mean":float(np.nanmean(field[mask])),
  "rms":base.weighted_rms(field,rr,mask),
  "share":base.projection_share(field,total,rr,mask),
 }


def solve_pair(jh,ch,args,out):
 print(f"PAIR J{jh:g}/C{ch:g}",flush=True)
 ctrl=base.azimuthal_average(args.nojet,ch,args);jet=base.azimuthal_average(args.jet,jh,args);base.check_grids(ctrl,jet)
 r=np.asarray(ctrl["r_km"],float);z=np.asarray(ctrl["z_km"],float);rm=r*1000;zm=z*1000
 case=base.build_case(ctrl,rm,zm,args)
 parts,total_src=sources(ctrl,jet)
 qsum=sum((q for q,f in parts.values()),np.zeros_like(ctrl["Q"]))
 fsum=sum((f for q,f in parts.values()),np.zeros_like(ctrl["Fnu"]))
 qclosure=float(np.linalg.norm(qsum-total_src[0])/max(np.linalg.norm(total_src[0]),1e-30))
 fclosure=float(np.linalg.norm(fsum-total_src[1])/max(np.linalg.norm(total_src[1]),1e-30))
 total=base.solve_one(case,total_src[0],total_src[1],rm,zm)
 component={name:base.solve_one(case,q,f,rm,zm) for name,(q,f) in parts.items()}
 uclosure=sum((v["u"] for v in component.values()),np.zeros_like(total["u"]))-total["u"]
 wclosure=sum((v["w"] for v in component.values()),np.zeros_like(total["w"]))-total["w"]
 rows=[]
 for name,res in component.items():
  row={"jet_hour":jh,"ctrl_hour":ch,"component":name,"q_source_closure":qclosure,"f_source_closure":fclosure,
       "u_response_closure":float(np.linalg.norm(uclosure)/max(np.linalg.norm(total["u"]),1e-30)),
       "w_response_closure":float(np.linalg.norm(wclosure)/max(np.linalg.norm(total["w"]),1e-30)),
       "rhs_rms":float(np.sqrt(np.nanmean(res["forcing_total"]**2)))}
  for reg,var in (("bl_inflow","u"),("inner_updraft","w"),("upper_outflow","u")):
   s=stats(res[var],total[var],r,z,reg,var)
   for k,v in s.items():row[f"{reg}_{k}"]=v
  rows.append(row)
 tag=f"J{jh:05.1f}_C{ch:05.1f}".replace(".","p")
 np.savez_compressed(out/f"{tag}_forcing_components.npz",r_km=r,z_km=z,total_u=total["u"],total_w=total["w"],
   **{f"{name}_u":res["u"] for name,res in component.items()},**{f"{name}_w":res["w"] for name,res in component.items()},
   **{f"{name}_rhs":res["forcing_total"] for name,res in component.items()})
 plot_pair(component,total,r,z,jh,ch,out/f"{tag}_forcing_component_responses.png")
 return rows


def plot_pair(comp,total,r,z,jh,ch,path):
 names=list(comp);fig,axes=plt.subplots(3,len(names),figsize=(2.5*len(names),8),sharex=True,sharey=True,constrained_layout=True)
 for row,(reg,var) in enumerate((("BL inflow","u"),("Inner ascent","w"),("Upper response","u"))):
  fields=[comp[n][var] for n in names];vmax=max(np.nanpercentile(np.abs(x),98.5) for x in fields)
  for col,(n,fld) in enumerate(zip(names,fields)):
   im=axes[row,col].pcolormesh(r,z,fld,cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto")
   if row==0:axes[row,col].set_title(n.replace("_","\n"),fontsize=8)
   if col==0:axes[row,col].set_ylabel(reg+"\nHeight (km)")
   if row==2:axes[row,col].set_xlabel("Radius (km)")
  fig.colorbar(im,ax=axes[row,:],label=("Delta u" if var=="u" else "Delta w")+" (m s-1)",fraction=.015)
 fig.suptitle(f"JET forcing-component response on matched CTRL operator: J{jh:g}/C{ch:g}")
 fig.savefig(path,dpi=190,bbox_inches="tight");plt.close(fig)


def summary_plots(df,out):
 comps=list(df.component.drop_duplicates());pairs=df[["jet_hour","ctrl_hour"]].drop_duplicates()
 labels=[f"J{j:g}/C{c:g}" for j,c in zip(pairs.jet_hour,pairs.ctrl_hour)]
 for metric,title in (("share","Signed projection on total forcing response"),("rms","Response RMS")):
  fig,axes=plt.subplots(1,3,figsize=(15,6),constrained_layout=True)
  for ax,reg,rt in zip(axes,("bl_inflow","inner_updraft","upper_outflow"),("BL inflow","Inner ascent","Upper outflow")):
   a=np.empty((len(comps),len(labels)))
   for i,c in enumerate(comps):
    q=df[df.component==c]
    a[i]=q[f"{reg}_{metric}"].to_numpy()
   lim=max(abs(np.nanmin(a)),abs(np.nanmax(a))) if metric=="share" else None
   im=ax.imshow(a,aspect="auto",cmap="RdBu_r" if metric=="share" else "magma",vmin=-lim if metric=="share" else 0,vmax=lim)
   ax.set_xticks(range(len(labels)),labels,rotation=35,ha="right");ax.set_yticks(range(len(comps)),comps)
   ax.set_title(rt);fig.colorbar(im,ax=ax,shrink=.75)
  fig.suptitle(title)
  fig.savefig(out/f"forcing_components_{metric}_heatmap.png",dpi=220,bbox_inches="tight");plt.close(fig)
 q=df.groupby("component",sort=False)[["bl_inflow_share","inner_updraft_share","upper_outflow_share"]].mean()
 q.to_csv(out/"forcing_component_mean_shares.csv")


def main():
 args=parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for item in args.pairs:
  j,c=map(float,item.split(":"));rows.extend(solve_pair(j,c,args,out))
 df=pd.DataFrame(rows);df.to_csv(out/"strength_matched_forcing_components.csv",index=False)
 summary_plots(df,out)
 manifest={"pairs":args.pairs,"eps_ratio":args.eps_ratio,"operator":"matched CTRL","components":["Q_eddy_radial","Q_eddy_vertical","Q_diffusion","Q_diabatic","Q_other","M_eddy_radial","M_eddy_vertical","M_diffusion","M_other"],"closure":"components are additive and exclude hadv/vadv double counting"}
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 print(df.groupby("component")[["bl_inflow_share","inner_updraft_share","upper_outflow_share"]].mean().to_json(indent=2))


if __name__=="__main__":main()
