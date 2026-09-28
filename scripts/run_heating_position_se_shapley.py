#!/usr/bin/env python3
"""Shapley attribution of diabatic-heating amplitude, location and shape in SE response."""
from __future__ import annotations
import argparse,json,math
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.ndimage import shift
import run_se_forcing_operator_factorial_full as base

FACTORS=("amplitude","location","shape")
REGIONS={"bl_inflow":("u",(50,300,.5,2)),"inner_updraft":("w",(20,150,2,12)),"upper_outflow":("u",(100,300,10,17))}

def args_parser():
 p=argparse.ArgumentParser();p.add_argument("--nojet",required=True);p.add_argument("--jet",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--pairs",nargs="+",required=True)
 p.add_argument("--eps-ratio",type=float,default=1e-4);p.add_argument("--max-r-km",type=float,default=300);p.add_argument("--max-z-km",type=float,default=20);p.add_argument("--dr-km",type=float,default=12);p.add_argument("--f",type=float,default=6.2e-5);p.add_argument("--elliptic-margin",type=float,default=0);p.add_argument("--baroclinic-scale",type=float,default=1)
 return p.parse_args()

def amp_centroid(q,r,z):
 w=np.maximum(q,0)*r[None];a=float(np.sum(w));return a,float(np.sum(w*r[None])/a),float(np.sum(w*z[:,None])/a)

def shifted(q,dsz,dsr): return shift(q,(dsz,dsr),order=1,mode="constant",cval=0,prefilter=False)

def representation(qc,qj,r,z):
 ac,rc,zc=amp_centroid(qc,r,z);aj,rj,zj=amp_centroid(qj,r,z);dr=float(np.mean(np.diff(r)));dz=float(np.mean(np.diff(z)));rr=.5*(rc+rj);zz=.5*(zc+zj)
 sc=shifted(qc/ac,(zz-zc)/dz,(rr-rc)/dr);sj=shifted(qj/aj,(zz-zj)/dz,(rr-rj)/dr)
 def build(use_a,use_l,use_s):
  # Preserve the observed endpoints exactly so the Shapley sum closes to JET minus CTRL.
  if not (use_a or use_l or use_s): return qc.copy()
  if use_a and use_l and use_s: return qj.copy()
  a=aj if use_a else ac; rt,zt=(rj,zj) if use_l else (rc,zc);s=sj if use_s else sc
  x=shifted(s,(zt-zz)/dz,(rt-rr)/dr);norm=float(np.sum(np.maximum(x,0)*r[None]));return a*x/max(norm,1e-30)
 return build,dict(ctrl_amplitude=ac,jet_amplitude=aj,ctrl_r=rc,jet_r=rj,ctrl_z=zc,jet_z=zj)

def shapley(values,var,factor):
 i=FACTORS.index(factor);others=[x for x in range(3) if x!=i];out=np.zeros_like(values[0][var])
 for k in range(3):
  for ss in combinations(others,k):
   mask=sum(1<<x for x in ss);w=math.factorial(k)*math.factorial(2-k)/math.factorial(3);out+=w*(values[mask|(1<<i)][var]-values[mask][var])
 return out

def mask_for(r,z,b):
 rr,zz=np.meshgrid(r,z);return (rr>=b[0])&(rr<=b[1])&(zz>=b[2])&(zz<=b[3])

def projection(x,y,r,m):
 rr=np.broadcast_to(r[None],x.shape);d=np.sum(rr[m]*y[m]**2);return float(np.sum(rr[m]*x[m]*y[m])/d) if d>0 else np.nan

def wrms(x,r,m):
 rr=np.broadcast_to(r[None],x.shape);return float(np.sqrt(np.sum(rr[m]*x[m]**2)/np.sum(rr[m])))

def plot_pair(effects,total,r,z,title,path):
 names=list(FACTORS)+["total"];fig,ax=plt.subplots(2,4,figsize=(13,6.3),sharex=True,sharey=True,constrained_layout=True)
 for row,var in enumerate(("u","w")):
  fields=[effects[n][var] for n in FACTORS]+[total[var]];vm=max(np.nanpercentile(abs(x),98.5) for x in fields)
  for col,(name,f) in enumerate(zip(names,fields)):
   im=ax[row,col].pcolormesh(r,z,f,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-vm,vcenter=0,vmax=vm),shading="auto");ax[row,col].set_title(name);ax[row,col].set(xlim=(0,300),ylim=(0,20))
   if row==1:ax[row,col].set_xlabel("Radius (km)")
   if col==0:ax[row,col].set_ylabel(("Delta u" if var=="u" else "Delta w")+" (m s-1)\nHeight (km)")
  fig.colorbar(im,ax=ax[row],shrink=.75)
 fig.suptitle(title);fig.savefig(path.with_suffix(".png"),dpi=240,bbox_inches="tight");fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight");plt.close(fig)

def one(jh,ch,a,out):
 c=base.azimuthal_average(a.nojet,ch,a);j=base.azimuthal_average(a.jet,jh,a);base.check_grids(c,j);r=np.asarray(c["r_km"]);z=np.asarray(c["z_km"]);rm=r*1000;zm=z*1000;case=base.build_case(c,rm,zm,a);zero=np.zeros_like(c["Q"])
 build,geo=representation(c["Q_diabatic"],j["Q_diabatic"],r,z);vals={}
 for mask in range(8):vals[mask]=base.solve_one(case,build(bool(mask&1),bool(mask&2),bool(mask&4)),zero,rm,zm)
 effects={f:{v:shapley(vals,v,f) for v in ("u","w")} for f in FACTORS};total=base.solve_one(case,j["Q_diabatic"]-c["Q_diabatic"],zero,rm,zm);obs={"u":j["ur"]-c["ur"],"w":j["w"]-c["w"]}
 closure=max(np.linalg.norm(sum(effects[f][v] for f in FACTORS)-total[v])/max(np.linalg.norm(total[v]),1e-30) for v in ("u","w"));rows=[]
 for reg,(var,b) in REGIONS.items():
  m=mask_for(r,z,b)
  for name,field in [(x,effects[x][var]) for x in FACTORS]+[("total",total[var])]:rows.append(dict(jet_hour=jh,ctrl_hour=ch,region=reg,variable=var,effect=name,mean_m_s=float(np.mean(field[m])),rms_m_s=wrms(field,r,m),share_on_total=projection(field,total[var],r,m),projection_on_observed=projection(field,obs[var],r,m),closure=closure,**geo))
 tag=f"J{jh:03.0f}_C{ch:03.0f}";plot_pair(effects,total,r,z,f"Heating-field Shapley response: {tag}",out/"figures"/(tag+"_heating_shapley"));np.savez_compressed(out/"products"/(tag+"_heating_shapley.npz"),r_km=r,z_km=z,total_u=total["u"],total_w=total["w"],**{f"{n}_{v}":effects[n][v] for n in FACTORS for v in ("u","w")})
 return rows

def main():
 a=args_parser();out=Path(a.output_dir);(out/"figures").mkdir(parents=True,exist_ok=True);(out/"products").mkdir(exist_ok=True);rows=[]
 for item in a.pairs:
  j,c=map(float,item.split(":"));rows+=one(j,c,a,out)
 df=pd.DataFrame(rows);df.to_csv(out/"products"/"heating_position_shapley_metrics.csv",index=False);df.groupby(["region","effect"])[["share_on_total","projection_on_observed"]].mean().to_csv(out/"products"/"heating_position_shapley_mean.csv")
 (out/"shapley_manifest.json").write_text(json.dumps({"factors":FACTORS,"operator":"matched CTRL","pairs":a.pairs,"eps_ratio":a.eps_ratio,"normalization":"positive cylindrical integral; full signed field retained"},indent=2),encoding="utf-8")
 print(df.groupby(["region","effect"])[["share_on_total","projection_on_observed"]].mean().to_string())
if __name__=="__main__":main()
