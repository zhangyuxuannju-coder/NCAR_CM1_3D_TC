#!/usr/bin/env python3
from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

SRC="/data/zhangyx/DATA/cm1out_25N_nojet.nc"
OUT=Path("output/ctrl_eyewall_cycle_25N_66_88h")
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"figures").mkdir(exist_ok=True)
(OUT/"products").mkdir(exist_ok=True)
RMAX=180.; DR=3.; SNAP={68,70,72,74,76,78,80,82,84}
edges=np.arange(0,RMAX+DR+.01,DR); rad=(edges[:-1]+edges[1:])/2; nr=len(rad)

def rmean(a,ib,valid):
    a=np.asarray(a,float).ravel(); b=ib.ravel(); m=valid.ravel()&np.isfinite(a)
    s=np.bincount(b[m],weights=a[m],minlength=nr); n=np.bincount(b[m],minlength=nr)
    o=np.full(nr,np.nan); np.divide(s,n,out=o,where=n>0); return o

with Dataset(SRC) as ds:
    ha=np.asarray(ds["time"][:],float)/3600
    its=np.where((ha>=66-.01)&(ha<=88+.01))[0]
    x=np.asarray(ds["xh"][:]); y=np.asarray(ds["yh"][:])
    zh=np.asarray(ds["zh"][:]); zf=np.asarray(ds["zf"][:])
    izu=int(np.argmin(abs(zh-1.2))); izr=int(np.argmin(abs(zh-2.0))); izw=int(np.argmin(abs(zf-5.0)))
    nt=len(its); vt=np.full((nt,nr),np.nan); rz=np.full((nt,nr),np.nan)
    cov=np.full((nt,nr),np.nan); wp=np.full((nt,nr),np.nan)
    pmin=np.full(nt,np.nan); vmax=np.full(nt,np.nan); rmw=np.full(nt,np.nan); snaps=[]
    for j,it in enumerate(its):
        ps=np.asarray(ds["psfc"][it],float); sm=gaussian_filter(ps,2,mode="nearest")
        iy,ix=np.unravel_index(np.nanargmin(sm),sm.shape); xc,yc=x[ix],y[iy]
        pmin[j]=np.nanmin(ps)/100
        pad=int(np.ceil(RMAX/np.median(np.diff(x))))+3
        x0,x1=max(0,ix-pad),min(len(x),ix+pad+1); y0,y1=max(0,iy-pad),min(len(y),iy+pad+1)
        xx,yy=np.meshgrid(x[x0:x1],y[y0:y1]); dx=xx-xc; dy=yy-yc
        rr=np.hypot(dx,dy); aa=np.arctan2(dy,dx); ib=np.floor(rr/DR).astype(int)
        valid=(rr<RMAX)&(ib>=0)&(ib<nr)
        us=np.asarray(ds["u"][it,izu,y0:y1,x0:x1+1],float)
        vs=np.asarray(ds["v"][it,izu,y0:y1+1,x0:x1],float)
        u=(us[:,:-1]+us[:,1:])/2; v=(vs[:-1]+vs[1:])/2
        vt[j]=rmean(-u*np.sin(aa)+v*np.cos(aa),ib,valid)
        vmax[j]=np.nanmax(np.hypot(u,v)[valid])
        sv=gaussian_filter1d(vt[j],.8,mode="nearest"); mm=(rad>=6)&(rad<=120)
        rmw[j]=rad[mm][np.nanargmax(sv[mm])]
        d=np.asarray(ds["dbz"][it,izr,y0:y1,x0:x1],float)
        rz[j]=10*np.log10(np.maximum(rmean(10**(d/10),ib,valid),1e-6))
        cov[j]=rmean((d>=30).astype(float),ib,valid)
        w=np.asarray(ds["w"][it,izw,y0:y1,x0:x1],float); wp[j]=rmean(np.maximum(w,0),ib,valid)
        if round(float(ha[it])) in SNAP:
            cref=np.asarray(ds["cref"][it,y0:y1,x0:x1],float)
            nz=np.where(zh<=18)[0]; db3=np.full((len(nz),nr),np.nan); ww3=np.full_like(db3,np.nan)
            for k,iz in enumerate(nz):
                dd=np.asarray(ds["dbz"][it,iz,y0:y1,x0:x1],float)
                db3[k]=10*np.log10(np.maximum(rmean(10**(dd/10),ib,valid),1e-6))
                ww=np.asarray(ds["w"][it,min(iz,len(zf)-1),y0:y1,x0:x1],float)
                ww3[k]=rmean(ww,ib,valid)
            snaps.append((float(ha[it]),xx-xc,yy-yc,cref,zh[nz],db3,ww3))
        print(f"{ha[it]:.0f} h {j+1}/{nt}",flush=True)

hours=ha[its]
rows=[]
for j,h in enumerate(hours):
    sv=gaussian_filter1d(vt[j],1,mode="nearest"); sd=gaussian_filter1d(rz[j],1,mode="nearest")
    pv,_=find_peaks(np.nan_to_num(sv,nan=-999),prominence=2,distance=6)
    pd,_=find_peaks(np.nan_to_num(sd,nan=-999),prominence=3,distance=6)
    pv=pv[(rad[pv]>=9)&(rad[pv]<=150)]; pd=pd[(rad[pd]>=9)&(rad[pd]<=150)]
    pv=pv[np.argsort(sv[pv])[::-1]][:2]; pd=pd[np.argsort(sd[pd])[::-1]][:2]
    rows.append([h,pmin[j],vmax[j],rmw[j],*(list(rad[pv])+[np.nan]*2)[:2],*(list(rad[pd])+[np.nan]*2)[:2],np.nanmax(cov[j])])
with open(OUT/"products/metrics.csv","w",newline="") as f:
    q=csv.writer(f); q.writerow(["time_h","pmin_hpa","vmax_ms","rmw_km","vt_peak1_km","vt_peak2_km","dbz_peak1_km","dbz_peak2_km","max_dbz30_coverage"]);q.writerows(rows)
np.savez_compressed(OUT/"products/radial_fields.npz",hours=hours,radius_km=rad,vt_ms=vt,dbz_2km=rz,dbz30_coverage=cov,w5_positive_ms=wp,pmin_hpa=pmin,vmax_ms=vmax,rmw_km=rmw)

fig,a=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
a[0,0].plot(hours,pmin,"ko-");a[0,0].invert_yaxis();a[0,0].set_ylabel("Minimum pressure (hPa)")
a[0,1].plot(hours,vmax,"r-o",label="Vmax m/s");a[0,1].plot(hours,rmw,"b-o",label="RMW km");a[0,1].legend()
m=a[1,0].contourf(hours,rad,vt.T,np.arange(0,61,5),cmap="turbo",extend="max");a[1,0].plot(hours,rmw,"w-",lw=2);fig.colorbar(m,ax=a[1,0],label="Tangential wind (m/s)")
m=a[1,1].contourf(hours,rad,rz.T,np.arange(0,56,5),cmap="YlGnBu",extend="both");fig.colorbar(m,ax=a[1,1],label="2-km reflectivity (dBZ)")
m=a[2,0].contourf(hours,rad,cov.T,np.linspace(0,1,11),cmap="viridis");fig.colorbar(m,ax=a[2,0],label="Azimuthal fraction dBZ >= 30")
m=a[2,1].contourf(hours,rad,wp.T,np.linspace(0,2,17),cmap="magma",extend="max");fig.colorbar(m,ax=a[2,1],label="Mean positive w at 5 km (m/s)")
for xax in a.flat:
    xax.axvspan(72,80,color="crimson",alpha=.08);xax.grid(alpha=.25)
for xax in a[1:].flat:xax.set_ylabel("Radius (km)")
a[2,0].set_xlabel("Time (h)");a[2,1].set_xlabel("Time (h)")
a[1,0].set_title("Low-level tangential wind");a[1,1].set_title("Eyewall reflectivity")
a[2,0].set_title("Eyewall completeness");a[2,1].set_title("Convective updraft proxy")
fig.suptitle("25N CTRL: 72-80 h intensity interruption",fontsize=16,fontweight="bold")
fig.savefig(OUT/"figures/overview.png",dpi=220);plt.close(fig)

n=len(snaps); rowsn=int(np.ceil(n/3))
fig,axs=plt.subplots(rowsn,3,figsize=(12,3.8*rowsn),constrained_layout=True);axs=np.ravel(axs)
for ax,s in zip(axs,snaps):
    h,xx,yy,cref,z,db3,ww3=s;m=ax.contourf(xx,yy,cref,np.arange(0,61,5),cmap="turbo",extend="max")
    ax.contour(xx,yy,cref,[20,30,40],colors="k",linewidths=.4);ax.plot(0,0,"rx");ax.set_aspect("equal");ax.set_title(f"{h:.0f} h");ax.set_xlim(-RMAX,RMAX);ax.set_ylim(-RMAX,RMAX)
for ax in axs[n:]:ax.axis("off")
fig.colorbar(m,ax=axs.tolist(),shrink=.7,label="Composite reflectivity (dBZ)")
fig.suptitle("25N CTRL horizontal eyewall morphology",fontsize=15,fontweight="bold")
fig.savefig(OUT/"figures/horizontal_cref.png",dpi=220);plt.close(fig)

fig,axs=plt.subplots(rowsn,3,figsize=(12,3.5*rowsn),constrained_layout=True);axs=np.ravel(axs)
for ax,s in zip(axs,snaps):
    h,xx,yy,cref,z,db3,ww3=s;m=ax.contourf(rad,z,db3,np.arange(0,56,5),cmap="YlGnBu",extend="both")
    cs=ax.contour(rad,z,ww3,[-1,-.5,.5,1,2],colors=["#225ea8","#41b6c4","#fd8d3c","#e31a1c","#800026"],linewidths=.7)
    ax.clabel(cs,fmt="%g",fontsize=6);ax.set_title(f"{h:.0f} h");ax.set_xlim(0,RMAX);ax.set_ylim(0,18);ax.set_xlabel("Radius (km)");ax.set_ylabel("Height (km)")
for ax in axs[n:]:ax.axis("off")
fig.colorbar(m,ax=axs.tolist(),shrink=.7,label="Azimuthal-mean reflectivity (dBZ)")
fig.suptitle("Radius-height eyewall structure; contours: mean w (m/s)",fontsize=14,fontweight="bold")
fig.savefig(OUT/"figures/radius_height.png",dpi=220);plt.close(fig)
json.dump({"source":SRC,"times":[66,88],"snapshots":[x[0] for x in snaps]},open(OUT/"manifest.json","w"),indent=2)
print(OUT,flush=True)
