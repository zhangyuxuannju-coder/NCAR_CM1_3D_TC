#!/usr/bin/env python3
"""CM1-native three-level extrapolated surface-pressure tendency diagnostic."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_thompson_intensity_baseline import _choose_center
from src.coordinates import destagger_axis
from src.tc_cylindrical import regular_polar_geometry,sample_scalar_to_polar,sample_wind_to_polar
RD,CP,RV=287.04,1005.7,461.5
GAMMA=CP/(CP-RD); EPS=RD/RV
TH_GROUPS={
 'microphysics':['ptb_mp'],'horizontal_diffusion':['ptb_hidiff'],'vertical_diffusion':['ptb_vidiff'],
 'horizontal_turbulence':['ptb_hturb'],'vertical_turbulence':['ptb_vturb'],
 'rayleigh_damping':['ptb_rdamp'],'radiation':['ptb_rad']}
QV_GROUPS={
 'microphysics':['qvb_mp'],'horizontal_diffusion':['qvb_hidiff'],'vertical_diffusion':['qvb_vidiff'],
 'horizontal_turbulence':['qvb_hturb'],'vertical_turbulence':['qvb_vturb']}
DYNAMIC=['mean_radial_advection','mean_vertical_advection','mean_radial_compression','mean_vertical_compression',
         'eddy_radial_advection','eddy_tangential_advection','eddy_vertical_advection','eddy_compression']
SOURCE=['moist_divergence']+list(TH_GROUPS)
TERMS=DYNAMIC+SOURCE

def trap(a,t): return np.trapezoid(a,t,axis=0)/(t[-1]-t[0])
def grad(a,x,axis): return np.gradient(a,x,axis=axis,edge_order=2)
def annulus_average(profile,r,r_inner,r_outer):
    if not (np.isclose(r[0],r_inner) and np.isclose(r[-1],r_outer)):
        raise ValueError('Radial grid must include both annulus boundaries')
    return 2/(r_outer**2-r_inner**2)*np.trapezoid(profile*r,r)
def scalar(ds,name,it): return np.asarray(ds[name][it],dtype=np.float64)
def centers_to(ds,last):
    x=np.asarray(ds['xh'][:],float);y=np.asarray(ds['yh'][:],float);out=[];prev=None
    for it in range(last+1):
        p=scalar(ds,'psfc',it);iy,ix,fb=_choose_center(gaussian_filter(p,2),x,y,prev,180)
        if fb: raise RuntimeError(f'center fallback at index {it}')
        prev=(x[ix],y[iy]);out.append((prev[0]*1000,prev[1]*1000))
    return np.asarray(out)
def native_divergence(ds,it,x,y,z,nkeep=None):
    nz=len(z) if nkeep is None else nkeep
    u=scalar(ds,'u',it)[:nz];v=scalar(ds,'v',it)[:nz];w=scalar(ds,'w',it)[:nz+1]
    du=(u[:,:,1:]-u[:,:,:-1])/np.diff(np.asarray(ds['xf'][:],float)*1000)[None,None,:]
    dv=(v[:,1:,:]-v[:,:-1,:])/np.diff(np.asarray(ds['yf'][:],float)*1000)[None,:,None]
    dw=(w[1:]-w[:-1])/np.diff(np.asarray(ds['zf'][:nz+1],float)*1000)[:,None,None]
    return du+dv,dw

def one_time(ds,it,center,velocity,radii,naz):
    x=np.asarray(ds['xh'][:],float)*1000;y=np.asarray(ds['yh'][:],float)*1000
    z=np.asarray(ds['zh'][:],float)*1000
    geo=regular_polar_geometry(x,y,*center,radii,naz)
    nkeep=5
    p=sample_scalar_to_polar(scalar(ds,'prs',it)[:nkeep],geo)
    th=sample_scalar_to_polar(scalar(ds,'th',it)[:nkeep],geo)
    qv=sample_scalar_to_polar(scalar(ds,'qv',it)[:nkeep],geo)
    u=destagger_axis(scalar(ds,'u',it)[:nkeep],2)-velocity[0]
    v=destagger_axis(scalar(ds,'v',it)[:nkeep],1)-velocity[1]
    ur,vt=sample_wind_to_polar(u,v,geo)
    w=sample_scalar_to_polar(destagger_axis(scalar(ds,'w',it),0)[:nkeep],geo)
    divh_native,divz_native=native_divergence(ds,it,x,y,z,nkeep)
    divh=sample_scalar_to_polar(divh_native[:nkeep],geo)
    divz=sample_scalar_to_polar(divz_native[:nkeep],geo)
    div=divh+divz
    phi=np.asarray(geo['azimuth_rad']); dphi=2*np.pi/naz
    pb=p.mean(1);ub=ur.mean(1);wb=w.mean(1)
    pp=p-pb[:,None,:];up=ur-ub[:,None,:];wp=w-wb[:,None,:];vp=vt-vt.mean(1)[:,None,:]
    db=div.mean(1);dp=div-db[:,None,:];dbh=divh.mean(1);dbz=divz.mean(1)
    dpbdr=grad(pb,radii,1); dpbdz=grad(pb,z[:nkeep],0)
    rdr=grad(radii[None,:]*ub,radii,1)/radii[None,:]
    dwdz=grad(wb,z[:nkeep],0)
    dppdr=grad(pp,radii,2); dppdz=grad(pp,z[:nkeep],0)
    dppdphi=(np.roll(pp,-1,axis=1)-np.roll(pp,1,axis=1))/(2*dphi)
    result={
      'mean_radial_advection':-ub*dpbdr,'mean_vertical_advection':-wb*dpbdz,
      'mean_radial_compression':-GAMMA*pb*dbh,'mean_vertical_compression':-GAMMA*pb*dbz,
      'eddy_radial_advection':-(up*dppdr).mean(1),
      'eddy_tangential_advection':-(vp*dppdphi/radii[None,None,:]).mean(1),
      'eddy_vertical_advection':-(wp*dppdz).mean(1),
      'eddy_compression':-GAMMA*(pp*dp).mean(1)}
    ptdiv=sample_scalar_to_polar(scalar(ds,'ptb_div',it)[:nkeep],geo)
    result['moist_divergence']=(GAMMA*p/th*ptdiv).mean(1)
    for group in TH_GROUPS:
        val=np.zeros_like(p)
        for name in TH_GROUPS[group]:
            if name in ds.variables: val += GAMMA*p/th*sample_scalar_to_polar(scalar(ds,name,it)[:nkeep],geo)
        for name in QV_GROUPS.get(group,[]):
            if name in ds.variables: val += GAMMA*p/(EPS+qv)*sample_scalar_to_polar(scalar(ds,name,it)[:nkeep],geo)
        result[group]=val.mean(1)
    # Extrapolate every pressure field/tendency from the lowest three scalar levels exactly as CM1 psfc.
    a=np.array([float(ds.getncattr(f'cgs{k}')) for k in (1,2,3)])
    surface={name:np.einsum('k,kr->r',a,val[:3]) for name,val in result.items()}
    surface['calculated_total']=sum(surface[name] for name in TERMS)
    surface['psfc_reconstructed']=np.einsum('k,kr->r',a,pb[:3])
    ps=sample_scalar_to_polar(scalar(ds,'psfc',it)[None],geo)[0].mean(0)
    surface['psfc_actual']=ps
    surface['psfc_reconstruction_error']=surface['psfc_reconstructed']-ps
    return surface

def run(config):
    base=json.loads((ROOT/config['base_config']).read_text(encoding='utf-8-sig'))
    out=ROOT/config['output_dir'];out.mkdir(parents=True,exist_ok=True)
    t0=config['center_hour'];half=config['half_window_h'];r0=config.get('radius_inner_km',0.0)*1000;R=config['radius_km']*1000;dr=config['radial_spacing_km']*1000
    radii=np.arange(r0,R+0.5*dr,dr)
    summary={'definitions':config,'constants':{'Rd':RD,'Cp':CP,'Rv':RV,'gamma_d':GAMMA,'epsilon':EPS},'cases':{}}
    all_profiles=[]
    for case,path in base['cases'].items():
      with Dataset(path) as ds:
        time=np.asarray(ds['time'][:],float); hours=time/3600
        ids=np.where((hours>=t0-half)&(hours<=t0+half))[0]
        if list(hours[ids]) != list(np.arange(t0-half,t0+half+1)): raise RuntimeError('window not hourly/complete')
        centers=centers_to(ds,int(ids[-1]+1));velocity=np.gradient(centers,time[:len(centers)],axis=0)
        hourly=[]
        for it in ids:
            hourly.append(one_time(ds,it,centers[it],velocity[it],radii,config['n_azimuth']))
            print(case,hours[it],'done',flush=True)
        avg={name:trap(np.stack([h[name] for h in hourly]),time[ids])*36 for name in TERMS+['calculated_total']}
        pstart=hourly[0]['psfc_actual'];pend=hourly[-1]['psfc_actual']
        actual=(pend-pstart)/(time[ids[-1]]-time[ids[0]])*36
        avg['actual']=actual;avg['residual']=actual-avg['calculated_total']
        pfield=np.stack([h['psfc_actual']/100 for h in hourly])
        recerr=max(float(np.max(np.abs(h['psfc_reconstruction_error']))) for h in hourly)
        row={'case':case,'reconstruction_max_abs_Pa':recerr,'radius_inner_km':r0/1000,'radius_outer_km':R/1000,'window_h':[float(hours[ids[0]]),float(hours[ids[-1]])]}
        for name in TERMS+['calculated_total','actual','residual']:
            row[name]=float(annulus_average(avg[name],radii,r0,R))
        row['closure_fraction_RMSE_over_actual_RMS']=float(np.sqrt(np.mean(avg['residual']**2))/max(np.sqrt(np.mean(actual**2)),1e-12))
        summary['cases'][case]=row
        for ir,r in enumerate(radii/1000):
            rec={'case':case,'radius_km':r}
            rec.update({name:avg[name][ir] for name in TERMS+['calculated_total','actual','residual']});all_profiles.append(rec)
        np.savez_compressed(out/f'{case}_pressure_tendency_fields.npz',radius_km=radii/1000,time_h=hours[ids],psfc_hpa=pfield,**avg)
    pd.DataFrame(all_profiles).to_csv(out/'radial_pressure_tendency_profiles.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot(out,summary)
    return out,summary

def plot(out,summary):
    config=summary['definitions']
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    cases=['CTRL','JET'];labels={'JET':'JET30'}
    pressure_all=np.concatenate([np.load(out/f'{c}_pressure_tendency_fields.npz')['psfc_hpa'].ravel() for c in cases]);pmin,pmax=float(np.nanmin(pressure_all)),float(np.nanmax(pressure_all))
    # Field + closure profile figure.
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,case in enumerate(cases):
      d=np.load(out/f'{case}_pressure_tendency_fields.npz');r=d['radius_km'];t=d['time_h'];p=d['psfc_hpa']
      pcm=axs[row,0].pcolormesh(r,t,p,shading='auto',cmap='viridis',vmin=pmin,vmax=pmax);fig.colorbar(pcm,ax=axs[row,0],label='Surface pressure (hPa)')
      axs[row,0].set(title=f"{labels.get(case,case)} inner-core pressure field",xlabel='Radius (km)',ylabel='Time (h)')
      axs[row,1].plot(r,d['actual'],color='k',lw=2,label='Actual')
      axs[row,1].plot(r,d['calculated_total'],color='#d73027',lw=2,label='Calculated sum')
      axs[row,1].plot(r,d['residual'],color='#4575b4',lw=1.7,label='Residual')
      axs[row,1].axhline(0,color='0.5',lw=.7);axs[row,1].set(title=f"{labels.get(case,case)} 77-83 h tendency",xlabel='Radius (km)',ylabel='hPa h$^{-1}$');axs[row,1].legend();axs[row,1].grid(alpha=.2)
    fig.suptitle('CM1 three-level-extrapolated inner-core surface pressure | center 80 h')
    fig.savefig(out/'inner_core_surface_pressure_field_80h.png',dpi=220);fig.savefig(out/'inner_core_surface_pressure_field_80h.pdf');plt.close(fig)
    # Every tendency component and disk-average validation.
    fig,axs=plt.subplots(3,2,figsize=(14,12),layout='constrained')
    groups=[('Mean dynamics',DYNAMIC[:4]),('Eddy dynamics',DYNAMIC[4:]),('Thermodynamic / physics',SOURCE)]
    for col,case in enumerate(cases):
      d=np.load(out/f'{case}_pressure_tendency_fields.npz');r=d['radius_km']
      for row,(title,names) in enumerate(groups):
        ax=axs[row,col]
        for name in names: ax.plot(r,d[name],label=name.replace('_',' '))
        ax.axhline(0,color='k',lw=.6);ax.set(title=f"{labels.get(case,case)}: {title}",xlabel='Radius (km)',ylabel='hPa h$^{-1}$');ax.set_yscale('symlog',linthresh=.01);ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Resolved surface-pressure tendency components | three-level CM1 extrapolation')
    fig.savefig(out/'inner_core_surface_pressure_terms_80h.png',dpi=220);fig.savefig(out/'inner_core_surface_pressure_terms_80h.pdf');plt.close(fig)
    rows=[]
    for case in cases:
      for name,val in summary['cases'][case].items():
        if isinstance(val,(float,int)): rows.append({'case':labels.get(case,case),'term':name,'hPa_per_h':val})
    pd.DataFrame(rows).to_csv(out/'disk_average_summary.csv',index=False)
    read=['# Inner-core surface-pressure tendency at 80 h','','Definition: CM1 psfc and every tendency term are extrapolated from the lowest three scalar pressure levels with native cgs1-cgs3 coefficients.',f"Window: {config['center_hour']-config['half_window_h']:g}-{config['center_hour']+config['half_window_h']:g} h; annulus: {config.get('radius_inner_km',0):g}-{config['radius_km']:g} km; regular {config['n_azimuth']}-angle sampling; {config['radial_spacing_km']:g}-km radial spacing.",'All products use actual psfc for the independent endpoint tendency. Calculated terms use instantaneous hourly fields and budget snapshots, then trapezoidal averaging.','','## Annulus averages (hPa/h)','']
    for case in cases:
      read.append('### '+labels.get(case,case));read.append('');read.append('|Term|Value|');read.append('|---|---:|')
      for name,val in summary['cases'][case].items():
        if isinstance(val,(float,int)):read.append(f'|{name}|{val:.6f}|')
      read.append('')
    read+=['## Evidence boundary','','The residual includes hourly sampling error, finite-difference/interpolation error, and unoutput model pressure corrections including any apmasscon adjustment. It is not assigned to a physical process. A small reconstruction error verifies the psfc mapping but does not verify tendency closure.','The source budget fields are output-step snapshots, not accumulated hourly means; closure quality must be assessed before process interpretation.','Run: python scripts/diagnose_inner_core_surface_pressure_tendency.py']
    (out/'README.md').write_text('\n'.join(read),encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='config/inner_core_surface_pressure_80h.json');a=p.parse_args();c=json.loads((ROOT/a.config).read_text(encoding='utf-8-sig'));o,s=run(c);print(json.dumps(s,indent=2))
