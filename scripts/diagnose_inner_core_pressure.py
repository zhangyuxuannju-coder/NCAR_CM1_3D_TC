#!/usr/bin/env python3
"""Single 80h moving-disk resolved pressure/mass diagnostic; no attribution."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_thompson_intensity_baseline import _choose_center
from src.coordinates import destagger_axis
from src.jet_mechanism_diagnostics import cylindrical_wind,radial_bin_indices,radial_mean,storm_relative_geometry
from src.tc_cylindrical import regular_polar_geometry,sample_scalar_to_polar,sample_wind_to_polar
from src.inner_core_pressure import G,circle_cell_weights,side_terms,window_average

def compute(config):
    base=json.loads((ROOT/config['base_config']).read_text(encoding='utf-8-sig'))
    out=ROOT/config['output_dir'];out.mkdir(parents=True,exist_ok=True)
    t0,half=config['center_hour'],config['half_window_h']
    tracks={}; mapping={}; rmws=[]
    for case,path in base['cases'].items():
        with Dataset(path) as ds:
            t=np.asarray(ds['time'][:],float)/3600
            ids=np.where((t>=t0-half)&(t<=t0+half))[0]
            if not len(ids) or not np.isclose(t[ids[0]],t0-half) or not np.isclose(t[ids[-1]],t0+half) or np.any(np.diff(t[ids])>1.01): raise ValueError('Incomplete window')
            x,y=np.asarray(ds['xh'][:],float),np.asarray(ds['yh'][:],float)
            previous=None;centers=[]
            for it in range(ids[-1]+2):
                p=np.asarray(ds['psfc'][it],float)
                iy,ix,fallback=_choose_center(gaussian_filter(p,2),x,y,previous,180)
                if fallback:raise ValueError('Tracking fallback')
                previous=(x[ix],y[iy]);centers.append(previous)
                if case=='CTRL' and it in ids:
                    u=destagger_axis(np.asarray(ds['u'][it,0],float),1)
                    v=destagger_axis(np.asarray(ds['v'][it,0],float),0)
                    geo=storm_relative_geometry(x*1000,y*1000,previous[0]*1000,previous[1]*1000)
                    _,vt=cylindrical_wind(u,v,geo['cos_azimuth'],geo['sin_azimuth'])
                    edges=np.arange(0,303001,3000);bi,valid=radial_bin_indices(geo['radius_m'],edges)
                    vm=radial_mean(vt,bi,valid,len(edges)-1);j=int(np.nanargmax(vm))
                    if j==0 or j==len(vm)-1:raise ValueError('Unreliable RMW')
                    rmws.append(float((edges[j]+edges[j+1])/2000))
            centers=np.asarray(centers)*1000
            seconds=t[:len(centers)]*3600
            velocity=np.gradient(centers,seconds,axis=0)
            tracks[case]=(ids,t[ids]*3600,centers[ids],velocity[ids])
            mapping[case]={'input':path,'dimensions':dict((k,len(v)) for k,v in ds.dimensions.items()),'variables':{k:{'dimensions':ds[k].dimensions,'attributes':{a:str(ds[k].getncattr(a)) for a in ds[k].ncattrs()}} for k in ds.variables},'global_attributes':{a:str(ds.getncattr(a)) for a in ds.ncattrs()}}
    radius=float(config['radius_km'] or np.median(rmws))*1000
    summaries={};allrows=[]
    for case,path in base['cases'].items():
        ids,seconds,centers,vel=tracks[case];rows=[]
        with Dataset(path) as ds:
            x,y,xf,yf,z,zf=[np.asarray(ds[k][:],float)*1000 for k in ('xh','yh','xf','yf','zh','zf')]
            dz=np.diff(zf)
            if not (np.all(z>zf[:-1]) and np.all(z<zf[1:])):raise ValueError('Bad layer geometry')
            for it,sec,(cx,cy),(cu,cv) in zip(ids,seconds,centers,vel):
                weights=circle_cell_weights(xf,yf,cx,cy,radius);area=np.pi*radius**2
                jj,ii=np.where(weights>0);j0,j1=max(0,jj.min()-2),min(len(y),jj.max()+3);i0,i1=max(0,ii.min()-2),min(len(x),ii.max()+3)
                wgt=weights[j0:j1,i0:i1]
                rho=np.asarray(ds['rho'][it,:,j0:j1,i0:i1],float)
                if not np.isfinite(rho).all() or np.any(rho<=0):raise ValueError('Invalid density')
                actualz=np.asarray(ds['zhval'][it,:,j0:j1,i0:i1],float) if ds['zhval'].dimensions[0]=='time' else np.asarray(ds['zhval'][:,j0:j1,i0:i1],float)
                if not np.allclose(actualz,z[:,None,None],atol=.02):raise ValueError('Nonflat native height: geometry needs extension')
                u=destagger_axis(np.asarray(ds['u'][it,:,j0:j1,i0:i1+1],float),2)
                v=destagger_axis(np.asarray(ds['v'][it,:,j0:j1+1,i0:i1],float),1)
                geo=regular_polar_geometry(x[i0:i1],y[j0:j1],cx,cy,np.array([radius]),config['n_azimuth'])
                if not geo['valid'].all():raise ValueError('Incomplete side coverage')
                rp=sample_scalar_to_polar(rho,geo)[:,:,0]
                ur,_=sample_wind_to_polar(u-cu,v-cv,geo)
                mean,eddy,side=side_terms(rp,ur[:,:,0],dz,radius)
                # Interface rho follows CM1 solve3.F extrapolation coefficients.
                rb=sum(float(ds.getncattr('cgs'+str(k+1)))*rho[k] for k in range(3))
                rt=sum(float(ds.getncattr('cgt'+str(k+1)))*rho[-k-1] for k in range(3))
                wb=np.asarray(ds['w'][it,0,j0:j1,i0:i1],float);wt=np.asarray(ds['w'][it,-1,j0:j1,i0:i1],float)
                vertical=-G/area*np.sum(wgt*(rt*wt-rb*wb))*36
                un=np.asarray(ds['u'][it,:,j0:j1,i0:i1+1],float)-cu
                vn=np.asarray(ds['v'][it,:,j0:j1+1,i0:i1],float)-cv
                wx=(xf[i0+1:i1]-x[i0:i1-1])/np.diff(x[i0:i1])
                wy=(yf[j0+1:j1]-y[j0:j1-1])/np.diff(y[j0:j1])
                fx=((1-wx)[None,None,:]*rho[:,:,:-1]+wx[None,None,:]*rho[:,:,1:])*un[:,:,1:-1]
                fy=((1-wy)[None,:,None]*rho[:,:-1,:]+wy[None,:,None]*rho[:,1:,:])*vn[:,1:-1,:]
                div=(fx[:,1:-1,1:]-fx[:,1:-1,:-1])/np.diff(xf[i0:i1+1])[None,None,1:-1]
                div+=(fy[:,1:,1:-1]-fy[:,:-1,1:-1])/np.diff(yf[j0:j1+1])[None,1:-1,None]
                native_side=-G/area*36*np.sum(div*dz[:,None,None]*wgt[None,1:-1,1:-1])
                p=np.asarray(ds['psfc'][it,j0:j1,i0:i1],float)
                rows.append(dict(case=case,time_h=sec/3600,center_x_m=cx,center_y_m=cy,translation_u=cu,translation_v=cv,P_R_hpa=np.sum(p*wgt)/area/100,column_dry_mass_kg=np.sum(rho*dz[:,None,None]*wgt),mean=mean,eddy=eddy,side=side,native_divergence_side=native_side,vertical=vertical,source=np.nan,top_pressure_tendency=np.nan,T_calc=np.nan,max_top_w=float(abs(wt).max()),max_bottom_w=float(abs(wb).max())))
                print(case,sec/3600,'done',flush=True)
        f=pd.DataFrame(rows);allrows.extend(rows)
        av={k:float(window_average(f[k].to_numpy(),seconds)) for k in ('mean','eddy','side','vertical','native_divergence_side')}
        av['actual']=float((f.P_R_hpa.iloc[-1]-f.P_R_hpa.iloc[0])/(2*half))
        av['mass_actual']=float(G/(np.pi*radius**2)*(f.column_dry_mass_kg.iloc[-1]-f.column_dry_mass_kg.iloc[0])/(2*half*3600)*36)
        av['resolved_sum']=av['side']+av['vertical']
        av['dry_mass_residual_to_resolved']=av['mass_actual']-av['resolved_sum']
        av['pressure_minus_resolved']=av['actual']-av['resolved_sum']
        av['mean_eddy_error']=float(np.max(abs(f['mean']+f.eddy-f.side)))
        av['side_minus_native_divergence']=av['side']-av['native_divergence_side']
        av['mass_minus_native_resolved']=av['mass_actual']-av['native_divergence_side']-av['vertical']
        av['residual_statistics_n1']={k:{'n':1,'mean':av[k],'RMSE':abs(av[k]),'max_absolute':abs(av[k])} for k in ('dry_mass_residual_to_resolved','pressure_minus_resolved')}
        summaries[case]=av
        np.savez_compressed(out/(case+'_intermediate.npz'),**{k:f[k].to_numpy() for k in f.columns})
    pd.DataFrame(allrows).to_csv(out/'hourly_diagnostics.csv',index=False)
    result={'config':config,'radius_km':radius/1000,'CTRL_RMW_reference_km':rmws,'reference_window_h':[t0-half,t0+half],'RMW_definition':'lowest scalar level cyclonic azimuthal mean; native 3km annuli, project baseline interface','top_km':float(zf[-1]/1000),'bottom_km':float(zf[0]/1000),'summaries':summaries,'missing':['geometric top pressure','native integrated dry mass flux/source incl optional pressure mass correction','precipitation sedimentation/source fluxes for total wet mass'],'mass_definition':'rho is dry-air density, not total supporting mass','status':'resolved terms only; full pressure closure, R_closure and T_N unavailable','source_evidence':'CM1 r20.3 solve2.F:1593-1596 dry rho=p/[T*(Rd+qv*Rv)]; 1622-1625 optional global dry mass correction; solve3.F:770-771 boundary rho extrapolation','damping':'candidate archived namelist zd=20km; not independently linked to this archived run','method':'native layer thickness; circle-cell intersections and cellwise constant volume/pressure; equal-angle bilinear scalar side sampling after arithmetic destagger; hourly instantaneous flux trapezoid; relative wind includes center translation once'}
    (out/'variable_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return out,result

def plot(out,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f=pd.read_csv(out/'hourly_diagnostics.csv');s=result['summaries'];cases=['CTRL','JET'];colors=['#2166ac','#b2182b']
    fig,axs=plt.subplots(3,2,figsize=(13,12),layout='constrained')
    for case,c in zip(cases,colors):
        q=f[f.case==case];axs[0,0].plot(q.time_h,q.P_R_hpa,'o-',color=c,label='JET30' if case=='JET' else case)
    axs[0,0].set(title='(a) Moving-disk surface pressure',xlabel='Time (h)',ylabel='hPa');axs[0,0].legend()
    keys=['mean','eddy','vertical'];labels=['Side mean','Side eddy','Top/bottom']
    for ax,case,c in zip([axs[0,1],axs[1,0]],cases,colors):
        ax.bar(labels,[s[case][k] for k in keys],color=c);ax.set(title='(b) '+('JET30' if case=='JET' else case)+' resolved dry-air terms',ylabel='hPa/h')
        ax.text(.02,.03,'Source and top pressure: unavailable',transform=ax.transAxes,fontsize=9)
    dk=['actual','resolved_sum']+keys
    axs[1,1].bar(['Actual','Resolved sum']+labels,[s['JET'][k]-s['CTRL'][k] for k in dk],color='#666666')
    axs[1,1].set(title='(c) JET30 - CTRL (77-83 h)',ylabel='hPa/h');axs[1,1].tick_params(axis='x',rotation=25)
    for i,case in enumerate(cases):
        axs[2,0].bar(i-.18,s[case]['actual'],width=.34,color=colors[i],label='Actual psfc' if i==0 else None)
        axs[2,0].bar(i+.18,s[case]['resolved_sum'],width=.34,color=colors[i],alpha=.4,label='Resolved dry mass flux sum' if i==0 else None)
    axs[2,0].set(xticks=[0,1],xticklabels=['CTRL','JET30'],title='(d) Independent pressure tendency comparison',ylabel='hPa/h');axs[2,0].legend(fontsize=8)
    axs[2,0].text(.02,.03,'Full T_calc unavailable; no pressure closure claimed',transform=axs[2,0].transAxes,fontsize=9)
    for i,case in enumerate(cases):
        axs[2,1].bar(i-.18,s[case]['dry_mass_residual_to_resolved'],width=.34,color=colors[i],label='Dry mass endpoint - resolved' if i==0 else None)
        axs[2,1].bar(i+.18,s[case]['pressure_minus_resolved'],width=.34,color=colors[i],alpha=.4,label='Actual psfc - resolved' if i==0 else None)
    axs[2,1].set(xticks=[0,1],xticklabels=['CTRL','JET30'],title='(e) Residuals to resolved terms (not full closure)',ylabel='hPa/h');axs[2,1].legend(fontsize=8)
    for ax in axs.flat:ax.axhline(0,color='k',lw=.6) if ax!=axs[0,0] else None;ax.grid(axis='y',alpha=.2)
    axs[2,1].text(.02,.03,'Spatial side/divergence check failed; preliminary flux sum',transform=axs[2,1].transAxes,fontsize=8)
    fig.suptitle('CTRL / JET30 inner-core pressure and dry-column mass | center 80 h',fontsize=14)
    fig.supxlabel(f"R={result['radius_km']:g} km (CTRL 77-83 h RMW median); full native column 0-{result['top_km']:.3f} km\nGaussian-smoothed psfc track; relative wind; 360-angle side sampling; flux trapezoid over +/-3 h\nDry-air mass; wet loading, source and geometric top pressure unresolved. All tendencies hPa/h.",fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/('inner_core_pressure_80h.'+ext),dpi=200)
    plt.close(fig)
    lines=['# 80 h inner-core diagnostic',f"Radius: {result['radius_km']} km; window 77-83 h; top {result['top_km']} km.",result['status'],result['method'],result['source_evidence'],'','|Case|Actual psfc|Resolved sum|Dry mass endpoint|Dry mass minus resolved|psfc minus resolved|','|---|---:|---:|---:|---:|---:|']
    for case in cases:lines.append('|'+case+'|'+'|'.join(f'{s[case][k]:.6f}' for k in ('actual','resolved_sum','mass_actual','dry_mass_residual_to_resolved','pressure_minus_resolved'))+'|')
    lines+=['','Units hPa/h. One window per case: temporal residual statistics have n=1; RMSE=max absolute=absolute residual, mean=signed residual.','Dry mass residual may include time sampling, spatial quadrature and unresolved numerical mass adjustment. Surface-pressure minus dry resolved sum also contains unmeasured top pressure and wet mass/static mapping; it is not R_closure or a pure nonhydrostatic term.','Missing: '+', '.join(result['missing']), 'Variable definitions/grid positions: variable_mapping.json; no geometric top pressure reconstructed from highest cell center.','Run: python scripts/diagnose_inner_core_pressure.py --config config/inner_core_pressure_80h.json']
    lines+=['','|Case|Native horizontal divergence|Circle side minus native|Dry mass minus native resolved|','|---|---:|---:|---:|']
    for case in cases:
        lines.append('|'+case+'|'+'|'.join(f'{s[case][k]:.6f}' for k in ('native_divergence_side','side_minus_native_divergence','mass_minus_native_resolved'))+'|')
    lines+=['Spatial side versus native-divergence disagreement is substantial; the circle-side resolved sum is preliminary and has not passed spatial validation.','', '|Case|Residual to resolved|Mean|RMSE|Max absolute|n|','|---|---|---:|---:|---:|---:|']
    for case in cases:
        for name,stats in s[case]['residual_statistics_n1'].items():
            lines.append('|'+case+'|'+name+'|'+'|'.join(f'{stats[k]:.6f}' for k in ('mean','RMSE','max_absolute'))+'|1|')
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/inner_core_pressure_80h.json');a=p.parse_args()
    c=json.loads((ROOT/a.config).read_text(encoding='utf-8-sig'));out,r=compute(c);plot(out,r);print(json.dumps(r,indent=2))
