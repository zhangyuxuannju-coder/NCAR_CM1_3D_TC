#!/usr/bin/env python3
"""Li Fig.5-style Thompson pair structure. No energy diagnostics."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.isentropic_energetics import _conditional_sum_and_mean, saturation_vapor_pressure_pa
C=dict(Rd=287.04,Rv=461.5,Cpd=1005.7,Cpv=1870.,Cl=4190.,Ci=2106.,Lv0=2501000.,Lf=333000.,Tf=273.15,p0=100000.)
def savejson(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def savecsv(path,rows):
    if rows:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def thermo(th,p,rv,rl,ri):
    T=th*(p/C['p0'])**(C['Rd']/C['Cpd'])
    H=p*rv/(C['Rd']/C['Rv']+rv)/saturation_vapor_pressure_pa(T)
    vlnH=np.zeros_like(rv); positive=rv>0
    vlnH[positive]=rv[positive]*np.log(H[positive])
    lv=C['Lv0']+(C['Cpv']-C['Cl'])*(T-C['Tf'])
    B=(C['Cpd']+ri*C['Ci']+(rv+rl)*C['Cl'])*np.log(T/C['Tf'])-C['Rd']*np.log(p/C['p0'])
    B+=(rv+rl)*C['Lf']/C['Tf']+rv*lv/T-C['Rv']*vlnH
    return C['Tf']*np.exp(B/(C['Cpd']+C['Ci']*(rv+rl+ri))),T

def bins(theta,rho,w,area,mask,ed):
    nz=len(theta); nb=len(ed)-1
    mass,flux,raw,count=[np.zeros((nz,nb)) for _ in range(4)]
    dm,net,wbar,gross=[np.zeros(nz) for _ in range(4)]
    for k in range(nz):
        good=mask & np.isfinite(theta[k]) & np.isfinite(rho[k]) & np.isfinite(w[k]) & (rho[k]>0)
        if not np.array_equal(good,mask): raise ValueError('Missing native samples in disk')
        coord=theta[k][good]; weight=rho[k][good]*area[good]
        dm[k]=weight.sum(); net[k]=np.sum(weight*w[k][good]); wbar[k]=net[k]/dm[k]
        wp=w[k][good]-wbar[k]
        mass[k],_= _conditional_sum_and_mean(coord,ed,weight)
        flux[k],_= _conditional_sum_and_mean(coord,ed,weight*wp)
        raw[k],_= _conditional_sum_and_mean(coord,ed,weight*w[k][good])
        count[k],_= _conditional_sum_and_mean(coord,ed,np.ones_like(coord))
        gross[k]=np.sum(abs(weight*wp))
    merr=float(np.max(abs(mass.sum(1)-dm)/dm))
    ferr=float(np.max(abs(flux.sum(1))/np.maximum(gross,1)))
    if merr>1e-10 or ferr>1e-10: raise ValueError(f'Closure failure {merr} {ferr}')
    return dict(M=mass/np.diff(ed),Fz=flux/np.diff(ed),Fz_raw=raw/np.diff(ed),count=count,
                domain_mass=dm,net_raw=net,wbar=wbar,gross=gross,mass_error=np.array(merr),flux_error=np.array(ferr))

def edge_mask(support):
    valid=np.zeros((support.shape[0],support.shape[1]+1),bool)
    valid[:,1:-1]=support[:,:-1]&support[:,1:]
    return valid

def cycle(ed,z,psi,support,level,near):
    fig,ax=plt.subplots()
    cs=ax.contour(ed,z,np.ma.array(psi,mask=~edge_mask(support)),levels=[level],corner_mask=False)
    accepted=[]; candidates=[]
    for i,s in enumerate(cs.allsegs[0]):
        if len(s)<4: continue
        closed=bool(np.allclose(s[0],s[-1],rtol=0,atol=1e-8))
        boundary=bool(np.any(s[:,0]<=ed[0]+1e-8) or np.any(s[:,0]>=ed[-1]-1e-8) or np.any(s[:,1]<=z[0]+1e-8) or np.any(s[:,1]>=z[-1]-1e-8))
        deep=bool(s[:,1].min()<=near and s[:,1].max()>=8000)
        area=float(abs(np.sum(s[:-1,0]*s[1:,1]-s[1:,0]*s[:-1,1]))/2)
        candidates.append(dict(candidate=i,closed=closed,boundary=boundary,deep=deep,zmin_m=float(s[:,1].min()),zmax_m=float(s[:,1].max()),area_K_m=area,accepted=closed and not boundary and deep))
        if closed and not boundary and deep: accepted.append((s,area))
    plt.close(fig)
    if not accepted: return None,candidates
    s=max(accepted,key=lambda a:(a[0][:,1].max(),a[1]))[0][:-1].copy()
    pz=RegularGridInterpolator((z,ed),np.gradient(psi,z,axis=0))
    pt=RegularGridInterpolator((z,ed),np.gradient(psi,ed,axis=1))
    nxt=np.roll(s,-1,axis=0); mid=(s+nxt)/2
    flow=np.c_[-pz(mid[:,[1,0]])/np.ptp(ed),pt(mid[:,[1,0]])/np.ptp(z)]
    score=np.sum(flow*(nxt-s)/np.array([np.ptp(ed),np.ptp(z)]))
    if score<0: s=s[::-1]
    s=np.roll(s,-int(np.argmin(s[:,0])),axis=0)
    low=np.flatnonzero(s[:,1]<=near); ib=int(low[np.argmax(s[low,0])]); ic=int(np.argmax(s[:,1]))
    abc={n:dict(theta_e_K=float(s[i,0]),z_m=float(s[i,1]),index=i) for n,i in [('a',0),('b',ib),('c',ic)]}
    return dict(vertices=np.vstack([s,s[0]]),points=abc,abc_in_flow_order=bool(0<ib<ic),level_kg_s=level),candidates

def weights(t,start,end):
    if start==end: return np.array([np.argmin(abs(t-start))]),np.ones(1)
    idx=np.flatnonzero((t>=start-1e-6)&(t<=end+1e-6)); tt=t[idx]
    if not len(tt): raise ValueError('Empty window')
    if len(tt)==1: return idx,np.ones(1)
    if np.allclose(np.diff(tt),np.diff(tt)[0]) and np.isclose(tt[0],start) and np.isclose(tt[-1],end): return idx,np.ones(len(tt))/len(tt)
    a=np.r_[np.diff(tt)[0]/2,(tt[2:]-tt[:-2])/2,np.diff(tt)[-1]/2]
    return idx,a/a.sum()

def panel(ax,d,ed,z,style,title,xlim):
    levels,cmap,norm,lines=style
    im=ax.contourf((ed[:-1]+ed[1:])/2,z/1000,np.ma.array(d['Fz']/1e9,mask=~d['support']),levels=levels,cmap=cmap,norm=norm,extend='both',corner_mask=False)
    ax.contour(ed,z/1000,np.ma.array(d['Psi']/1e9,mask=~edge_mask(d['support'])),levels=lines,colors='blue',linewidths=.7,corner_mask=False)
    if d['cycle']:
        c=d['cycle']; v=c['vertices']; ax.plot(v[:,0],v[:,1]/1000,'k-',lw=2)
        for n,p in c['points'].items():
            ax.plot(p['theta_e_K'],p['z_m']/1000,'ko',ms=4)
            ax.annotate(n,(p['theta_e_K'],p['z_m']/1000),xytext=(5,5),textcoords='offset points')
    else: ax.text(.03,.97,'No valid fixed-level closed cycle',va='top',transform=ax.transAxes,fontsize=8)
    ax.set(title=title,xlabel=r'Ice-reference $\theta_e$ (K)',ylabel='Height (km)',xlim=xlim,ylim=(0,min(18.,z[-1]/1000)))
    return im

def plot_outputs(cfg,out,results,z,smoke):
    ed=np.arange(200,701,cfg['theta_bin_K']);xlim=(380.,320.)
    peak=max(float(abs(d['Fz']).max()/1e9) for case in results.values() for d in case.values());maxval=max(.1,float(10**np.ceil(np.log10(peak))))
    pos=np.array([.01,.02,.05,.1,.2,.5,1,2,5,10,20,50,100]);pos=np.r_[pos[pos<maxval],maxval];levels=np.r_[-pos[::-1],pos]
    n=len(pos);colors=np.vstack([plt.cm.Purples(np.linspace(.95,.2,n)),[[1,1,1,1]],plt.cm.YlOrRd(np.linspace(.15,.95,n))]);cmap=ListedColormap(colors);cmap.set_under(colors[0]);cmap.set_over(colors[-1]);norm=BoundaryNorm(levels,cmap.N)
    lp=np.array([.5,1,2,5,10,20]);lines=np.r_[-lp[::-1],lp];style=(levels,cmap,norm,lines)
    savejson(out/'plot_config.json',dict(Fz_scale=1e9,Psi_scale=1e9,Fz_levels=levels.tolist(),Psi_levels=lines.tolist(),theta_xlim_K=list(xlim),zmax_km=min(18.,float(z[-1]/1000))))
    def savefig(fig,name):
        fig.savefig(out/f'{name}.png',dpi=220);fig.savefig(out/f'{name}.pdf');plt.close(fig)
    mainwindow='75h' if smoke else '72_78h'
    fig,axes=plt.subplots(1,2,figsize=(10,6),constrained_layout=True)
    for ax,label in zip(axes,results):im=panel(ax,results[label][mainwindow],ed,z,style,label,xlim)
    fig.suptitle('75 h single output (validation)' if smoke else '72–78 h time mean, centered at 75 h',fontsize=14)
    fig.colorbar(im,ax=axes,label=r'$F_z$ ($10^9$ kg s$^{-1}$ K$^{-1}$)',shrink=.8);savefig(fig,'main_isentropic_structure')
    if not smoke:
        fig,axes=plt.subplots(2,3,figsize=(14,10),constrained_layout=True)
        for i,label in enumerate(results):
            for j,(name,title) in enumerate([('74_76h','74–76 h time mean'),('73_77h','73–77 h time mean'),('72_78h','72–78 h time mean')]):im=panel(axes[i,j],results[label][name],ed,z,style,f'{label}: {title}',xlim)
        fig.colorbar(im,ax=axes,label=r'$F_z$ ($10^9$ kg s$^{-1}$ K$^{-1}$)',shrink=.75);savefig(fig,'window_sensitivity')
    fig,ax=plt.subplots(figsize=(6,7),constrained_layout=True)
    for label,color in [('CTRL','red'),('JET30','blue')]:
        c=results[label][mainwindow]['cycle']
        if c:
            v=c['vertices'];ax.plot(v[:,0],v[:,1]/1000,color=color,label=label)
            for n,p in c['points'].items():
                ax.plot(p['theta_e_K'],p['z_m']/1000,'o',color=color,ms=4);ax.annotate(n,(p['theta_e_K'],p['z_m']/1000),xytext=(4,4),textcoords='offset points',color=color)
    ax.set(xlabel=r'Ice-reference $\theta_e$ (K)',ylabel='Height (km)',title=r'Fixed $\Psi=-10^7$ kg s$^{-1}$ cycles',xlim=xlim,ylim=(0,min(18.,z[-1]/1000)))
    if ax.lines:ax.legend()
    else:ax.text(.5,.5,'No valid fixed-level closed cycles',ha='center',transform=ax.transAxes)
    savefig(fig,'cycle_comparison')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path);ap.add_argument('--smoke',action='store_true');ap.add_argument('--plot-only',action='store_true');args=ap.parse_args()
    cfg=dict(cases={'CTRL':'/data/zhangyx/DATA/cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc','JET30':'/data/zhangyx/DATA/cm1out_25N_JET_R2_Thompson_OML100_240h.nc'},baseline_dir='output/thompson_240h_se_attribution/baseline',output_dir='output/li2023_fig5_thompson_CTRL_JET30_72_78h',radius_km=1000.,radii_km=[800.,1000.,1200.],theta_bin_K=1.,bin_sensitivity_K=[.5,1.,2.],cycle_level_kg_s=-1e7,near_surface_m=1000.,minimum_mean_samples=1.)
    if args.config: cfg=json.loads(args.config.read_text())
    out=ROOT/cfg['output_dir']; out=out/'validation_75h' if args.smoke else out;out.mkdir(parents=True,exist_ok=True)
    savejson(out/'config.json',cfg)
    if args.plot_only:
        results={}
        windows=['75h'] if args.smoke else ['74_76h','73_77h','72_78h']
        for label in cfg['cases']:
            results[label]={}
            for name in windows:
                tag=f"{label}_{name}_R{cfg['radius_km']:g}_dtheta{cfg['theta_bin_K']:g}"
                if not (out/f'{tag}.npz').exists():
                    start,end=map(float,name.replace('h','').split('_'))
                    with np.load(out/f'{label}_hourly_bins.npz') as h:
                        ii,a=weights(h['time_h'],start,end)
                        d={k:np.sum(h[k][ii]*a[:,None,None],axis=0) for k in ['M','Fz','Fz_raw','count']}
                        for k in ['domain_mass','net_raw','wbar','gross']:
                            d[k]=np.sum(h[k][ii]*a[:,None],axis=0)
                        z=h['z_m'];ed=h['theta_edges_K']
                        d['mass_error']=np.array(np.max(abs(np.sum(d['M']*np.diff(ed),axis=1)-d['domain_mass'])/d['domain_mass']))
                        d['flux_error']=np.array(np.max(abs(np.sum(d['Fz']*np.diff(ed),axis=1))/np.maximum(d['gross'],1)))
                        d['Psi']=np.c_[np.zeros(len(z)),np.cumsum(d['Fz']*np.diff(ed),axis=1)]
                        d['support']=(d['count']>=cfg['minimum_mean_samples'])&(d['M']>0)
                        c,candidates=cycle(ed,z,d['Psi'],d['support'],cfg['cycle_level_kg_s'],cfg['near_surface_m'])
                        np.savez_compressed(out/f'{tag}.npz',**d,z_m=z,theta_edges_K=ed,time_h=h['time_h'][ii],alpha=a)
                        savejson(out/f'{tag}_cycle.json',dict(status='valid' if c else 'no valid fixed-level deep closed contour',candidates=candidates,points=c['points'] if c else None,abc_in_flow_order=c['abc_in_flow_order'] if c else None,level_kg_s=cfg['cycle_level_kg_s']))
                        if c:savecsv(out/f'{tag}_cycle.csv',[dict(point=i,theta_e_K=float(v[0]),z_m=float(v[1])) for i,v in enumerate(c['vertices'])])
                with np.load(out/f'{tag}.npz') as arrays:
                    d={k:arrays[k] for k in arrays.files}; z=d['z_m']
                c=json.loads((out/f'{tag}_cycle.json').read_text())
                d['cycle']=None
                if c['status']=='valid':
                    with (out/f'{tag}_cycle.csv').open() as f:
                        rows=list(csv.DictReader(f))
                    d['cycle']=dict(points=c['points'],vertices=np.array([[float(r['theta_e_K']),float(r['z_m'])] for r in rows]))
                results[label][name]=d
        plot_outputs(cfg,out,results,z,args.smoke)
        print(f'Replotted cached diagnostics: {out}',flush=True)
        return
    base=ROOT/cfg['baseline_dir']; manifest=json.loads((base/'baseline_manifest.json').read_text())
    with (base/'hourly_intensity_baseline.csv').open() as f: baseline=list(csv.DictReader(f))
    inputs={r['case']:r['input'] for r in manifest['cases']}
    variants=[(cfg['radius_km'],cfg['theta_bin_K'])]+[(r,cfg['theta_bin_K']) for r in cfg['radii_km'] if r!=cfg['radius_km']]+[(cfg['radius_km'],b) for b in cfg['bin_sensitivity_K'] if b!=cfg['theta_bin_K']]
    hourly={}; checks=[]; schemas={}; centres={}; grid_ref=None
    for label,path in cfg['cases'].items():
        key='JET' if label=='JET30' else label
        if inputs[key]!=path: raise ValueError('Baseline input mismatch')
        with Dataset(path) as ds:
            required=['th','prs','rho','qv','qc','qr','qi','qs','qg','w','zhval']
            for n in required:
                expected=('time','zf','yh','xh') if n=='w' else ('time','zh','yh','xh')
                if ds[n].dimensions!=expected: raise ValueError(f'{n}: unexpected dimensions')
            units=dict(th='K',prs='Pa',rho='kg/m^3',w='m/s',time='seconds',xh='km',yh='km',xf='km',yf='km',zh='km',zf='km',zhval='m',qv='kg/kg',qc='kg/kg',qr='kg/kg',qi='kg/kg',qs='kg/kg',qg='kg/kg')
            for n,u in units.items():
                if ds[n].units!=u: raise ValueError(f'{n}: unexpected unit')
            if ds['rho'].long_name!='dry-air density': raise ValueError('Density basis unverified')
            g={n:np.array(ds[n][:],float)*1000 for n in ['xh','yh','zh','xf','yf','zf']};z=g['zh']
            if grid_ref is not None and any(not np.array_equal(g[n],grid_ref[n]) for n in g):raise ValueError('Different grids')
            grid_ref=g;t=np.array(ds['time'][:],float)/3600
            idx=np.flatnonzero((t>=72-1e-6)&(t<=78+1e-6))
            if not len(idx): raise ValueError('No output in window')
            centres[label]=[]
            for it in idx:
                rows=[r for r in baseline if r['case']==key and np.isclose(float(r['time_h']),t[it])]
                if len(rows)!=1 or int(rows[0]['time_index'])!=it:raise ValueError('Centre time mismatch')
                centres[label].append(rows[0])
            schemas[label]=dict(path=path,size_bytes=Path(path).stat().st_size,time_range_h=[float(t[0]),float(t[-1])],selected_times_h=t[idx].tolist(),variables={n:dict(dimensions=list(ds[n].dimensions),units=ds[n].units,long_name=ds[n].long_name) for n in required},density_basis='dry-air density; metadata plus EOS verified',water_basis='CM1 dry-air mixing ratios',liquid_species=['qc','qr'],ice_species=['qi','qs','qg'],excluded_number_concentrations=['nci','ncr'],constants=C,model_top_m=float(g['zf'][-1]))
            hourly[label]={v:[] for v in variants}
            selected=[int(idx[np.argmin(abs(t[idx]-75))])] if args.smoke else idx
            for it in selected:
                rec=next(r for r in centres[label] if int(r['time_index'])==it)
                cx=float(rec['center_x_km'])*1000;cy=float(rec['center_y_km'])*1000;R=max(cfg['radii_km'])*1000
                margin=min(cx-g['xf'][0],g['xf'][-1]-cx,cy-g['yf'][0],g['yf'][-1]-cy)
                if R>margin:raise ValueError('Disk incomplete')
                ix=np.flatnonzero(abs(g['xh']-cx)<=R);iy=np.flatnonzero(abs(g['yh']-cy)<=R)
                sx=slice(ix[0],ix[-1]+1);sy=slice(iy[0],iy[-1]+1)
                def field(n):return np.ma.filled(ds[n][it,:,sy,sx],np.nan).astype(float)
                xx,yy=np.meshgrid(g['xh'][sx]-cx,g['yh'][sy]-cy);radius=np.hypot(xx,yy)
                area=np.diff(g['yf'])[sy,None]*np.diff(g['xf'])[None,sx]
                th=field('th');p=field('prs');rho=field('rho');rv=field('qv');water=[field(n) for n in ['qc','qr','qi','qs','qg']]
                minq=min(float(a.min()) for a in [rv]+water);neg=sum(int(np.sum(a<0)) for a in [rv]+water)
                if minq < -1e-8:raise ValueError(f'Negative water {minq}')
                rv=np.maximum(rv,0);water=[np.maximum(a,0) for a in water]
                theta,T=thermo(th,p,rv,water[0]+water[1],water[2]+water[3]+water[4])
                if theta.min()<200 or theta.max()>=700:raise ValueError('Common bins incomplete')
                wz=field('w');f=(z-g['zf'][:-1])/np.diff(g['zf']);w=wz[:-1]*(1-f[:,None,None])+wz[1:]*f[:,None,None]
                height_error=float(abs(field('zhval')-z[:,None,None]).max())
                if height_error>.02:raise ValueError('Non-horizontal heights require adaptation')
                eos=p/((C['Rd']+C['Rv']*rv)*T);err=float((abs(rho-eos)/eos).max())
                if err>1e-4:raise ValueError(f'EOS mismatch {err}')
                for r,b in variants:
                    ed=np.arange(200,700+b/2,b);d=bins(theta,rho,w,area,radius<=r*1000,ed);d['time_h']=np.array(t[it]);hourly[label][(r,b)].append(d)
                checks.append(dict(case=label,time_h=float(t[it]),index=int(it),center_x_km=cx/1000,center_y_km=cy/1000,theta_min_K=float(theta.min()),theta_max_K=float(theta.max()),EOS_max_relative_error=err,height_max_error_m=height_error,water_clipped_count=neg,minimum_water=minq,full_disk_coverage=True))
                print(f'{label} {t[it]:g}h binned; EOS error={err:.2e}',flush=True)
    savejson(out/'input_schema.json',schemas);savecsv(out/'sample_checks.csv',checks)
    windows=[('75h',75,75)] if args.smoke else [('74_76h',74,76),('73_77h',73,77),('72_78h',72,78)]
    results={};metrics=[];wrows=[]
    for label in hourly:
        results[label]={}
        for r,b in variants:
            samples=hourly[label][(r,b)];tt=np.array([s['time_h'] for s in samples]);ed=np.arange(200,700+b/2,b)
            for name,start,end in windows:
                if (r,b)!=(cfg['radius_km'],cfg['theta_bin_K']) and name!='72_78h' and not args.smoke:continue
                ii,a=weights(tt,start,end);d={k:sum(weight*samples[i][k] for i,weight in zip(ii,a)) for k in samples[0] if k!='time_h'}
                d['Psi']=np.c_[np.zeros(len(z)),np.cumsum(d['Fz']*np.diff(ed),axis=1)];d['support']=(d['count']>=cfg['minimum_mean_samples'])&(d['M']>0)
                c,candidates=cycle(ed,z,d['Psi'],d['support'],cfg['cycle_level_kg_s'],cfg['near_surface_m']);d['cycle']=c
                tag=f'{label}_{name}_R{r:g}_dtheta{b:g}'
                arrays={k:v for k,v in d.items() if isinstance(v,np.ndarray)};arrays.update(theta_edges_K=ed,z_m=z,time_h=tt[ii],alpha=a)
                np.savez_compressed(out/f'{tag}.npz',**arrays)
                savejson(out/f'{tag}_cycle.json',dict(status='valid' if c else 'no valid fixed-level deep closed contour',candidates=candidates,points=c['points'] if c else None,abc_in_flow_order=c['abc_in_flow_order'] if c else None,level_kg_s=cfg['cycle_level_kg_s']))
                if c:savecsv(out/f'{tag}_cycle.csv',[dict(point=i,theta_e_K=float(v[0]),z_m=float(v[1])) for i,v in enumerate(c['vertices'])])
                metric=dict(case=label,window=name,radius_km=r,theta_bin_K=b,Fz_positive_peak=float(d['Fz'].max()),Fz_negative_peak=float(d['Fz'].min()),Psi_min=float(d['Psi'].min()),Psi_max=float(d['Psi'].max()),mass_recovery_error=float(d['mass_error']),flux_recovery_error=float(d['flux_error']),Psi_endpoint_max_kg_s=float(abs(d['Psi'][:,-1]).max()),cycle_top_km=c['points']['c']['z_m']/1000 if c else None,time_h=tt[ii].tolist(),max_gap_h=float(np.diff(tt[ii]).max()) if len(ii)>1 else 0,complete_endpoints=bool(start==end or (np.isclose(tt[ii][0],start) and np.isclose(tt[ii][-1],end))))
                metrics.append(metric)
                for i,weight in zip(ii,a):wrows.append(dict(case=label,window=name,radius_km=r,theta_bin_K=b,time_h=float(tt[i]),alpha=float(weight)))
                if (r,b)==(cfg['radius_km'],cfg['theta_bin_K']):results[label][name]=d
        ss=hourly[label][(cfg['radius_km'],cfg['theta_bin_K'])];tt=np.array([s['time_h'] for s in ss]);stack={k:np.stack([s[k] for s in ss]) for k in ['M','Fz','Fz_raw','count','domain_mass','net_raw','wbar','gross']}
        mt=np.gradient(stack['M'],tt*3600,axis=0) if len(tt)>1 else np.full_like(stack['M'],np.nan)
        np.savez_compressed(out/f'{label}_hourly_bins.npz',**stack,M_t=mt,z_m=z,time_h=tt,theta_edges_K=np.arange(200,701,cfg['theta_bin_K']))
    savejson(out/'diagnostic_metrics.json',metrics);savecsv(out/'time_weights.csv',wrows)
    plot_outputs(cfg,out,results,z,args.smoke)
    text=['# Li Fig.5-style isentropic structure','', '仅执行指定CTRL/JET30结构诊断，不实施完整能量收支。时间平均是单成员数据的适应方法，不是原文集合平均。','',
    '## 实现','', '- 复用原有分箱函数、液面饱和水汽压函数及已核实输入的逐时中心/强度基线；原核心代码不修改。',
    '- 逐时原始三维热力学→Li附录A4冰参考θe→逐层质量加权扣除w均值→分箱→时间平均通量→Psi积分→提取平均线。',
    '- CM1 constants.F默认常数；Cpd=1005.7，Rd=287.04，Rv=461.5，Cl=4190，Ci=2106，Cpv=1870，Lv(Tf)=2501000，Lf=333000；Lv随T线性变化。液水参考RH使用Bolton饱和水汽压，不截断超饱和；rv=0时rv ln H=0。',
    '- rho是输出干空气密度，以状态方程独立核对；液态=qc+qr，固态=qi+qs+qg；不将nci/ncr计入混合比。水物质<-1e-8报错，微小负值置零且记录。',
    '- 实际非均匀xf/yf单元面积，圆域按单元中心纳入；无需柱坐标或1km径向插值，保留原网格热力学差异。',
    '- w由zf按实际高度线性插值到zh；zhval逐时检查。不是有限体积垂直输送守恒映射，保存插值后未经均值扣除的Fz_raw/net_raw供审计；不宣称插值严格质量守恒。',
    '- 分箱200–700K覆盖全部原始样本；默认1K；M单位kg/(m K)，Fz为kg/(s K)，Psi为kg/s，不乘Δz。',
    '- 等间隔且端点完整时等权，否则实际时间梯形权重；真实采样、端点和缺口记录在time_weights和metrics。',
    '- 空箱/平均原始样本数<1掩膜；Psi边界需两侧相邻箱均有效，不跨缺测填闭合。',
    '- 固定Psi=-1e7 kg/s，选闭合、边界内、近地面<=1km、顶部>=8km的最高循环，顶部相同选最大面积；没有合格线时报告。',
    '- a整条线最低θe，b<=1km段最高θe，c最高高度；按(-Psi_z,Psi_theta)定流向并报告abc顺序。近地面和深循环限制是本实现的公开选择。',
    '- 薄线级别/色阶属于数值实现选择，不声称精确原文设定；横轴递减，两组所有窗口统一坐标色标。','',
    '## 验证与限制','', '- 质量恢复、修正Fz积分和Psi端点检查通过才交付；扣除均值后的闭合是构造结果，不证明真实控制体没有侧向交换和质量积累。',
    '- M_t为逐时M的差分，保存于hourly_bins；没有独立Ftheta，因此不能验证完整等熵连续方程或假定稳态。',
    '- 分箱保存全部59层；主图按参考形式显示0–18km，模式顶约25km；本数据未记录海绵层起始高度，不能使用其他试验namelist代替。',
    '- 时间平均循环不是气块真实轨迹；同时间CTRL/JET30差异包含强度及阶段差异，不构成JET因果证明。','',
    '## 窗口/参数敏感性','', '|Case|Window|R km|Δθ K|Fz+ kg/s/K|Fz− kg/s/K|Cycle top km|','|---|---|---:|---:|---:|---:|---:|']
    for m in metrics:
        top='无合格闭合线' if m['cycle_top_km'] is None else f"{m['cycle_top_km']:.3f}"
        text.append(f"|{m['case']}|{m['window']}|{m['radius_km']:g}|{m['theta_bin_K']:g}|{m['Fz_positive_peak']:.4e}|{m['Fz_negative_peak']:.4e}|{top}|")
    text+=['','## 已有强度基线','']
    for label,recs in centres.items():
        a,b=recs[0],recs[-1]
        text.append(f"- {label}: 72→78h中心气压 {float(a['psfc_center_hpa']):.2f}→{float(b['psfc_center_hpa']):.2f} hPa；最低标量层方位平均最大Vt {float(a['vt0_axisymmetric_max_m_s']):.2f}→{float(b['vt0_axisymmetric_max_m_s']):.2f} m/s；RMW {float(a['rmw0_km']):.1f}→{float(b['rmw0_km']):.1f} km。不是10m最大风。")
        mm=[m for m in metrics if m['case']==label and (m['radius_km'],m['theta_bin_K'])==(cfg['radius_km'],cfg['theta_bin_K'])]
        if len(mm)==3:text.append(f"- {label}: 主窗口/74–76h平均正通量峰值比={mm[2]['Fz_positive_peak']/max(mm[0]['Fz_positive_peak'],1):.3f}；顶部与闭合形状见表和window_sensitivity。保持用户指定72–78h主窗口。")
    text+=['','## 交付与复跑','', '- main_isentropic_structure.png/pdf：主图；window_sensitivity.png/pdf：三窗口；cycle_comparison.png/pdf：有效粗线叠加。',
    '- *_R*_dtheta*.npz：分箱M/Fz/Fz_raw/Psi、原始样本覆盖、wbar、原始净输送、时间/权重；*_cycle.json/csv：候选线与abc。',
    '- *_hourly_bins.npz：逐时分箱与M_t；sample_checks.csv/time_weights.csv/diagnostic_metrics.json：质量、时间和敏感性记录。',
    '- config.json/input_schema.json/plot_config.json/provenance.json：变量映射、配置、常数、来源及代码哈希。','', '```sh','cd /data1/home/zhangyx/project/TC_dynamic',f"/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python scripts/plot_li2023_isentropic_time_mean.py --config {out.relative_to(ROOT)}/config.json"+(' --smoke' if args.smoke else ''),'```','',
    '## 来源','', '- Li et al. (2023), Eq7–8/Appendix A4/Fig5，核对用户本地PDF：https://doi.org/10.1175/JAS-D-22-0186.1',
    '- NCAR CM1 constants：https://github.com/NCAR/CM1/blob/main/src/constants.F；服务器r20参考：/data1/home/zhangyx/data/yuanlong_cm1/src/constants.F。',
    '- 原中心/强度基线：output/thompson_240h_se_attribution/baseline，manifest确认本任务输入。','']
    (out/'README.md').write_text('\n'.join(text),encoding='utf-8')
    savejson(out/'provenance.json',dict(script=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,reused=['src.isentropic_energetics._conditional_sum_and_mean','src.isentropic_energetics.saturation_vapor_pressure_pa'],baseline=cfg['baseline_dir']))
    print(f'Completed: {out}',flush=True)
if __name__=='__main__':main()
