"""Reuse Bui pipeline; literal fixed-RHS factorial SE, matched-pair averages."""
import sys,json,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()))
from scripts import run_se_forcing_operator_factorial_full as base
from scripts.find_r2_matches_by_phase import pressure_series,tc_metrics
from src.se_bui import build_forcing
from src.se_nonuniform import assemble_flux_form_matrix

OUT=Path('output/strength_sensitivity_phase_mixed_se_25N');OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('/data/zhangyx/DATA')
stem='cm1out_25N_{}_R2_Thompson_OML100{}240h.nc'
paths={'CTRL':DATA/stem.format('CTRL','_')}
for u in [15,30,45,60]:paths['U'+str(u)]=DATA/stem.format('JET','_' if u==30 else '_U'+str(u)+'_')
args=SimpleNamespace(output_dir=str(OUT),max_r_km=300.,max_z_km=20.,dr_km=12.,f=2*7.2921159e-5*np.sin(np.deg2rad(25)),eps_ratio=1e-5,elliptic_margin=0.,baroclinic_scale=1.)
cache=OUT/'cache';cache.mkdir(exist_ok=True)
metric={};pressure={};audit=[]
def metrics(label,h):
    key=f'{label}_{h:g}'
    f=cache/(key+'_metrics.json')
    if key not in metric:
        metric[key]=json.loads(f.read_text()) if f.exists() else tc_metrics(str(paths[label]),h)
        f.write_text(json.dumps(metric[key]))
    return metric[key]
def avg(label,h):
    f=cache/f'{label}_{h:g}_sources.npz'
    if f.exists():return dict(np.load(f,allow_pickle=True))
    print('Reading sources',label,h,flush=True)
    a=base.azimuthal_average(str(paths[label]),h,args)
    np.savez_compressed(f,**a);return a
def solver(case,r,z):
    den=case['basic']['rho']*np.maximum(r,.5*np.min(np.diff(r)))[None,:]
    A=assemble_flux_form_matrix(case['K1_reg']/den,case['K2_reg']/den,case['K3_reg']/den,r,z)
    lu=splu(A.tocsc())
    def solve(rhs):
        rhs=np.asarray(rhs,float).copy();rhs[[0,-1],:]=0;rhs[:,[0,-1]]=0
        x=lu.solve(rhs.ravel());res=np.linalg.norm(A@x-rhs.ravel())/max(np.linalg.norm(rhs),1e-30)
        if not np.isfinite(res) or res>1e-6:raise ValueError(f'SE residual {res}')
        return x.reshape(rhs.shape),res
    return solve
def pair(label,j,c):
    C,J=avg('CTRL',c),avg(label,j);base.check_grids(C,J)
    r=C['r_km']*1000;z=C['z_km']*1000
    oc,oj=[base.build_case(x,r,z,args) for x in (C,J)]
    fc,fj=[build_forcing(o['basic'],a['Q'],a['Fnu'],r,z) for o,a in [(oc,C),(oj,J)]]
    sc,sj=solver(oc,r,z),solver(oj,r,z)
    cc,ecc=sc(fc['forcing_total']);cj,ecj=sc(fj['forcing_total'])
    jc,ejc=sj(fc['forcing_total']);jj,ejj=sj(fj['forcing_total'])
    thermal,et=sc(fj['forcing_thermal']-fc['forcing_thermal'])
    momentum,em=sc(fj['forcing_momentum']-fc['forcing_momentum'])
    ps={'operator':jc-cc,'thermal':thermal,'momentum':momentum,'interaction':jj-jc-cj+cc,'total':jj-cc}
    closure=ps['total']-sum(ps[k] for k in ['operator','thermal','momentum','interaction'])
    rel=np.linalg.norm(closure)/max(np.linalg.norm(ps['total']),1e-30)
    if rel>1e-6:raise ValueError(f'Closure {rel}')
    den=C['rho']*np.maximum(r,.5*np.min(np.diff(r)))[None,:]
    result={'r_km':r/1000,'z_km':z/1000,'actual_jet_ur':J['ur'],'actual_delta_ur':J['ur']-C['ur']}
    for k,p in ps.items():
        result[k+'_psi']=p;result[k+'_ur']=-np.gradient(p,z,axis=0,edge_order=2)/den
        result[k+'_w']=np.gradient(p,r,axis=1,edge_order=2)/den
    result.update(residuals=np.array([ecc,ecj,ejc,ejj,et,em]),closure=rel,changed_ctrl=oc['changed'],changed_jet=oj['changed'])
    np.savez_compressed(OUT/f'{label}_J{j:g}_C{c:g}.npz',**result)
    return result
for label,path in paths.items():
    if not path.exists():raise FileNotFoundError(path)
    print('Pressure',label,flush=True);pressure[label]=pressure_series(str(path))
for label in ['U15','U30','U45','U60']:
    means={}
    for start,end,step in [(50,75,5),(80,150,10)]:
        records=[]
        for j in range(start,end+1,step):
            jm=metrics(label,j);jp=float(np.interp(j,*pressure[label]));choices=[]
            candidates=sorted(range(start,end+1),key=lambda c:abs(float(np.interp(c,*pressure['CTRL']))-jp))
            for c in candidates:
                cm=metrics('CTRL',c);dp=jp-float(np.interp(c,*pressure['CTRL']));dv=jm['vmax_ms']-cm['vmax_ms'];dr=jm['rmw_km']-cm['rmw_km']
                choices.append(dict(case=label,phase=f'{start}_{end}',jet_h=j,ctrl_h=c,dP=dp,dV=dv,dRMW=dr,cost=(dp/2)**2+(dv/2)**2+(dr/12)**2,accepted=abs(dv)<=2 and abs(dr)<=6))
            valid=[x for x in choices if x['accepted']];sel=min(valid or choices,key=lambda x:x['cost']);audit.append(sel)
            (OUT/'matching_audit.json').write_text(json.dumps(audit,indent=2))
            print('Match',sel,flush=True)
            if sel['accepted']:records.append(pair(label,j,sel['ctrl_h']))
        if records:
            mean={k:np.mean([x[k] for x in records],axis=0) for k in records[0]};mean['n_pairs']=len(records)
            means[f'{start}_{end}']=mean;np.savez_compressed(OUT/f'{label}_{start}_{end}_mean.npz',**mean)
    for overlay in ['actual_jet_ur','actual_delta_ur']:
        fig,ax=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        available=list(means.values());vmax=max([np.nanmax(np.abs(m[k+'_ur'])) for m in available for k in ['operator','thermal','momentum']]+[1e-6])
        for row,phase in enumerate(['50_75','80_150']):
            for col,k in enumerate(['operator','thermal','momentum']):
                a=ax[row,col];a.set_title(f'{phase} h | {k}');a.set_xlim(0,300);a.set_ylim(0,20)
                if phase not in means:a.text(.5,.5,'No accepted matches',transform=a.transAxes,ha='center');continue
                m=means[phase];im=a.contourf(m['r_km'],m['z_km'],m[k+'_ur'],levels=np.linspace(-vmax,vmax,25),cmap='RdBu_r',extend='both')
                a.contour(m['r_km'],m['z_km'],m[overlay],levels=[-10,-5,-2,2,5,10,15],colors='k',linewidths=.65)
                a.text(.97,.95,f"n={m['n_pairs']}",transform=a.transAxes,ha='right');a.set_xlabel('Radius (km)');a.set_ylabel('Height (km)')
        if available:fig.colorbar(im,ax=ax.ravel().tolist(),label='SE radial-wind response (m/s)',shrink=.8)
        fig.suptitle(f'{label} vs CTRL | colours: SE response | black: {overlay}')
        fig.savefig(OUT/f'{label}_{overlay}.png',dpi=180,bbox_inches='tight');plt.close(fig)
(OUT/'README.json').write_text(json.dumps(dict(f=args.f,domain_km=[300,20],dr_km=12,eps=args.eps_ratio,averaging='Pairwise solutions then arithmetic mean',matching='Same-phase hourly CTRL; |dV|<=2 m/s and |dRMW|<=6 km; low-level axisymmetric max below 2km, original 12km bins',RHS='Each own-state full Bui thermal/momentum RHS is built once and held fixed across operators',velocity='Common CTRL density for all effects',caveat='Regularized balanced projection, not causal proof; thermal includes existing full Q sources; rejected matches excluded'),indent=2))
print('COMPLETE',flush=True)
