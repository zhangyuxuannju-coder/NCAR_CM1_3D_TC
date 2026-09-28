"""Orchestrate existing Bui assembly and sparse solver; never edit core routines."""
import sys,json,csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()))
from scripts import plot_inertial_operator_evolution_matched as old
from src.se_bui import build_basic_state,invert_balanced_theta,regularize_ellipticity,assemble_operator
from src._se_pipeline_single import psi_to_uw,_rho_ext_from_rho_zr
ROOT=Path('output/same_time_se_window3h_50_150_new01')
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
args=SimpleNamespace(f=6.1636e-5,eps_ratio=1e-5)
original=spla.spsolve
class MatrixReady(Exception):pass
def matrix(op,r,z,rho):
    box={}
    def capture(m,b,**kwargs):box['m']=m;raise MatrixReady()
    spla.spsolve=capture
    try:old.solve(op,np.zeros_like(rho),rho,r,z)
    except MatrixReady:pass
    finally:spla.spsolve=original
    return box['m']
def state(label,h):
    name=f'{label}_{h:g}_sources.npz'
    for p in [ROOT/'sources',Path('output/jet30_matched_se_1200_plot500/cache'),Path('output/inertial_thermal_window3h_20260914_new01/sources')]:
        if (p/name).exists():return dict(np.load(p/name,allow_pickle=True))
    raise FileNotFoundError(name)
def winds(p,rho,r,z):
    ext=np.zeros((len(r),len(z)+2));ext[:,1:-1]=p.T;ext[:,-1]=ext[:,-2]
    u,w=psi_to_uw(ext,_rho_ext_from_rho_zr(rho),r,np.mean(np.diff(r)),np.mean(np.diff(z)))
    return u[:,1:-1].T,w[:,1:-1].T
def rel(a,b):return float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-30))
names=['thermal','momentum','forcing_total','static','baroclinic','inertial','metric','operator_total','higher_order','density','sum']
metrics=[];numerical=[];cache_keys=None
for centre in range(50,151,10):
    records=[]
    for h in range(centre-3,centre+4):
        a=dict(np.load(ROOT/f'hourly_{h:03d}.npz'));C,J=state('CTRL',h),state('U30',h)
        cache_keys=list(C);r,z=a['r_km']*1000,a['z_km']*1000
        basics=[];ks=[];ops=[];regs=[]
        for s in [C,J]:
            assert np.allclose(s['r_km'],a['r_km']) and np.allclose(s['z_km'],a['z_km'])
            theta,_=invert_balanced_theta(s['ut'],s['theta'],r,z,args.f,outer_smooth_window=1)
            b=build_basic_state(s['ut'],theta,s['rho'],r,z,args.f)
            k=regularize_ellipticity(b['K1_raw'],b['K2_raw'],b['K3_raw'],eps_ratio=args.eps_ratio,margin=0.)[:3]
            _,op,reg=old.build_ctrl_operator(b,r,z,args)
            basics.append(b);ks.append(k);ops.append(op);regs.append(reg)
        mc,mj=[matrix(op,r,z,s['rho']) for op,s in zip(ops,[C,J])]
        fac=spla.factorized(mc.tocsc())
        def solve(rhs):
            # Existing old.solve assembly/velocity conversion; reuse one identical LU factorization.
            spla.spsolve=lambda m,b,**kwargs:fac(b)
            try:return old.solve(ops[0],rhs,C['rho'],r,z)
            finally:spla.spsolve=original
        pc,pj=a['ctrl_psi'],a['jet_psi'];vector=pc.T.ravel();dvector=(pj-pc).T.ravel()
        dm=mj-mc;zero=np.zeros_like(C['rho']);dks=[j-c for j,c in zip(ks[1],ks[0])]
        groups={}
        for i,label in enumerate(['static','baroclinic','inertial']):
            kvals=[zero.copy(),zero.copy(),zero.copy()];kvals[i]=dks[i]
            groups[label]=matrix(assemble_operator(basics[0],*kvals,r,z),r,z,C['rho'])
        groups['metric']=dm-sum(groups.values())
        result=dict(r_km=a['r_km'],z_km=a['z_km'],rho=C['rho'],jet_ur=a['jet_ur'],delta_ur=a['delta_ur'],ctrl_ur=a['ctrl_ur'],SE_delta=a['jet_u']-a['ctrl_u'])
        for label in ['thermal','momentum']:
            result[label+'_rhs']=a[label+'_rhs']
        result['forcing_total_rhs']=a['thermal_rhs']+a['momentum_rhs']
        for label,m in groups.items():result[label+'_rhs']=(-(m@vector)).reshape(pc.T.shape).T
        result['operator_total_rhs']=(-(dm@vector)).reshape(pc.T.shape).T
        result['higher_order_rhs']=(-(dm@dvector)).reshape(pc.T.shape).T
        for label in names[:-2]:
            response=solve(result[label+'_rhs'])
            for k in ['psi','u','w']:result[label+'_'+k]=response[k]
        uj,wj=winds(pj,C['rho'],r,z)
        result['density_u']=a['jet_u']-uj;result['density_psi']=zero
        _,actual_wj=winds(pj,J['rho'],r,z);result['density_w']=actual_wj-wj
        for k in ['psi','u','w']:
            result['sum_'+k]=result['forcing_total_'+k]+result['operator_total_'+k]+result['higher_order_'+k]+result['density_'+k]
        test=dict(hour=h,operator_component_error=rel(sum(result[n+'_rhs'] for n in groups)-result['operator_total_rhs'],result['operator_total_rhs']),forcing_response_error=rel(result['thermal_u']+result['momentum_u']-result['forcing_total_u'],result['forcing_total_u']),complete_SE_difference_error=rel(result['sum_u']-result['SE_delta'],result['SE_delta']),ctrl_raw_nonelliptic=regs[0]['raw_nonelliptic_fraction'],jet_raw_nonelliptic=regs[1]['raw_nonelliptic_fraction'],ctrl_regularized_fraction=regs[0]['changed_coefficient_fraction'],jet_regularized_fraction=regs[1]['changed_coefficient_fraction'])
        numerical.append(test)
        assert all(np.all(np.isfinite(v)) for v in result.values())
        np.savez_compressed(OUT/f'hourly_{h:03d}.npz',**result)
        records.append(result)
    m={k:np.mean([q[k] for q in records],axis=0) for k in records[0]}
    np.savez_compressed(OUT/f'window_{centre:03d}.npz',**m,hours=np.arange(centre-3,centre+4))
    r,z=m['r_km'],m['z_km'];rr,zz=np.meshgrid(r,z)
    view=(rr<=500)&(zz<=18);low=(rr>=20)&(rr<=150)&(zz<=2);upper=(rr>=50)&(rr<=400)&(zz>=10)&(zz<=17)
    weight=m['rho']*rr*np.gradient(r)[None,:]*np.gradient(z)[:,None]
    def avg(f,mask):return float(np.sum(f[mask]*weight[mask])/np.sum(weight[mask]))
    for region,mask,sign in [('BL_inflow',low,-1),('outflow',upper,1)]:
        row=dict(hour=centre,region=region,ctrl=sign*avg(m['ctrl_ur'],mask),jet=sign*avg(m['jet_ur'],mask),actual_difference=sign*avg(m['delta_ur'],mask))
        for label in names:row[label]=sign*avg(m[label+'_u'],mask)
        metrics.append(row)
    # Actual outflow centroids use positive radial wind, not signed diagnostic responses.
    for case in ['ctrl','jet']:
        q=np.maximum(m[case+'_ur'],0)*weight*upper
        centroid=dict(hour=centre,case=case,height_km=float(np.sum(q*zz)/max(np.sum(q),1e-30)),radius_km=float(np.sum(q*rr)/max(np.sum(q),1e-30)))
        with (OUT/'outflow_centroids.jsonl').open('a') as f:f.write(json.dumps(centroid)+'\n')
    dr=np.arange(r[0],500.001,2.);dz=np.arange(z[0],18.001,.1);R,Z=np.meshgrid(dr,dz);points=np.c_[Z.ravel(),R.ravel()]
    def display(f):return RegularGridInterpolator((z,r),f,method='linear',bounds_error=True)(points).reshape(Z.shape)
    p5,p95=np.percentile(m['delta_ur'][view],[5,95]);ulim=max(abs(p5),abs(p95),1e-6)
    def panel(fig,ax,field,title,wind,limit):
        im=ax.contourf(dr,dz,display(field),levels=np.linspace(-limit,limit,31),cmap='RdBu_r',extend='both')
        old.contour_radial_wind(ax,dr,dz,display(m[wind]))
        ax.set(xlim=(0,500),ylim=(0,18),title=title,xlabel='Radius (km)',ylabel='Height (km)')
        fig.colorbar(im,ax=ax,shrink=.8)
    for wind in ['jet_ur','delta_ur']:
        for kind,labels in [('forcing',['thermal','momentum','forcing_total']),('operator',['static','baroclinic','inertial','metric','operator_total'])]:
            fig,axes=plt.subplots(2,len(labels),figsize=(5*len(labels),9),sharex=True,sharey=True)
            for col,label in enumerate(labels):
                rhs=m[label+'_rhs'];lim=max(float(np.percentile(np.abs(rhs[view]),99)),1e-30)
                panel(fig,axes[0,col],rhs,label+' RHS (K^-1 s^-3)',wind,lim)
                panel(fig,axes[1,col],m[label+'_u'],label+' response ur (m/s)',wind,ulim)
            fig.suptitle(f'{centre} h +/-3 h | complete {kind} | black: {wind} | solve 1200 km, view 500 km; linear display interpolation')
            fig.tight_layout();fig.savefig(OUT/f'{centre:03d}_{kind}_{wind}.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
    fields=[m['delta_ur'],m['forcing_total_u'],m['operator_total_u'],m['higher_order_u'],m['sum_u'],m['sum_u']-m['delta_ur']]
    titles=['Actual JET-CTRL','All RHS responses','All operator responses (leading)','Coupled / higher-order response','Sum incl. higher-order and density','Sum - actual difference']
    for ax,f,title in zip(axes.ravel(),fields,titles):panel(fig,ax,f,title,'delta_ur',ulim)
    fig.suptitle(f'{centre} h +/-3 h | radial-wind difference / responses, m/s | common actual P5/P95 colour scale')
    fig.tight_layout();fig.savefig(OUT/f'{centre:03d}_comparison.png',dpi=150);plt.close(fig)
    print('DONE WINDOW',centre,'max relative SE reconstruction error',max(t['complete_SE_difference_error'] for t in numerical if abs(t['hour']-centre)<=3),flush=True)
    (OUT/'progress.json').write_text(json.dumps(dict(completed_centre_h=centre)))
for name,rows in [('regional_metrics.csv',metrics),('numerical_checks.csv',numerical)]:
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
for i,region in enumerate(['BL_inflow','outflow']):
    rows=[q for q in metrics if q['region']==region];t=[q['hour'] for q in rows]
    for name,color in [('ctrl','black'),('jet','firebrick')]:axes[i,0].plot(t,[q[name] for q in rows],label=name.upper(),color=color)
    for name in ['actual_difference','forcing_total','operator_total','higher_order','sum']:axes[i,1].plot(t,[q[name] for q in rows],label=name)
    for ax in axes[i]:ax.axhline(0,color='0.5',lw=.6);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set(ylabel=region+' (m/s)',xlabel='Time (h)')
fig.suptitle('Mass-weighted signed radial-wind measures: BL inflow 20-150 km, 0-2 km; outflow 50-400 km, 10-17 km')
fig.tight_layout();fig.savefig(OUT/'inflow_outflow_trends.png',dpi=150);plt.close(fig)
summary=dict(ctrl='/data/zhangyx/DATA/cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc',jet='/data/zhangyx/DATA/cm1out_25N_JET_R2_Thompson_OML100_240h.nc',source_cache=str(ROOT),source_cache_keys=cache_keys,centres_h=list(range(50,151,10)),window='same-time +/-3 h; solve hourly then mean, no intensity matching',formulation='Bui general, thermal-wind inverted basic-state theta; existing regularized sparse solver',f=args.f,eps_ratio=args.eps_ratio,solve_radius_km=1200,native_dr_km=12,native_shape=list(m['delta_ur'].shape),native_z_km=z.tolist(),solver_dz_km=float(np.mean(np.diff(z))),view_radius_km=500,display_interpolation='Linear 2 km radial, 0.1 km vertical; no extrapolation; metrics on native grid',boundaries='Existing streamfunction Neumann radial/top ghosts, zero bottom ghost; not established as physical impermeable stress-free velocity BC; mixed-derivative stencil issue remains',core_modified=False,forcing_terms='All retained build_forcing RHS: thermal + tangential momentum; cached Q/Fnu are aggregate sources, unavailable model terms cannot be claimed included',operator_terms='Regularized K1/K2/K3 changes at CTRL density; metric group is exact remaining discrete matrix difference including density gradients; leading reference is CTRL SE psi, not actual CTRL wind; no smoothing/cutoff',higher_order='-deltaL*(psiJET-psiCTRL), inverted with CTRL operator; retains coupled operator/forcing response',density='JET-minus-CTRL streamfunction-to-wind density conversion at psiJET; not an extra RHS forcing',strict_internal_check_only=True,max_SE_reconstruction_relative_error=max(q['complete_SE_difference_error'] for q in numerical),max_component_relative_error=max(q['operator_component_error'] for q in numerical),warning='Trends and balanced projections are diagnostics, not unique causal attribution; early 50/60 h amplitudes were anomalous in prior SE-vs-actual audit')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'README.md').write_text('# Complete SE difference attribution\n\nSee summary.json for definitions, approximations, boundary limitations and source files. All 11 windows have forcing/operator plots with actual JET and actual JET-CTRL contours, plus comparison plots. Interpolation is display-only. regional_metrics.csv uses native-grid CTRL-density mass weighting; it is not a surface mass-flux measure. No core routines or earlier products changed.\n')
print('COMPLETE',flush=True)
