"""Audit unchanged existing solver: capture its matrix, compare superposition and CM1."""
import sys,json,csv
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import scipy.sparse.linalg as sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import plot_inertial_operator_evolution_matched as old
from src.se_bui import build_basic_state,invert_balanced_theta
from src._se_pipeline_single import psi_to_uw,_rho_ext_from_rho_zr
ROOT=Path('output/same_time_se_window3h_50_150_new01')
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
args=SimpleNamespace(eps_ratio=1e-5,f=6.1636e-5)
original_spsolve=sparse.spsolve
captured={}
def capture(matrix,rhs,*args,**kwargs):
    x=original_spsolve(matrix,rhs,*args,**kwargs)
    captured.update(matrix=matrix,rhs=rhs,x=x,residual=float(np.linalg.norm(matrix@x-rhs)/max(np.linalg.norm(rhs),1e-30)))
    return x
def solve(op,rhs,rho,r,z):
    sparse.spsolve=capture
    try:v=old.solve(op,rhs,rho,r,z)
    finally:sparse.spsolve=original_spsolve
    return v,dict(captured)
def state(label,h):
    name=f'{label}_{h:g}_sources.npz'
    for directory in [ROOT/'sources',Path('output/jet30_matched_se_1200_plot500/cache'),Path('output/inertial_thermal_window3h_20260914_new01/sources')]:
        p=directory/name
        if p.exists():return dict(np.load(p,allow_pickle=True))
    raise FileNotFoundError('Required existing source cache: '+name)
def vector(p):return p.T.ravel()
def relative(diff,ref):return float(np.linalg.norm(diff)/max(np.linalg.norm(ref),1e-30))
rows=[];hourly_rows=[]
for centre in range(50,151,10):
    parts=[];tests=[]
    for h in range(centre-3,centre+4):
        a=dict(np.load(ROOT/f'hourly_{h:03d}.npz'));C,J=state('CTRL',h),state('U30',h)
        r,z=a['r_km']*1000,a['z_km']*1000
        ops=[]
        for s in [C,J]:
            t,_=invert_balanced_theta(s['ut'],s['theta'],r,z,args.f,outer_smooth_window=1)
            b=build_basic_state(s['ut'],t,s['rho'],r,z,args.f)
            _,op,_=old.build_ctrl_operator(b,r,z,args);ops.append(op)
        test=dict(hour=h)
        for i,label,s in [(0,'ctrl',C),(1,'jet',J)]:
            q,mq=solve(ops[i],a[label+'_thermal_rhs'],s['rho'],r,z)
            m,mm=solve(ops[i],a[label+'_momentum_rhs'],s['rho'],r,z)
            for k in ['psi','u','w']:test[label+'_'+k+'_superposition_error']=relative(q[k]+m[k]-a[label+'_'+k],a[label+'_'+k])
            rhs=a[label+'_thermal_rhs']+a[label+'_momentum_rhs']
            test[label+'_stored_total_residual']=relative(mq['matrix']@vector(a[label+'_psi'])-rhs.T.ravel(),rhs.T.ravel())
            test[label+'_solve_residual']=max(mq['residual'],mm['residual'])
        fc=a['ctrl_thermal_rhs']+a['ctrl_momentum_rhs'];fj=a['jet_thermal_rhs']+a['jet_momentum_rhs']
        cj,_=solve(ops[0],fj,C['rho'],r,z);jc,_=solve(ops[1],fc,C['rho'],r,z)
        psiC,psiJ=a['ctrl_psi'],a['jet_psi']
        pforce=cj['psi']-psiC;pop=jc['psi']-psiC;pint=psiJ-jc['psi']-cj['psi']+psiC
        def wind(p):
            ext=np.zeros((r.size,z.size+2));ext[:,1:-1]=p.T;ext[:,-1]=ext[:,-2]
            u,w=psi_to_uw(ext,_rho_ext_from_rho_zr(C['rho']),r,float(np.mean(np.diff(r))),float(np.mean(np.diff(z))))
            return u[:,1:-1].T
        full_op=wind(pop);interaction=wind(pint);force=a['thermal_u']+a['momentum_u']
        rho_term=a['jet_u']-wind(psiJ)
        deltaSE=a['jet_u']-a['ctrl_u'];closed=force+full_op+interaction+rho_term
        test['fixed_CTRL_forcing_difference_error']=relative(force-wind(pforce),wind(pforce))
        test['full_difference_psi_closure']=relative(pforce+pop+pint-(psiJ-psiC),psiJ-psiC)
        test['full_difference_u_closure']=relative(closed-deltaSE,deltaSE)
        tests.append(test);hourly_rows.append(test)
        parts.append(dict(r_km=a['r_km'],z_km=a['z_km'],actual_delta=a['delta_ur'],SE_delta=deltaSE,current_sum=force+a['inertial_u'],full_sum=closed,inertial=a['inertial_u'],full_operator=full_op,interaction=interaction,density_term=rho_term,ctrl_actual=a['ctrl_ur'],jet_actual=a['jet_ur'],ctrl_SE=a['ctrl_u'],jet_SE=a['jet_u']))
    mean={k:np.mean([p[k] for p in parts],axis=0) for k in parts[0]}
    np.savez_compressed(OUT/f'closure_{centre:03d}.npz',**mean)
    mask=(mean['r_km']<=500)[None,:]&(mean['z_km']<=18)[:,None]
    row=dict(hour=centre,numerical_u_closure_max=max(t['full_difference_u_closure'] for t in tests),superposition_max=max(t[k] for t in tests for k in t if 'superposition' in k),stored_matrix_residual_max=max(t[k] for t in tests for k in t if 'stored_total_residual' in k))
    for key in ['ctrl','jet']:row[key+'_SE_vs_actual_relative_error']=relative((mean[key+'_SE']-mean[key+'_actual'])[mask],mean[key+'_actual'][mask])
    row['SE_delta_vs_actual_relative_error']=relative((mean['SE_delta']-mean['actual_delta'])[mask],mean['actual_delta'][mask])
    row['current_sum_vs_SE_delta_relative_error']=relative((mean['current_sum']-mean['SE_delta'])[mask],mean['SE_delta'][mask])
    row['current_sum_vs_actual_delta_relative_error']=relative((mean['current_sum']-mean['actual_delta'])[mask],mean['actual_delta'][mask])
    rows.append(row);print(json.dumps(row),flush=True)
    if centre in [80,110]:
        fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True,constrained_layout=True)
        fields=[mean['actual_delta'],mean['SE_delta'],mean['current_sum'],mean['current_sum']-mean['SE_delta'],mean['full_sum']-mean['SE_delta'],mean['SE_delta']-mean['actual_delta']]
        titles=['Actual JET-CTRL','Full SE JET-CTRL','Thermal + momentum + inertial','Current sum - full SE difference','Exact sum - full SE difference','Full SE difference - actual difference']
        q=np.percentile(mean['actual_delta'][mask],[5,95]);lim=max(abs(q[0]),abs(q[1]),1e-6)
        for ax,f,title in zip(axes.ravel(),fields,titles):
            im=ax.pcolormesh(mean['r_km'],mean['z_km'],f,cmap='RdBu_r',vmin=-lim,vmax=lim,shading='auto');ax.set(xlim=(0,500),ylim=(0,18),title=title,xlabel='Radius (km)')
        axes[0,0].set_ylabel('Height (km)');axes[1,0].set_ylabel('Height (km)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Radial wind difference (m/s)',extend='both');fig.suptitle(f'{centre} h +/-3 h closure audit, same actual-difference P5/P95 scale')
        fig.savefig(OUT/f'closure_comparison_{centre:03d}.png',dpi=180);plt.close(fig)
for name,data in [('window_metrics.csv',rows),('hourly_numerical_tests.csv',hourly_rows)]:
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(OUT/'summary.json').write_text(json.dumps(dict(source=str(ROOT),core_modified=False,method='Existing old.solve unchanged, in-process capture of actual sparse matrix/rhs for residual measurement; existing psi_to_uw common CTRL density',exact_identity='SE wind difference = thermal difference + momentum difference + full operator change + interaction + density conversion term',warning='Current masked smoothed inertial K3 forcing based on actual CTRL wind is not full operator perturbation; closure to SE is not closure to CM1; actual solver uses radial zero-derivative ghost conditions, bottom zero psi ghost, top zero-derivative ghost, not four-side Dirichlet',window_metrics=rows),indent=2))
print('COMPLETE',flush=True)
