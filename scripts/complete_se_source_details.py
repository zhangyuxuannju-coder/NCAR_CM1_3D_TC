"""Reuse complete orchestration functions and existing Bui source builder."""
import sys,json,csv
from pathlib import Path
source=Path('scripts/run_complete_se_attribution.py').read_text()
prefix=source[:source.index("names=['thermal'")]
exec(compile(prefix,'reused_complete_se_helpers','exec'),globals())
from src.se_bui import build_forcing
REFERENCE=Path('output/se_complete_forcing_operator_window3h_new01')
groups={'Q_eddy':'thermal','Q_diffusion':'thermal','Q_diabatic':'thermal','Q_other_model':'thermal','F_lambda_eddy':'momentum','F_lambda_diffusion':'momentum','F_lambda_other_model':'momentum'}
DETAIL=OUT/'source_details';DETAIL.mkdir(exist_ok=False)
checks=[];terms_used=[]
for centre in range(50,151,10):
    records=[]
    for h in range(centre-3,centre+4):
        a=dict(np.load(ROOT/f'hourly_{h:03d}.npz'));states=[state('CTRL',h),state('U30',h)];r,z=a['r_km']*1000,a['z_km']*1000;basics=[]
        for s in states:
            theta,_=invert_balanced_theta(s['ut'],s['theta'],r,z,args.f,outer_smooth_window=1)
            basics.append(build_basic_state(s['ut'],theta,s['rho'],r,z,args.f))
        for case,s in zip(['CTRL','JET'],states):
            terms_used.append(dict(hour=h,case=case,thermal=s['thermal_budget_terms_used'].tolist(),momentum=s['momentum_budget_pairs_used'].tolist()))
        _,op,_=old.build_ctrl_operator(basics[0],r,z,args);mc=matrix(op,r,z,states[0]['rho']);fac=spla.factorized(mc.tocsc());zero=np.zeros_like(states[0]['rho'])
        rec={}
        for key,kind in groups.items():
            rhs=[]
            for s,b in zip(states,basics):
                f=build_forcing(b,s[key] if kind=='thermal' else zero,s[key] if kind=='momentum' else zero,r,z)
                rhs.append(f['forcing_'+kind])
            rec[key+'_rhs']=rhs[1]-rhs[0]
            spla.spsolve=lambda m,b,**kwargs:fac(b)
            try:v=old.solve(op,rec[key+'_rhs'],states[0]['rho'],r,z)
            finally:spla.spsolve=original
            for k in ['psi','u','w']:rec[key+'_'+k]=v[k]
        for kind in ['thermal','momentum']:
            total=sum(rec[k+'_rhs'] for k in groups if groups[k]==kind)
            checks.append(dict(hour=h,kind=kind,RHS_sum_relative_error=rel(total-a[kind+'_rhs'],a[kind+'_rhs']),response_sum_relative_error=rel(sum(rec[k+'_u'] for k in groups if groups[k]==kind)-a[kind+'_u'],a[kind+'_u'])))
        np.savez_compressed(DETAIL/f'hourly_{h:03d}.npz',**rec);records.append(rec)
    m=dict(np.load(REFERENCE/f'window_{centre:03d}.npz'));m.update({k:np.mean([q[k] for q in records],axis=0) for k in records[0]})
    np.savez_compressed(DETAIL/f'window_{centre:03d}.npz',**m)
    r,z=m['r_km'],m['z_km'];rr,zz=np.meshgrid(r,z);mask=(rr<=500)&(zz<=18)
    dr=np.arange(r[0],500.001,2);dz=np.arange(z[0],18.001,.1);R,Z=np.meshgrid(dr,dz);points=np.c_[Z.ravel(),R.ravel()]
    def display(f):return RegularGridInterpolator((z,r),f,method='linear',bounds_error=True)(points).reshape(Z.shape)
    q=np.percentile(m['delta_ur'][mask],[5,95]);ulim=max(abs(q[0]),abs(q[1]),1e-6)
    for wind in ['jet_ur','delta_ur']:
        fig,axes=plt.subplots(2,7,figsize=(28,9),sharex=True,sharey=True)
        for col,key in enumerate(groups):
            for row,kind in enumerate(['rhs','u']):
                f=m[key+'_'+kind];lim=max(np.percentile(np.abs(f[mask]),99),1e-30) if row==0 else ulim
                im=axes[row,col].contourf(dr,dz,display(f),levels=np.linspace(-lim,lim,31),cmap='RdBu_r',extend='both')
                old.contour_radial_wind(axes[row,col],dr,dz,display(m[wind]));axes[row,col].set(xlim=(0,500),ylim=(0,18),title=key+(' RHS' if row==0 else ' response ur'),xlabel='Radius (km)');fig.colorbar(im,ax=axes[row,col],shrink=.8)
        axes[0,0].set_ylabel('Height (km)');axes[1,0].set_ylabel('Height (km)')
        fig.suptitle(f'{centre} h +/-3 h | all cached Bui source groups | black: {wind} | RHS K^-1 s^-3; response m/s; display-only linear interpolation')
        fig.tight_layout();fig.savefig(DETAIL/f'{centre:03d}_all_sources_{wind}.png',dpi=125);plt.close(fig)
    print('SOURCE DETAILS DONE',centre,flush=True)
with (DETAIL/'source_sum_checks.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
(DETAIL/'summary.json').write_text(json.dumps(dict(groups=groups,terms_used=terms_used,max_RHS_sum_relative_error=max(q['RHS_sum_relative_error'] for q in checks),max_response_sum_relative_error=max(q['response_sum_relative_error'] for q in checks),warning='Unavailable model budget terms may be zero; terms_used lists what existed. Q_dtheta_dt and model advective closure checks are not additional independent Bui forcing terms; including them would double count.'),indent=2))
metrics=list(csv.DictReader((REFERENCE/'regional_metrics.csv').open()))
metrics=[{k:(v if k=='region' else float(v)) for k,v in q.items()} for q in metrics]
block=source[source.index('fig,axes=plt.subplots(2,2'):source.index('summary=dict(ctrl=')]
block=block.replace("if q['region']==region","if q['region']==region and q['hour']>=80").replace("inflow_outflow_trends.png","inflow_outflow_trends_80_150.png")
exec(compile(block,'reused_trends_zoom','exec'),globals())
print('COMPLETE SOURCE DETAILS',flush=True)
