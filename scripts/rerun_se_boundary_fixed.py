"""Re-solve existing hourly RHS, then reuse complete attribution/plots unchanged."""
import sys,json,csv
from pathlib import Path
source=Path('scripts/run_complete_se_attribution.py').read_text()
prefix=source[:source.index("names=['thermal'")].replace('OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)','OUT=None')
exec(compile(prefix,'existing_complete_helpers','exec'),globals())
BASE=Path('output/se_boundary_fixed_baseline_new01');BASE.mkdir(exist_ok=False)
VALID=Path('output/se_boundary_fix_validation_new01')
OLD=ROOT
comparison=[];residuals=[]
for centre in range(50,151,10):
    records=[]
    for h in range(centre-3,centre+4):
        archived=dict(np.load(OLD/f'hourly_{h:03d}.npz'))
        a={k:v for k,v in archived.items() if not k.startswith('inertial_')}
        C,J=state('CTRL',h),state('U30',h);r,z=a['r_km']*1000,a['z_km']*1000;ops=[]
        for s in [C,J]:
            theta,_=invert_balanced_theta(s['ut'],s['theta'],r,z,args.f,outer_smooth_window=1)
            b=build_basic_state(s['ut'],theta,s['rho'],r,z,args.f);_,op,reg=old.build_ctrl_operator(b,r,z,args);ops.append(op)
        for op,s,case in zip(ops,[C,J],['ctrl','jet']):
            rhs=a[case+'_thermal_rhs']+a[case+'_momentum_rhs'];v=old.solve(op,rhs,s['rho'],r,z)
            for k in ['psi','u','w']:a[case+'_'+k]=v[k]
            mc=matrix(op,r,z,s['rho']);res=rel(mc@v['psi'].T.ravel()-rhs.T.ravel(),rhs.T.ravel())
            residuals.append(dict(hour=h,case=case,relative_residual=res,top_ur_max=float(np.max(np.abs(v['u'][-1]))),side_w_max=float(np.max(np.abs(v['w'][:,[0,-1]])))))
            mask=(a['r_km']<=500)[None,:]&(a['z_km']<=18)[:,None]
            comparison.append(dict(hour=h,case=case,old_max_abs_u=float(np.max(np.abs(archived[case+'_u']))),new_max_abs_u=float(np.max(np.abs(v['u']))),old_inner_max_abs_u=float(np.max(np.abs(archived[case+'_u'][mask]))),new_inner_max_abs_u=float(np.max(np.abs(v['u'][mask]))),old_top_band_max_abs_u=float(np.max(np.abs(archived[case+'_u'][a['z_km']>=17]))),new_top_band_max_abs_u=float(np.max(np.abs(v['u'][a['z_km']>=17])))))
        for key in ['thermal','momentum']:
            v=old.solve(ops[0],a[key+'_rhs'],C['rho'],r,z)
            for k in ['psi','u','w']:a[key+'_'+k]=v[k]
        np.savez_compressed(BASE/f'hourly_{h:03d}.npz',**a);records.append(a)
    mean={k:np.mean([x[k] for x in records],axis=0) for k in records[0]}
    np.savez_compressed(BASE/f'window_{centre:03d}.npz',**mean)
    print('BASELINE RE-SOLVED',centre,flush=True)
for name,data in [('boundary_residual_checks.csv',residuals),('old_new_amplitudes.csv',comparison)]:
    with (VALID/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
# Plot old/new mean fields with identical limits and display interpolation.
for h in [60,80,110]:
    a=dict(np.load(OLD/f'window_{h:03d}.npz'));b=dict(np.load(BASE/f'window_{h:03d}.npz'));r,z=a['r_km'],a['z_km'];rr,zz=np.meshgrid(r,z);mask=(rr<=500)&(zz<=18)
    dr=np.arange(r[0],500.001,2.);dz=np.arange(z[0],18.001,.1);R,Z=np.meshgrid(dr,dz);points=np.c_[Z.ravel(),R.ravel()]
    q=np.percentile(a['delta_ur'][mask],[5,95]);limit=max(abs(q[0]),abs(q[1]),1e-6)
    fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
    for row,case in enumerate(['ctrl','jet']):
        for col,(field,title) in enumerate([(a[case+'_u'],'Before'),(b[case+'_u'],'Corrected BC'),(b[case+'_u']-a[case+'_u'],'Corrected - before')]):
            f=RegularGridInterpolator((z,r),field,method='linear',bounds_error=True)(points).reshape(Z.shape)
            im=axes[row,col].contourf(dr,dz,f,levels=np.linspace(-limit,limit,31),cmap='RdBu_r',extend='both');axes[row,col].set(xlim=(0,500),ylim=(0,18),title=case.upper()+' '+title,xlabel='Radius (km)',ylabel='Height (km)');fig.colorbar(im,ax=axes[row,col])
    fig.suptitle(f'{h} h +/-3 h | existing RHS and regularization unchanged | radial wind m/s | common actual-difference scale (saturated)')
    fig.tight_layout();fig.savefig(VALID/f'boundary_before_after_{h:03d}.png',dpi=150);plt.close(fig)
(VALID/'boundary_fix_summary.json').write_text(json.dumps(dict(changes=['Radial mixed-derivative cancellation','Outer radial first derivative zero','Centered mirrored top ghost, matching second/mixed/first derivatives','Wind conversion: side w=0, top interior u=0, inner-radius u from psi derivative'],unchanged=['Bui formulation','RHS and actual fields','Regularization eps=1e-5','1200 km solve, 12 km radial grid','Mean-dz discretization'],remaining=['Native z grid is nonuniform, but existing sparse solver still uses mean dz; this numerical approximation was not silently replaced','Separate SOR pathway has different radial node layout and was not changed; current rerun uses sparse path only','Streamfunction Neumann conditions are not velocity stress-free impermeable boundaries'],max_relative_residual=max(x['relative_residual'] for x in residuals),max_top_ur=max(x['top_ur_max'] for x in residuals),max_side_w=max(x['side_w_max'] for x in residuals)),indent=2))
# Reuse the prior complete attribution driver with corrected baseline and ghost layout.
source=source.replace("ROOT=Path('output/same_time_se_window3h_50_150_new01')",f"ROOT=Path('{BASE}')")
source=source.replace("[ROOT/'sources',Path('output/jet30", "[Path('output/same_time_se_window3h_50_150_new01/sources'),ROOT/'sources',Path('output/jet30")
source=source.replace('ext[:,-1]=ext[:,-2]','ext[:,-1]=ext[:,-3]')
source=source.replace('Existing streamfunction Neumann radial/top ghosts, zero bottom ghost; not established as physical impermeable stress-free velocity BC; mixed-derivative stencil issue remains','Corrected mirrored streamfunction Neumann at radial/top physical nodes, zero bottom ghost; velocity conversion consistent. Not velocity stress-free impermeable BC. Separate SOR and mean-dz issues remain.')
snapshot=VALID/'complete_attribution_boundary_fixed.py';snapshot.write_text(source)
exec(compile(source,str(snapshot),'exec'),globals())
