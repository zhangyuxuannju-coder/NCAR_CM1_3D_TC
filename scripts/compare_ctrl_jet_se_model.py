"""Compare cached CTRL flow with corrected CTRL and JET SE solutions."""
import sys,json,csv
from pathlib import Path
source=Path('scripts/run_complete_se_attribution.py').read_text()
prefix=source[:source.index("names=['thermal'")].replace('OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)','OUT=None')
exec(compile(prefix,'existing_cache_helpers','exec'),globals())
from scripts.analyze_secondary_circulation_pathways import fit_psi
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
BASE=Path('output/se_boundary_fixed_baseline_new01')
rows=[]
for h in range(50,151,10):
    print('START CTRL COMPARE',h,flush=True)
    a=dict(np.load(BASE/f'window_{h:03d}.npz'));states=[state('CTRL',t) for t in range(h-3,h+4)]
    r,z=a['r_km']*1000,a['z_km']*1000
    assert all(np.allclose(s['r_km']*1000,r) and np.allclose(s['z_km']*1000,z) for s in states)
    fr=np.mean([s['rho']*s['ur'] for s in states],axis=0);fz=np.mean([s['rho']*s['w'] for s in states],axis=0)
    p,fit_error,fit_info=fit_psi(fr,fz,r,z)
    # Common gauge at lowest-level outer-radius point; raw SE psi retained separately.
    model=p-p[0,-1];se=a['ctrl_psi']-a['ctrl_psi'][0,-1]
    jet_se=a['jet_psi']-a['jet_psi'][0,-1]
    model_u=np.mean([s['ur'] for s in states],axis=0);se_u=a['ctrl_u'];jet_se_u=a['jet_u']
    assert np.allclose(model_u,a['ctrl_ur'])
    d=dict(r_km=r/1000,z_km=z/1000,model_psi=model,se_psi=se,se_psi_raw=a['ctrl_psi'],
           jet_se_psi=jet_se,jet_se_psi_raw=a['jet_psi'],model_u=model_u,se_u=se_u,jet_se_u=jet_se_u)
    np.savez_compressed(OUT/f'ctrl_comparison_{h:03d}.npz',**d)
    rr,zz=np.meshgrid(r/1000,z/1000);mask=(rr<=500)&(zz<=18)
    row=dict(hour=h,model_mass_flux_fit_nrmse=fit_error,fit_stop=fit_info['stop'],fit_iterations=fit_info['iterations'])
    for key in ['psi','u']:
        x,y=d['model_'+key][mask],d['se_'+key][mask]
        row[key+'_model_maxabs']=float(np.max(np.abs(x)));row[key+'_SE_maxabs']=float(np.max(np.abs(y)))
        row[key+'_relative_L2_error']=rel(y-x,x);row[key+'_RMSE']=float(np.sqrt(np.mean((y-x)**2)))
        row[key+'_pattern_correlation']=float(np.corrcoef(x,y)[0,1])
    rows.append(row)
    dr=np.arange(r[0]/1000,500.001,2.);dz=np.arange(z[0]/1000,18.001,.1);R,Z=np.meshgrid(dr,dz);points=np.c_[Z.ravel(),R.ravel()]
    def display(f):return RegularGridInterpolator((z/1000,r/1000),f,method='linear',bounds_error=True)(points).reshape(Z.shape)
    for scale in ['shared_scale','structure_scale']:
        fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True,sharey=True)
        for rownum,key in enumerate(['psi','u']):
            fields=[d['model_'+key],d['se_'+key],d['jet_se_'+key]];unit=1e9 if key=='psi' else 1.
            shared=max(float(np.max(np.abs(f[mask])))/unit for f in fields)
            for col,(field,label) in enumerate(zip(fields,['CTRL CM1 cached flow','CTRL Corrected SE','JET Corrected SE'])):
                f=field/unit;lim=max(shared if scale=='shared_scale' else float(np.max(np.abs(f[mask]))),1e-10)
                levels=np.linspace(-lim,lim,31)
                if key=='psi':
                    im=axes[rownum,col].contour(dr,dz,display(f),levels=levels,cmap='turbo',linewidths=.85)
                    axes[rownum,col].clabel(im,im.levels[::5],fontsize=7,fmt='%.2g')
                else:im=axes[rownum,col].contourf(dr,dz,display(f),levels=levels,cmap='RdBu_r',extend='both')
                fig.colorbar(im,ax=axes[rownum,col],shrink=.8,label='psi (10^9 kg/s; no 2pi)' if key=='psi' else 'ur (m/s; outward +)')
                axes[rownum,col].set(xlim=(0,500),ylim=(0,18),title=label,xlabel='Radius (km)',ylabel='Height (km)')
        fig.suptitle(f'CTRL CM1 / CTRL SE / JET SE | {h} h +/-3 h | '+('Common scale per row: amplitude comparison' if scale=='shared_scale' else 'Independent scales: structure only, compare numeric colourbars')+'\nSE solved 1200 km; cached-flow kinematic psi fit, not SE; display interpolation 2 km x 0.1 km')
        fig.tight_layout();fig.savefig(OUT/f'ctrl_cm1_ctrl_se_jet_se_{h:03d}_{scale}.png',dpi=150);plt.close(fig)
    print(json.dumps(rows[-1]),flush=True)
with (OUT/'comparison_metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(dict(ctrl='/data/zhangyx/DATA/cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc',jet='/data/zhangyx/DATA/cm1out_25N_JET_R2_Thompson_OML100_240h.nc',SE_cache=str(BASE),centres_h=list(range(50,151,10)),window='same-time +/-3 hourly samples; existing hourly SE solutions averaged',columns=['CTRL cached CM1 flow','CTRL corrected SE','JET corrected SE'],model_psi='Existing analyze_secondary_circulation_pathways.fit_psi least-squares kinematic reconstruction from mean(rho_bar*ur_bar) and mean(rho_bar*w_bar) over full cached 1200 km domain. Not SE or literal model-output psi variable.',gauge='Each displayed psi shifted by its own lowest-level outer-radius value; raw CTRL/JET SE psi retained',model_u='Arithmetic mean of actual storm-centred azimuthal-average CTRL ur; unchanged cached output',mass_flux_approximation='rho_bar*u_bar used, not azimuthal mean(rho*u); density-wind covariance omitted, no giant 3D reread',units='psi kg/s with u=-psi_z/(rho*r), w=psi_r/(rho*r); no 2pi factor on any psi',grid='Original 49x100 grid; SE still mean dz; fit_psi uses actual nonuniform coordinates',display='Linear 2 km radial x 0.1 km vertical, no extrapolation; view 0-500 km,0-18 km',scales='Shared-scale figures compare magnitude; independent-scale figures compare structure only. All limits native maxima in view, no percentile clipping.',limitations=['Kinematic psi is a best fit when mass continuity is not steady/closed; fit mismatch saved','Current sparse Neumann boundary repaired, not physical stress-free impermeable BC','SE mean-dz approximation unresolved; regularization/balance may cause amplitude mismatch'],core_modified=False),indent=2))
print('COMPLETE CTRL COMPARISON',flush=True)
