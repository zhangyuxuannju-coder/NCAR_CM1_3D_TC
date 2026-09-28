"""Window/plot wrapper only; use existing inertial computation and SE solver."""
import sys,json,csv
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import plot_inertial_operator_evolution_matched as inertial
from scripts import run_se_forcing_operator_factorial_full as sources
from src.se_bui import build_forcing,build_basic_state,invert_balanced_theta

OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
DATA=Path('/data/zhangyx/DATA')
a=SimpleNamespace(ctrl=str(DATA/'cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc'),jet=str(DATA/'cm1out_25N_JET_R2_Thompson_OML100_240h.nc'),output_dir=str(OUT),max_r_km=1200.,dr_km=12.,max_z_km=20.,plot_max_z_km=18.,mask_radius_km=100.,f=6.1636e-5,eps_ratio=1e-5,smooth_sigma_z=.75,smooth_sigma_r=1.)
matches=[x for x in json.loads(Path('output/jet30_matched_se_1200_plot500/matching_audit.json').read_text()) if x['accepted']]
(OUT/'matches.json').write_text(json.dumps(matches,indent=2))
oldcache=Path('output/jet30_matched_se_1200_plot500/cache')
newcache=OUT/'sources';newcache.mkdir()
def state(label,h):
    name=f'{label}_{h:g}_sources.npz'
    for p in [oldcache/name,newcache/name]:
        if p.exists():return dict(np.load(p,allow_pickle=True))
    path=a.ctrl if label=='CTRL' else a.jet
    result=sources.azimuthal_average(path,h,a);np.savez_compressed(newcache/name,**result);return result
def hourly(j,c):
    out=OUT/f'hourly_J{j}_C{c}.npz'
    if out.exists():return dict(np.load(out))
    q=inertial.compute_one(j,c,0.,0.,0.,a)
    C,J=state('CTRL',c),state('U30',j)
    r,z=q['r']*1000,q['z']*1000
    for s in [C,J]:
        assert np.allclose(s['r_km'],q['r']) and np.allclose(s['z_km'],q['z'])
    bc,bj=[] ,[]
    basics=[]
    for s in [C,J]:
        theta,_=invert_balanced_theta(s['ut'],s['theta'],r,z,a.f,outer_smooth_window=1)
        basics.append(build_basic_state(s['ut'],theta,s['rho'],r,z,a.f))
    bc,bj=basics
    fc=build_forcing(bc,C['Q'],C['Fnu'],r,z)
    fj=build_forcing(bj,J['Q'],J['Fnu'],r,z)
    delta=fj['forcing_thermal']-fc['forcing_thermal']
    basic,operator,reg=inertial.build_ctrl_operator(bc,r,z,a)
    response=inertial.solve(operator,delta,basic['rho'],r,z)
    result=dict(r_km=q['r'],z_km=q['z'],inertial_u=q['u_se'],inertial_w=q['w_se'],inertial_psi=q['psi'],inertial_rhs=q['forcing'],thermal_u=response['u'],thermal_w=response['w'],thermal_psi=response['psi'],thermal_rhs=delta,jet_ur=q['ur_jet'],delta_ur=q['ur_jet']-q['ur_ctrl'],raw_nonelliptic_fraction=reg['raw_nonelliptic_fraction'],changed_coefficient_fraction=reg['changed_coefficient_fraction'])
    assert all(np.all(np.isfinite(v)) for v in result.values())
    np.savez_compressed(out,**result);return result
metrics=[]
for rec in matches:
    j,c=rec['jet_h'],rec['ctrl_h'];print('START WINDOW',j,c,flush=True)
    records=[hourly(j+d,c+d) for d in range(-3,4)]
    mean={k:np.mean([v[k] for v in records],axis=0) for k in records[0]}
    np.savez_compressed(OUT/f"{rec['phase']}_J{j:g}_C{c:g}_window.npz",**mean,jet_hours=np.arange(j-3,j+4),ctrl_hours=np.arange(c-3,c+4))
    r,z=mean['r_km'],mean['z_km'];maskr=r<=500
    for region,maskz in [('outflow_10_17',(z>=10)&(z<=17)),('whole_0_18',z<=18)]:
        row=dict(phase=rec['phase'],jet_h=j,ctrl_h=c,region=region,raw_nonelliptic_fraction=float(mean['raw_nonelliptic_fraction']),changed_coefficient_fraction=float(mean['changed_coefficient_fraction']))
        weight=np.gradient(z)[maskz,None]*r[None,maskr]*np.gradient(r)[None,maskr]
        for key in ['inertial','thermal']:
            field=mean[key+'_u'][maskz][:,maskr];row[key+'_rms_u']=float(np.sqrt(np.sum(weight*field**2)/np.sum(weight)))
        metrics.append(row)
    for overlay in ['jet_ur','delta_ur']:
        fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
        view=(z<=18)[:,None]&maskr[None,:]
        limit=max(float(np.max(abs(mean[k+'_u'][view]))) for k in ['inertial','thermal']);limit=max(limit,1e-6)
        for ax,key in zip(axes,['inertial','thermal']):
            im=ax.pcolormesh(r,z,mean[key+'_u'],shading='auto',cmap=inertial.CMAP,vmin=-limit,vmax=limit)
            inertial.contour_radial_wind(ax,r,z,mean[overlay])
            if key=='inertial':ax.axvline(100,color='purple',ls='--',lw=.8)
            ax.set(xlim=(0,500),ylim=(0,18),xlabel='Radius (km)',title=key+' response')
        axes[0].set_ylabel('Height (km)');fig.suptitle(f'JET30 {j:g} h / CTRL {c:g} h | +/-3 h response mean | black: {overlay}')
        fig.subplots_adjust(right=.85);fig.colorbar(im,cax=fig.add_axes([.88,.16,.02,.65]),label='SE radial response (m/s)')
        fig.savefig(OUT/f"{rec['phase']}_J{j:g}_C{c:g}_{overlay}.png",dpi=180,bbox_inches='tight');plt.close(fig)
    with (OUT/'comparison_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    print('DONE WINDOW',j,c,flush=True)
(OUT/'summary.json').write_text(json.dumps(dict(ctrl=a.ctrl,jet=a.jet,matches=matches,solve_radius_km=1200,plot_radius_km=500,eps_ratio=a.eps_ratio,f=a.f,window='Hourly offsets -3 through +3, solve first then average; central matches retained, no hourly rematching; phase-edge windows extend outside phase',inertial='Existing compute_one unchanged: smoothed K3_raw difference and CTRL ur; RHS zero below 100 km',thermal='Own-state full Bui thermal RHS JET minus CTRL; no radial forcing mask; same existing regularized CTRL operator and solve as inertial',solver='Existing mean-layer-spacing sparse solver unchanged, homogeneous Dirichlet; residual not reported by existing solver',warning='Modified-basic-state balanced projection, not actual circulation or unique cause; differing forcing masks must be considered when comparing response amplitudes',panels=metrics),indent=2))
print('COMPLETE',flush=True)
