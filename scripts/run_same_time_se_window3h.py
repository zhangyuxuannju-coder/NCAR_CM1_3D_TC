"""Same-time window and plot orchestration; existing scientific routines unchanged."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import plot_inertial_operator_evolution_matched as inertial
from scripts import run_se_forcing_operator_factorial_full as sources
from src.se_bui import build_basic_state,invert_balanced_theta,build_forcing
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
DATA=Path('/data/zhangyx/DATA')
a=SimpleNamespace(ctrl=str(DATA/'cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc'),jet=str(DATA/'cm1out_25N_JET_R2_Thompson_OML100_240h.nc'),output_dir=str(OUT),max_r_km=1200.,dr_km=12.,max_z_km=20.,plot_max_z_km=18.,mask_radius_km=100.,f=6.1636e-5,eps_ratio=1e-5,smooth_sigma_z=.75,smooth_sigma_r=1.)
cache=OUT/'sources';cache.mkdir()
def state(label,h):
    name=f'{label}_{h:g}_sources.npz'
    for directory in [Path('output/jet30_matched_se_1200_plot500/cache'),Path('output/inertial_thermal_window3h_20260914_new01/sources'),cache]:
        p=directory/name
        if p.exists():return dict(np.load(p,allow_pickle=True))
    s=sources.azimuthal_average(a.ctrl if label=='CTRL' else a.jet,h,a)
    np.savez_compressed(cache/name,**s);return s
def hourly(h):
    q=inertial.compute_one(h,h,0.,0.,0.,a)
    C,J=state('CTRL',h),state('U30',h)
    r,z=q['r']*1000,q['z']*1000
    basics=[];rhs=[];operators=[];regs=[]
    for s in [C,J]:
        assert np.allclose(s['r_km'],q['r']) and np.allclose(s['z_km'],q['z'])
        theta,_=invert_balanced_theta(s['ut'],s['theta'],r,z,a.f,outer_smooth_window=1)
        b=build_basic_state(s['ut'],theta,s['rho'],r,z,a.f)
        _,op,reg=inertial.build_ctrl_operator(b,r,z,a)
        basics.append(b);operators.append(op);regs.append(reg)
        rhs.append(build_forcing(b,s['Q'],s['Fnu'],r,z))
    result=dict(r_km=q['r'],z_km=q['z'],jet_ur=q['ur_jet'],ctrl_ur=q['ur_ctrl'],delta_ur=q['ur_jet']-q['ur_ctrl'],inertial_rhs=q['forcing'],inertial_u=q['u_se'],inertial_w=q['w_se'],inertial_psi=q['psi'])
    for i,label in enumerate(['ctrl','jet']):
        v=inertial.solve(operators[i],rhs[i]['forcing_total'],basics[i]['rho'],r,z)
        for key,outkey in [('psi','psi'),('u','u'),('w','w')]:result[label+'_'+outkey]=v[key]
        for key in ['thermal','momentum']:result[label+'_'+key+'_rhs']=rhs[i]['forcing_'+key]
        for key in ['raw_nonelliptic_fraction','changed_coefficient_fraction']:result[label+'_'+key]=regs[i][key]
    for key in ['thermal','momentum']:
        delta=rhs[1]['forcing_'+key]-rhs[0]['forcing_'+key]
        v=inertial.solve(operators[0],delta,basics[0]['rho'],r,z)
        result[key+'_rhs']=delta
        for vkey in ['psi','u','w']:result[key+'_'+vkey]=v[vkey]
    assert all(np.all(np.isfinite(v)) for v in result.values())
    np.savez_compressed(OUT/f'hourly_{h:03d}.npz',**result)
    return result
def panel(fig,ax,m,key,title,wind,limit=None,flow=False):
    r,z=m['r_km'],m['z_km'];field=m[key]
    mask=(z<=18)[:,None]&(r<=500)[None,:]
    if limit is None:limit=max(float(np.max(abs(field[mask]))),1e-30)
    im=ax.pcolormesh(r,z,field,cmap=inertial.CMAP,shading='auto',vmin=-limit,vmax=limit)
    inertial.contour_radial_wind(ax,r,z,m[wind])
    if flow:
        wz=m[key[:-1]+'w'];iz=np.where(z<=18)[0][::4];ir=np.where(r<=500)[0][::3]
        ax.quiver(r[ir],z[iz],field[np.ix_(iz,ir)],wz[np.ix_(iz,ir)],color='0.3',width=.002)
    ax.set(xlim=(0,500),ylim=(0,18),title=title,xlabel='Radius (km)')
    fig.colorbar(im,ax=ax,shrink=.75)
completed=[]
for h in range(50,151,10):
    print('START WINDOW',h,flush=True)
    records=[hourly(t) for t in range(h-3,h+4)]
    mean={k:np.mean([x[k] for x in records],axis=0) for k in records[0]}
    np.savez_compressed(OUT/f'window_{h:03d}.npz',**mean,hours=np.arange(h-3,h+4))
    fig,axes=plt.subplots(2,4,figsize=(20,9),sharex=True,sharey=True)
    for row,label in enumerate(['ctrl','jet']):
        for col,(key,title) in enumerate([('psi','SE streamfunction (kg/s)'),('u','SE flow: colour ur (m/s), arrows ur/w'),('thermal_rhs','Thermal RHS (K^-1 s^-3)'),('momentum_rhs','Momentum RHS (K^-1 s^-3)')]):
            panel(fig,axes[row,col],mean,label+'_'+key,label.upper()+' '+title,label+'_ur',flow=key=='u')
        axes[row,0].set_ylabel('Height (km)')
    fig.suptitle(f'Same-time CTRL / JET30 | {h} h +/-3 h response mean | black: actual case radial wind')
    fig.tight_layout();fig.savefig(OUT/f'{h:03d}_ctrl_jet.png',dpi=180);plt.close(fig)
    for wind in ['jet_ur','delta_ur']:
        fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
        mask=(mean['z_km']<=18)[:,None]&(mean['r_km']<=500)[None,:]
        limit=max([float(np.max(abs(mean[k+'_u'][mask]))) for k in ['thermal','momentum','inertial']]+[1e-6])
        for col,key in enumerate(['thermal','momentum','inertial']):
            panel(fig,axes[0,col],mean,key+'_rhs',key+' difference/equivalent RHS (K^-1 s^-3)',wind)
            panel(fig,axes[1,col],mean,key+'_u',key+' response ur (m/s), arrows ur/w',wind,limit=limit,flow=True)
            if key=='inertial':
                for row in range(2):axes[row,col].axvline(100,color='purple',ls='--',lw=.8)
        axes[0,0].set_ylabel('Height (km)');axes[1,0].set_ylabel('Height (km)')
        fig.suptitle(f'{h} h +/-3 h response mean | black: {wind} | full solve 1200 km, view 500 km')
        fig.tight_layout();fig.savefig(OUT/f'{h:03d}_responses_{wind}.png',dpi=180);plt.close(fig)
    completed.append(h);(OUT/'progress.json').write_text(json.dumps(completed))
    print('DONE WINDOW',h,flush=True)
(OUT/'summary.json').write_text(json.dumps(dict(ctrl=a.ctrl,jet=a.jet,centres_h=completed,window='7 hourly same-time pairs per centre, solve first then mean; no intensity matching or overall time mean',solve_radius_km=1200,view_radius_km=500,f=a.f,eps_ratio=a.eps_ratio,solver='Existing mean-layer-spacing sparse solve unchanged; homogeneous Dirichlet; no solver residual output',operator='Inertial K3_raw equivalent forcing only, existing spatial smoothing and 100 km RHS cutoff; not full operator difference',forcing='Full own-state thermal/momentum RHS, differences JET minus CTRL projected with CTRL operator; no RHS cutoff for thermal/momentum',flow_arrows='ur/w direction and relative distribution, display coordinates km, not physical particle trajectories',evidence='Regularized balanced projection, not actual wind difference or causal proof; inspect anomalous amplitudes and differing masks',plots='Per centre: CTRL/JET 2x4 plus 2x3 forcing/response panels with actual JET wind and actual wind difference overlays'),indent=2))
print('COMPLETE',flush=True)
