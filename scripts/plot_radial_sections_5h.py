"""Plot-only wrapper; reuse existing radial extraction and hourly caches."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts import animate_r2_ctrl_jet_radial as old
import numpy as np
import matplotlib.pyplot as plt
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
old.OUT=OUT;old.CACHE=OUT/'cache';old.CACHE.mkdir()
records=[]
def load(case,h):
    same=Path('output/same_time_se_window3h_50_150_new01')/f'hourly_{h:03d}.npz'
    if same.exists():
        d=np.load(same);return dict(r_km=d['r_km'],z_km=d['z_km'],ur=d['ctrl_ur' if case=='CTRL' else 'jet_ur'],source=str(same))
    label='CTRL' if case=='CTRL' else 'U30'
    for directory in ['output/jet30_matched_se_1200_plot500/cache','output/inertial_thermal_window3h_20260914_new01/sources','output/same_time_se_window3h_50_150_new01/sources']:
        p=Path(directory)/f'{label}_{h:g}_sources.npz'
        if p.exists():
            d=np.load(p,allow_pickle=True);return dict(r_km=d['r_km'],z_km=d['z_km'],ur=d['ur'],source=str(p))
    d=old.extract(case,old.CASES[case],h)
    return dict(r_km=d['r_km'],z_km=d['z_km'],ur=d['ur'],source='Existing animate_r2_ctrl_jet_radial.extract')
for h in range(50,151,5):
    print('TIME',h,flush=True)
    c,j=load('CTRL',h),load('JET',h)
    r,z=c['r_km'],c['z_km'];rm=r<=500;zm=z<=20
    assert np.allclose(r[rm],j['r_km'][j['r_km']<=500]) and np.allclose(z[zm],j['z_km'][j['z_km']<=20])
    u=c['ur'][zm][:,rm];v=j['ur'][j['z_km']<=20][:,j['r_km']<=500];delta=v-u
    np.savez_compressed(OUT/f'radial_{h:03d}h.npz',r_km=r[rm],z_km=z[zm],ctrl_ur=u,jet_ur=v,delta_ur=delta)
    fig,axes=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True,constrained_layout=True)
    for ax,field,label in zip(axes,[u,v,delta],['CTRL','JET30','JET30 - CTRL']):
        im=ax.pcolormesh(r[rm],z[zm],field,cmap='RdBu_r',vmin=-15,vmax=15,shading='auto')
        ax.contour(r[rm],z[zm],field,levels=[0],colors='0.3',linewidths=.5)
        ax.set(xlim=(0,500),ylim=(0,20),xlabel='Radius (km)',title=label)
    axes[0].set_ylabel('Height (km)')
    fig.colorbar(im,ax=axes,label='Radial wind (m/s; outward +)',extend='both',shrink=.85)
    fig.suptitle(f'Storm-centred azimuthal-mean radial wind | {h} h | instantaneous')
    fig.savefig(OUT/f'radial_{h:03d}h.png',dpi=180);plt.close(fig)
    records.append(dict(hour=h,ctrl_source=c['source'],jet_source=j['source']))
(OUT/'metadata.json').write_text(json.dumps(dict(inputs=old.CASES,hours=list(range(50,151,5)),time_average=False,radius_km=500,height_km=20,colour_scale_m_s=[-15,15],all_panels_and_times_same_scale=True,definition='Existing storm-centred azimuthal mean radial wind; no SE solve and no storm-motion subtraction',records=records),indent=2))
print('COMPLETE',flush=True)
