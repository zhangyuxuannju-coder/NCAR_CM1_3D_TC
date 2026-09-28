#!/usr/bin/env python3
"""100-km disk mean/eddy/thermodynamic surface-pressure tendency time series."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_inner_core_surface_pressure_tendency import one_time,DYNAMIC,SOURCE
MEAN=DYNAMIC[:4]; EDDY=DYNAMIC[4:]
def disk_average(v,r,R):
    rr=np.concatenate(([0.],r));vv=np.concatenate(([v[0]],v))
    return 2/R**2*np.trapezoid(vv*rr,rr)
def window_mean(values,time): return np.trapezoid(values,time,axis=0)/(time[-1]-time[0])
def baseline_tracks(path):
    rows=list(csv.DictReader(open(path,encoding='utf-8')));out={}
    for case in ('CTRL','JET'):
      q=[r for r in rows if r['case']==case];t=np.array([float(r['time_h'])*3600 for r in q]);c=np.array([[float(r['center_x_km'])*1000,float(r['center_y_km'])*1000] for r in q]);out[case]=(t,c,np.gradient(c,t,axis=0))
    return out
def run(cfg):
    base=json.loads((ROOT/cfg['base_config']).read_text(encoding='utf-8-sig'));out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    tracks=baseline_tracks(ROOT/cfg['baseline_centres']);R=cfg['radius_km']*1000;dr=cfg['radial_spacing_km']*1000;r=np.arange(dr,R+0.5*dr,dr)
    centres=np.arange(cfg['start_hour'],cfg['end_hour']+1,cfg['center_stride_h']);needed=np.arange(cfg['start_hour']-cfg['half_window_h'],cfg['end_hour']+cfg['half_window_h']+1)
    result=[]
    for case,path in base['cases'].items():
      tt,cc,vv=tracks[case]
      with Dataset(path) as ds:
        model_t=np.asarray(ds['time'][:],float);hour=model_t/3600;cache={}
        for h in needed:
          ix=np.where(np.isclose(hour,h))[0]
          if len(ix)!=1:raise RuntimeError(f'{case}: missing {h} h')
          it=int(ix[0]);cache[h]=one_time(ds,it,cc[it],vv[it],r,cfg['n_azimuth']);print(case,h,'done',flush=True)
        for t0 in centres:
          hs=np.arange(t0-cfg['half_window_h'],t0+cfg['half_window_h']+1);secs=hs*3600
          avg={name:window_mean(np.stack([cache[h][name] for h in hs]),secs)*36 for name in DYNAMIC+SOURCE}
          mean=sum(avg[n] for n in MEAN);eddy=sum(avg[n] for n in EDDY);thermo=sum(avg[n] for n in SOURCE);calc=mean+eddy+thermo
          actual=(cache[hs[-1]]['psfc_actual']-cache[hs[0]]['psfc_actual'])/(secs[-1]-secs[0])*36;res=actual-calc
          result.append({'case':case,'time_h':t0,'mean_hpa_h':disk_average(mean,r,R),'eddy_hpa_h':disk_average(eddy,r,R),'thermodynamic_hpa_h':disk_average(thermo,r,R),'calculated_hpa_h':disk_average(calc,r,R),'actual_hpa_h':disk_average(actual,r,R),'residual_hpa_h':disk_average(res,r,R)})
    df=pd.DataFrame(result);df.to_csv(out/'pressure_groups_timeseries.csv',index=False)
    summary={'config':cfg,'group_definitions':{'mean':MEAN,'eddy':EDDY,'thermodynamic':SOURCE},'statistics':{}}
    for case in ('CTRL','JET'):
      q=df[df.case==case];summary['statistics'][case]={'residual_mean_hpa_h':float(q.residual_hpa_h.mean()),'residual_rmse_hpa_h':float(np.sqrt(np.mean(q.residual_hpa_h**2))),'residual_max_abs_hpa_h':float(abs(q.residual_hpa_h).max())}
    d=df[df.case=='JET'].set_index('time_h').drop(columns='case')-df[df.case=='CTRL'].set_index('time_h').drop(columns='case');summary['statistics']['JET_minus_CTRL']={'residual_mean_hpa_h':float(d.residual_hpa_h.mean()),'residual_rmse_hpa_h':float(np.sqrt(np.mean(d.residual_hpa_h**2))),'residual_max_abs_hpa_h':float(abs(d.residual_hpa_h).max())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');plot(df,out,cfg,summary);return summary
def plot(df,out,cfg,summary):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,1,figsize=(13,11),sharex=True,layout='constrained');colors={'mean':'#2166ac','eddy':'#7b3294','thermodynamic':'#d73027'}
    for ax,case in zip(axs[:2],('CTRL','JET')):
      q=df[df.case==case];t=q.time_h
      for g,c in colors.items():ax.plot(t,q[f'{g}_hpa_h'],color=c,lw=1.8,label=g.capitalize())
      ax.plot(t,q.actual_hpa_h,color='k',lw=2,label='Actual');ax.plot(t,q.calculated_hpa_h,color='#fdae61',lw=1.5,label='Group sum');ax.axhline(0,color='.4',lw=.7);ax.set_ylabel('hPa h$^{-1}$');ax.set_title('JET30' if case=='JET' else case);ax.grid(alpha=.2);ax.legend(ncol=5,fontsize=8)
    c=df[df.case=='CTRL'].set_index('time_h').drop(columns='case');j=df[df.case=='JET'].set_index('time_h').drop(columns='case');d=j-c;t=d.index
    for g,col in colors.items():axs[2].plot(t,d[f'{g}_hpa_h'],color=col,lw=1.8,label=g.capitalize())
    axs[2].plot(t,d.actual_hpa_h,color='k',lw=2,label='Actual');axs[2].plot(t,d.calculated_hpa_h,color='#fdae61',lw=1.5,label='Group sum');axs[2].plot(t,d.residual_hpa_h,color='.45',lw=1,label='Residual');axs[2].axhline(0,color='.4',lw=.7);axs[2].set(title='JET30 - CTRL',xlabel='Center time (h)',ylabel='hPa h$^{-1}$');axs[2].grid(alpha=.2);axs[2].legend(ncol=6,fontsize=8)
    fig.suptitle(f"100-km disk-mean surface-pressure tendency | +/-{cfg['half_window_h']:g} h windows")
    fig.savefig(out/'pressure_groups_timeseries_50_150h_r100.png',dpi=220);fig.savefig(out/'pressure_groups_timeseries_50_150h_r100.pdf');plt.close(fig)
    lines=['# 100-km surface-pressure tendency groups, 50-150 h','','Each center time uses a complete +/-3 h trapezoidal window. Spatial statistic is the full r<=100 km disk mean.','Mean = four mean advection/compression terms; eddy = four eddy advection/compression terms; thermodynamic = moist-divergence correction plus microphysics, diffusion, turbulence, damping and radiation.','','Residual statistics (hPa/h):','```',json.dumps(summary['statistics'],indent=2),'```','','The hourly budget fields are instantaneous output-step snapshots. Absolute closure and JET-CTRL difference closure are reported separately; process interpretation is not made when closure is poor.','Run: python scripts/plot_inner_core_pressure_groups_timeseries.py --config config/inner_core_pressure_groups_50_150h_r100.json']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',default='config/inner_core_pressure_groups_50_150h_r100.json');a=p.parse_args();cfg=json.loads((ROOT/a.config).read_text(encoding='utf-8-sig'));print(json.dumps(run(cfg),indent=2))
