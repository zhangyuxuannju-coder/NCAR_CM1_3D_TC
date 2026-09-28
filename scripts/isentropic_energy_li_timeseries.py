#!/usr/bin/env python3
"""Fixed-level 60--150 h MAFALDA energetics from cached hourly conditional bins."""
from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import isentropic_energy_li_validation as ev
base=ev.base

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 cfg=json.loads((ROOT/'config/isentropic_energy_li_fig7_8.json').read_text());out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
 start=float(cfg['timeseries_start_h']);end=float(cfg['timeseries_end_h']);half=float(cfg['validation_half_window_h']);level=-1.3e9
 readcfg=dict(cfg);readcfg['validation_center_h']=(start+end)/2;readcfg['validation_half_window_h']=(end-start)/2+half
 hourly={};checks=[]
 for label,path in ev.CASES.items():
  cache=out/(label+'_hourly_energy_bins.npz')
  if cache.exists():
   with np.load(cache) as q:loaded={k:q[k] for k in q.files}
   t=loaded.pop('time_h');loaded.pop('alpha',None);z=loaded.pop('z_m');edges=loaded.pop('theta_edges_K')
   if not (np.isclose(t[0],start-half) and np.isclose(t[-1],end+half)):raise ValueError('cached hourly range mismatch')
   stack=loaded;qc=[]
  else:stack,t,a,z,edges,qc=ev.read_case(label,path,out,readcfg)
  hourly[label]=(stack,t,z,edges);checks+=qc
 rows=[];invalid=[]
 for center in np.arange(start,end+1e-6,float(cfg['timeseries_step_h'])):
  for label,(h,t,z,edges) in hourly.items():
   ii=np.flatnonzero((t>=center-half-1e-6)&(t<=center+half+1e-6));row=dict(case=label,center_h=float(center),window_start_h=float(center-half),window_end_h=float(center+half),cycle_level_kg_s=level,valid=False)
   if len(ii)!=7 or not np.allclose(t[ii],np.arange(center-half,center+half+1)):
    row['reason']='incomplete seven-hour window';rows.append(row);invalid.append(row.copy());continue
   a=np.ones(7)/7;d={k:np.sum(v[ii]*a.reshape((7,)+(1,)*(v.ndim-1)),axis=0) for k,v in h.items()}
   d['Psi']=np.c_[np.zeros(len(z)),np.cumsum(d['Fz']*np.diff(edges),axis=1)];d['support']=(d['count']>=1)&(d['M']>0)
   for name in ev.VARS:d[name]=np.divide(d[name+'_num'],d['M']*np.diff(edges)[None,:],out=np.full_like(d[name+'_num'],np.nan),where=d['M']>0)
   c,cands=base.cycle(edges,z,d['Psi'],np.ones_like(d['support'],bool),level,cfg['near_surface_m'])
   if not c or not c['abc_in_flow_order'] or c['vertices'][:,1].min()<z[0]+1 or c['vertices'][:,1].max()>=z[-1]:
    row['reason']='no valid fixed-level deep closed geometry';rows.append(row);invalid.append(row.copy());continue
   path=c['vertices'];tc=.5*(edges[:-1]+edges[1:]);st={n:ev.interp_strict(z,tc,d[n],path) for n in ev.VARS}
   if any(np.any(~np.isfinite(v)) for v in st.values()):
    row['reason']='missing conditional state along contour';rows.append(row);invalid.append(row.copy());continue
   e=ev.integrate(path,st,c['points']);
   # Quasi-steady diagnostic: domain L1 norms in theta-z space, both kg/s.
   mt=np.gradient(h['M'],t*3600,axis=0)[ii].mean(axis=0);div=np.gradient(d['Fz'],z,axis=0)
   dz=np.gradient(z)[:,None];dt=np.diff(edges)[None,:]
   e['mass_tendency_L1_kg_s']=float(np.sum(abs(mt)*dz*dt));e['flux_divergence_L1_kg_s']=float(np.sum(abs(div)*dz*dt))
   e.update(valid=True,reason='',cycle_top_km=float(c['vertices'][:,1].max()/1000),cycle_bottom_m=float(c['vertices'][:,1].min()),path_points=len(path),mass_error=float(d['mass_error']),flux_error=float(d['flux_error']),Psi_endpoint_max_kg_s=float(abs(d['Psi'][:,-1]).max()))
   row.update(e);rows.append(row)
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with (out/'energy_timeseries.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 dump(out/'invalid_windows.json',invalid)
 # Fig. 7 style; gaps are invalid windows and are not interpolated.
 colors={'CTRL':'#222222','JET30':'#0072B2'};fig,axs=plt.subplots(3,2,figsize=(12,13),layout='constrained');panels=[('Wmax',r'$W_{max}$'),('WKE',r'$W_{KE}$'),('WP',r'$W_P$'),('GP','GP'),('R','Residual R')]
 for ax,(key,title) in zip(axs.flat[:5],panels):
  for label in colors:
   q=[r for r in rows if r['case']==label];ax.plot([r['center_h'] for r in q],[r.get(key,np.nan) if r['valid'] else np.nan for r in q],color=colors[label],label=label)
  ax.set(title=title,ylabel='J kg$^{-1}$ dry air',xlim=(start,end));ax.grid(alpha=.2)
 fig7_values=[float(r[k]) for r in rows if r['valid'] for k,_ in panels];fig7_lo=min(fig7_values);fig7_hi=max(fig7_values);fig7_pad=.05*max(fig7_hi-fig7_lo,1.);fig7_ylim=(fig7_lo-fig7_pad,fig7_hi+fig7_pad)
 for a0 in axs.flat[:5]:a0.set_ylim(fig7_ylim)
 axs.flat[0].legend()
 ax=axs.flat[5]
 for label in colors:
  q=[r for r in rows if r['case']==label];x=[r['center_h'] for r in q];ax.plot(x,[r.get('mass_tendency_L1_kg_s',np.nan) if r['valid'] else np.nan for r in q],color=colors[label],ls='-',label=label+' tendency');ax.plot(x,[r.get('flux_divergence_L1_kg_s',np.nan) if r['valid'] else np.nan for r in q],color=colors[label],ls='--',label=label+' flux divergence')
 ax.set(title='Quasi-steady L1 check',ylabel='kg s$^{-1}$',xlabel='Window center (h)',xlim=(start,end));ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
 fig.suptitle('Single-member 7-h means; R=1000 km; dtheta=1 K; fixed Psi=-1.3e9 kg/s',fontsize=11)
 fig.savefig(out/'li_fig7_energy_timeseries.png',dpi=200);fig.savefig(out/'li_fig7_energy_timeseries.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 fig8_keys=['W_ca','W_ab_bc'];fig8_values=[float(r[k]) for r in rows if r['valid'] for k in fig8_keys];fig8_lo=min(fig8_values);fig8_hi=max(fig8_values);fig8_pad=.05*max(fig8_hi-fig8_lo,1.);fig8_ylim=(fig8_lo-fig8_pad,fig8_hi+fig8_pad)
 for ax,key,title in zip(axs,fig8_keys,['Return branch c→a','Inflow + ascent a→b→c']):
  for label in colors:
   q=[r for r in rows if r['case']==label];ax.plot([r['center_h'] for r in q],[r.get(key,np.nan) if r['valid'] else np.nan for r in q],color=colors[label],label=label)
  ax.axhline(0,color='.6',lw=.7);ax.set(title=title,xlabel='Window center (h)',ylabel='friction-force work (J kg$^{-1}$ dry air)',xlim=(start,end),ylim=fig8_ylim);ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Positive: friction force does positive work under Eq. (10) sign convention; fixed cycle and 7-h means',fontsize=10)
 fig.savefig(out/'li_fig8_leg_dissipation.png',dpi=200);fig.savefig(out/'li_fig8_leg_dissipation.pdf');plt.close(fig)
 valid=sum(r['valid'] for r in rows);summary=dict(center_range_h=[start,end],data_read_range_h=[start-half,end+half],total_case_windows=len(rows),valid_case_windows=valid,invalid_case_windows=len(rows)-valid,cycle_level_kg_s=level,quasi_steady_definition='L1 integral over theta-z of abs(dM/dt) and abs(dFz/dz); units kg/s',single_member_time_average=True)
 dump(out/'timeseries_summary.json',summary);dump(out/'timeseries_provenance.json',dict(script=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reused='isentropic_energy_li_validation hourly conditional bins; no missing-state interpolation'))
 print(json.dumps(summary,indent=2));print('Completed',out)
if __name__=='__main__':main()
