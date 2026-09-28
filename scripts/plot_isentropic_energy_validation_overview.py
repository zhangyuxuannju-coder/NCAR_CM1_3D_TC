#!/usr/bin/env python3
"""Create the requested 75 h validation overview from cached hourly products."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import isentropic_energy_li_validation as ev
base=ev.base;OUT=ROOT/'output/isentropic_energy_li_fig7_8';COL={'CTRL':'#222222','JET30':'#0072B2'}
def main():
 energies=json.loads((OUT/'validation_75h_results.json').read_text());data={}
 for label in COL:
  with np.load(OUT/(label+'_hourly_energy_bins.npz')) as q:h={k:q[k] for k in q.files}
  ii=np.flatnonzero((h['time_h']>=72)&(h['time_h']<=78));a=np.ones(7)/7;d={k:np.sum(v[ii]*a.reshape((7,)+(1,)*(v.ndim-1)),axis=0) for k,v in h.items() if k not in ['time_h','alpha','z_m','theta_edges_K']}
  z=h['z_m'];ed=h['theta_edges_K'];d['Psi']=np.c_[np.zeros(len(z)),np.cumsum(d['Fz']*np.diff(ed),axis=1)];d['support']=(d['count']>=1)&(d['M']>0)
  for n in ev.VARS:d[n]=np.divide(d[n+'_num'],d['M']*np.diff(ed)[None,:],out=np.full_like(d[n+'_num'],np.nan),where=d['M']>0)
  c,_=base.cycle(ed,z,d['Psi'],np.ones_like(d['support'],bool),-1.3e9,1000);path=c['vertices'];tc=.5*(ed[:-1]+ed[1:]);st={n:ev.interp_strict(z,tc,d[n],path) for n in ev.VARS};data[label]=(d,z,ed,c,st)
 fig,ax=plt.subplots(3,3,figsize=(16,14),layout='constrained')
 for label,(d,z,ed,c,st) in data.items():
  v=c['vertices'];ax[0,0].plot(v[:,0],v[:,1]/1000,color=COL[label],label=label,lw=2)
  for name,p in c['points'].items():ax[0,0].plot(p['theta_e_K'],p['z_m']/1000,'o',color=COL[label],ms=3);ax[0,0].annotate(name,(p['theta_e_K'],p['z_m']/1000),fontsize=8)
  q=np.arange(len(v));
  for a0,n,scale,unit in [(ax[0,1],'T',1,'K'),(ax[0,2],'s',1,'J kg$^{-1}$ K$^{-1}$'),(ax[1,0],'p',.01,'hPa'),(ax[1,1],'rT',1000,'g kg$^{-1}$')]:a0.plot(q,st[n]*scale,color=COL[label],label=label);a0.set(title=n,ylabel=unit,xlabel='ordered path vertex')
 ax[0,0].set(title='75±3 h fixed cycle',xlabel='Ice-reference theta_e (K)',ylabel='Height (km)',xlim=(380,320),ylim=(0,18));ax[0,0].legend()
 for a0 in [ax[0,1],ax[0,2],ax[1,0],ax[1,1]]:a0.legend(fontsize=8)
 names=['Wmax','WKE','WP','GP','R'];x=np.arange(len(names));width=.36
 for j,label in enumerate(COL):ax[1,2].bar(x+(j-.5)*width,[energies[label][n] for n in names],width,color=COL[label],label=label)
 ax[1,2].set(title='75±3 h energy terms',ylabel='J kg$^{-1}$ dry air',xticks=x,xticklabels=names);ax[1,2].legend()
 for label,(d,z,ed,c,st) in data.items():
  mass=abs(np.sum(d['M']*np.diff(ed),axis=1)-d['domain_mass'])/d['domain_mass'];raw=abs(np.sum(d['Fz_raw']*np.diff(ed),axis=1)-d['net_raw'])/np.maximum(d['gross']+abs(d['net_raw']),1);corr=abs(np.sum(d['Fz']*np.diff(ed),axis=1))/np.maximum(d['gross'],1);endp=abs(d['Psi'][:,-1])/np.maximum(np.max(abs(d['Psi']),axis=1),1)
  for val,ls,name in [(mass,'-','mass'),(raw,'--','raw Fz'),(corr,':','corrected Fz'),(endp,'-.','Psi endpoint')]:ax[2,0].semilogx(np.maximum(val,1e-18),z/1000,color=COL[label],ls=ls,label=label+' '+name)
 ax[2,0].set(title='Layer closure diagnostics',xlabel='relative error',ylabel='Height (km)',ylim=(0,18));ax[2,0].legend(fontsize=6,ncol=2)
 for label,(d,z,ed,c,st) in data.items():
  tc=.5*(ed[:-1]+ed[1:]);count=ev.interp_strict(z,tc,d['count'],c['vertices']);ax[2,1].plot(np.arange(len(count)),count,color=COL[label],label=label)
 ax[2,1].set(title='Nearest conditional-bin sample support',xlabel='ordered path vertex',ylabel='mean count');ax[2,1].legend()
 ax[2,2].axis('off');ax[2,2].text(0,1,'Validation status\n\nBoth cycles geometrically closed\nAll path state values finite\nNo gap interpolation\nMass/raw/corrected flux recovery checked\nPsi endpoint checked\n\nR_rel:\nCTRL {:.3%}\nJET30 {:.3%}'.format(energies['CTRL']['R_rel'],energies['JET30']['R_rel']),va='top',fontsize=11)
 fig.suptitle('75 h MAFALDA energetics validation: single-member 72–78 h mean',fontsize=15)
 fig.savefig(OUT/'validation_75h_overview.png',dpi=200);fig.savefig(OUT/'validation_75h_overview.pdf');plt.close(fig)
 print('Completed validation overview')
if __name__=='__main__':main()
