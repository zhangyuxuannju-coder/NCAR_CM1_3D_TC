#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('output/jet_diabatic_heating_position_25N_ctrl_vs_jet30_9deg')
pos=pd.read_csv(out/'products/heating_position_metrics.csv')
sha=pd.read_csv(out/'products/heating_position_shapley_metrics.csv')
matched=pos[pos.comparison.str.startswith('matched_')]
rows=[]
for name,g in matched.groupby('comparison',sort=False):
    c=g[g.case=='CTRL'].iloc[0]; j=g[g.case=='JET'].iloc[0]
    rows.append(dict(pair=name.replace('matched_',''),dr=j.centroid_r_km-c.centroid_r_km,
                     dz=j.centroid_z_km-c.centroid_z_km,
                     dinside=j.inside_rmw_fraction-c.inside_rmw_fraction))
d=pd.DataFrame(rows)
regions=['bl_inflow','inner_updraft','upper_outflow']; effects=['amplitude','location','shape']
labels={'bl_inflow':'BL inflow','inner_updraft':'Eyewall updraft','upper_outflow':'Upper outflow'}
colors={'amplitude':'#4C78A8','location':'#F58518','shape':'#54A24B'}
fig,ax=plt.subplots(2,2,figsize=(11.2,7.6),constrained_layout=True)
x=np.arange(len(d));
ax[0,0].bar(x,d.dr,color=np.where(d.dr<0,'#3B82F6','#EF4444'));ax[0,0].axhline(0,color='k',lw=.8);ax[0,0].set(ylabel='JET - CTRL radial centroid (km)',title='(a) Matched heating displacement');ax[0,0].set_xticks(x,d.pair,rotation=25)
ax[0,1].bar(x,d.dz,color=np.where(d.dz>0,'#8B5CF6','#9CA3AF'));ax[0,1].axhline(0,color='k',lw=.8);ax[0,1].set(ylabel='JET - CTRL vertical centroid (km)',title='(b) Matched heating displacement');ax[0,1].set_xticks(x,d.pair,rotation=25)
for jj,reg in enumerate(regions):
    q=sha[(sha.region==reg)&(sha.effect.isin(effects))].groupby('effect').mean(numeric_only=True)
    bottom=0
    for eff in effects:
        v=q.loc[eff,'share_on_total'];ax[1,0].bar(jj,v,bottom=bottom,color=colors[eff],label=eff if jj==0 else None);bottom+=v
    for k,eff in enumerate(effects):
        vals=sha[(sha.region==reg)&(sha.effect==eff)].projection_on_observed
        ax[1,1].bar(jj+(k-1)*.22,vals.mean(),width=.2,color=colors[eff],label=eff if jj==0 else None)
        ax[1,1].errorbar(jj+(k-1)*.22,vals.mean(),yerr=vals.std(ddof=1)/np.sqrt(len(vals)),color='k',capsize=2,lw=.8)
ax[1,0].axhline(0,color='k',lw=.8);ax[1,0].set_xticks(range(3),[labels[x] for x in regions]);ax[1,0].set(ylabel='Share of direct-Q SE response',title='(c) Amplitude-location-shape Shapley attribution');ax[1,0].legend(frameon=False,ncol=3,fontsize=9)
ax[1,1].axhline(0,color='k',lw=.8);ax[1,1].set_xticks(range(3),[labels[x] for x in regions]);ax[1,1].set(ylabel='Projection on observed JET - CTRL circulation',title='(d) Contribution to simulated circulation difference');ax[1,1].legend(frameon=False,ncol=3,fontsize=9)
fig.suptitle('JET-induced diabatic-heating displacement and balanced secondary-circulation response',fontsize=14)
for p in [out/'figures/summary_heating_position_and_response.png',out/'figures/summary_heating_position_and_response.pdf']:
    fig.savefig(p,dpi=260 if p.suffix=='.png' else None,bbox_inches='tight')
d.to_csv(out/'products/matched_heating_displacement.csv',index=False)
print(d.to_string(index=False));print('max_closure',sha.closure.abs().max())
