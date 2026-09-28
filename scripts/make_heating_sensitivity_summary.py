#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('output/jet_diabatic_heating_position_25N_ctrl_vs_jet30_9deg')
pos=pd.read_csv(out/'products/heating_position_metrics.csv')
a=pd.read_csv(out/'products/heating_position_shapley_metrics_dr12.csv');b=pd.read_csv(out/'products/heating_position_shapley_metrics_native.csv')
m=pos[pos.comparison.str.startswith('matched_')];rows=[]
for n,g in m.groupby('comparison',sort=False):
 c=g[g.case=='CTRL'].iloc[0];j=g[g.case=='JET'].iloc[0];rows.append([n.replace('matched_',''),j.centroid_r_km-c.centroid_r_km,j.centroid_z_km-c.centroid_z_km])
d=pd.DataFrame(rows,columns=['pair','dr','dz']);x=np.arange(len(d));fig,ax=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
ax[0,0].bar(x,d.dr,color='#3B82F6');ax[0,0].axhline(0,color='k',lw=.8);ax[0,0].set(ylabel='JET - CTRL radial centroid (km)',title='(a) Strength/RMW-matched displacement');ax[0,0].set_xticks(x,d.pair,rotation=25)
ax[0,1].bar(x,d.dz,color='#8B5CF6');ax[0,1].axhline(0,color='k',lw=.8);ax[0,1].set(ylabel='JET - CTRL vertical centroid (km)',title='(b) Strength/RMW-matched displacement');ax[0,1].set_xticks(x,d.pair,rotation=25)
effects=['amplitude','location','shape'];xx=np.arange(3)
for df,label,marker in [(a,'12.000-km bins','o'),(b,'11.985-km bins','s')]:
 q=df[(df.region=='inner_updraft')&df.effect.isin(effects)].groupby('effect').share_on_total.mean()
 ax[1,0].plot(xx,[q[e] for e in effects],marker=marker,lw=1.5,label=label)
 q=df[(df.region=='inner_updraft')&(df.effect=='total')]
 ax[1,1].plot(x,q.projection_on_observed.to_numpy(),marker=marker,lw=1.5,label=label)
ax[1,0].axhline(0,color='k',lw=.8);ax[1,0].set_xticks(xx,effects);ax[1,0].set(ylabel='Share of direct-Q SE updraft response',title='(c) Position/shape attribution is bin-sensitive');ax[1,0].legend(frameon=False)
ax[1,1].axhline(0,color='k',lw=.8);ax[1,1].set_xticks(x,d.pair,rotation=25);ax[1,1].set(ylabel='Projection on observed updraft difference',title='(d) Total direct-Q projection is more stable');ax[1,1].legend(frameon=False)
fig.suptitle('JET-induced heating displacement: robust signals and resolution limit',fontsize=14)
for suf in ['png','pdf']:fig.savefig(out/f'figures/summary_heating_position_resolution_sensitivity.{suf}',dpi=260 if suf=='png' else None,bbox_inches='tight')
