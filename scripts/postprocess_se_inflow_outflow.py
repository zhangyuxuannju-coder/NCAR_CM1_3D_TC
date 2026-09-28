"""Postprocess existing SE arrays only; no model/core/solver changes."""
import sys,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('output/same_time_se_window3h_50_150_new01')
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
rows=[]
for h in range(50,151,10):
    a=dict(np.load(ROOT/f'window_{h:03d}.npz'))
    r,z=a['r_km'],a['z_km'];rr,zz=np.meshgrid(r,z)
    weight=np.gradient(z)[:,None]*r[None,:]*np.gradient(r)[None,:]
    regions={'BL':(rr>=18)&(rr<=300)&(zz>=.5)&(zz<=2),'OUT':(rr>=200)&(rr<=400)&(zz>=10)&(zz<=17)}
    row=dict(hour=h,ctrl_reg=float(a['ctrl_changed_coefficient_fraction']),jet_reg=float(a['jet_changed_coefficient_fraction']))
    def mean(f,m):return float(np.sum(f[m]*weight[m])/np.sum(weight[m]))
    for region,m in regions.items():
        sign=-1 if region=='BL' else 1
        row[region+'_observed_delta']=sign*mean(a['delta_ur'],m)
        for case in ['ctrl','jet']:row[region+'_'+case]=sign*mean(a[case+'_ur'],m)
        for key in ['inertial','thermal']:
            u=a[key+'_u'];d=a['delta_ur']
            row[region+'_'+key+'_signed']=sign*mean(u,m)
            row[region+'_'+key+'_rms']=float(np.sqrt(mean(u*u,m)))
            row[region+'_'+key+'_projection']=float(np.sum(weight[m]*u[m]*d[m])/max(np.sum(weight[m]*d[m]**2),1e-30))
    out=regions['OUT']
    for case in ['ctrl','jet']:
        p=np.maximum(a[case+'_ur'],0)*weight*out;total=np.sum(p)
        for name,coord in [('r',rr),('z',zz)]:
            centre=float(np.sum(p*coord)/total);row[case+'_out_'+name]=centre
            row[case+'_out_width_'+name]=float(np.sqrt(np.sum(p*(coord-centre)**2)/total))
        row[case+'_out_max']=float(np.max(a[case+'_ur'][out]))
    positive=out&(a['ctrl_ur']>0);den=np.sum(weight[positive]*a['ctrl_ur'][positive])
    for key in ['inertial','thermal']:
        u=a[key+'_u']
        for name,coord in [('r',rr),('z',zz)]:
            row[key+'_linear_shift_'+name]=float(np.sum(weight[positive]*(coord[positive]-row['ctrl_out_'+name])*u[positive])/den)
        row[key+'_peak500']=float(np.max(abs(u[(rr<=500)&(zz<=18)])))
    row['thermal_amplitude_QA_flag']=row['thermal_peak500']>50
    rows.append(row)
with (OUT/'metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
hours=[x['hour'] for x in rows]
def vals(key,qa=False):return [np.nan if qa and x['thermal_amplitude_QA_flag'] else x[key] for x in rows]
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,constrained_layout=True)
for col,region in enumerate(['BL','OUT']):
    axes[0,col].plot(hours,vals(region+'_ctrl'),'k-o',label='Actual CTRL')
    axes[0,col].plot(hours,vals(region+'_jet'),'r-o',label='Actual JET30')
    axes[0,col].set_title('BL 18-300 km, 0.5-2 km' if region=='BL' else 'Outflow 200-400 km, 10-17 km')
    axes[0,col].set_ylabel('Signed mean inward speed (m/s)' if region=='BL' else 'Signed mean outward speed (m/s)')
    axes[1,col].plot(hours,vals(region+'_observed_delta'),'k-o',label='Observed JET - CTRL')
    axes[1,col].plot(hours,vals(region+'_inertial_signed'),'b-o',label='Inertial single-term SE')
    axes[1,col].plot(hours,vals(region+'_thermal_signed',True),'r-o',label='Thermal-difference SE')
    axes[1,col].axhline(0,color='.5',lw=.6)
    axes[1,col].set_ylabel('Inward contribution (+)' if region=='BL' else 'Outward contribution (+)')
for ax in axes.ravel():ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('Time (h)')
fig.suptitle('Existing same-time +/-3h SE responses; thermal 50/60h QA flagged and omitted from curves, retained in CSV')
fig.savefig(OUT/'inflow_outflow_signed_contributions.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,constrained_layout=True)
for col,name in enumerate(['z','r']):
    for case,colour in [('ctrl','k'),('jet','r')]:axes[0,col].plot(hours,vals(case+'_out_'+name),colour+'-o',label='Actual '+case.upper())
    axes[0,col].set_ylabel('Positive-ur weighted centre (km)');axes[0,col].set_title('Outflow height' if name=='z' else 'Outflow radius')
    axes[1,col].plot(hours,np.array(vals('jet_out_'+name))-np.array(vals('ctrl_out_'+name)),'k-o',label='Observed centre difference')
    for key,colour in [('inertial','b'),('thermal','r')]:axes[1,col].plot(hours,vals(key+'_linear_shift_'+name,key=='thermal'),colour+'-o',label=key+' SE linear centre shift')
    axes[1,col].axhline(0,color='.5',lw=.6);axes[1,col].set_ylabel('Shift (km); positive up/out')
for ax in axes.ravel():ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('Time (h)')
fig.suptitle('200-400 km, 10-17 km only; linear shifts on fixed positive CTRL mask, not full nonlinear position changes')
fig.savefig(OUT/'outflow_geometry_actual_and_linear.png',dpi=180);plt.close(fig)
(OUT/'README.json').write_text(json.dumps(dict(source=str(ROOT),new_SE_solves=0,BL='18-300 km, 0.5-2 km; minus cylindrical-volume-weighted signed ur mean, positive contribution strengthens inward motion; includes outward patches',OUT='200-400 km, 10-17 km; signed ur mean and positive-ur-weighted geometry',projection='Weighted dot(SE response, observed ur difference) / weighted observed difference squared; pattern projection, not physical percentage or closure',linear_geometry='delta c = integral((coordinate-cCTRL)*delta ur*dV)/integral(urCTRL*dV), fixed positive CTRL mask; meaningful as local linear direction, not finite attribution',QA='Thermal peak absolute ur above 50 m/s is an amplitude screening flag, not a universal physical threshold; raw values retained CSV and flagged points omitted only from plots',limitations='Mean-layer-spacing existing SE solver, no residual report; inertial-only K3 forcing spatially smoothed and masked below100km, thermal unmasked; balanced projection cannot represent full frictional boundary layer or unique causal intensity mechanism'),indent=2))
for row in rows:print(json.dumps(row),flush=True)
print('COMPLETE',flush=True)
