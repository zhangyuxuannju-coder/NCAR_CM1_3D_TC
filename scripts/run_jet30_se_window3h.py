"""Window orchestration only: reuse existing factorial SE core unchanged."""
from pathlib import Path
import json,csv
source=Path('scripts/run_strength_phase_mixed_se_25N.py').read_text().split('for label,path in paths.items():')[0]
source=source.replace('output/strength_sensitivity_phase_mixed_se_25N','output/jet30_se_window3h_1200_plot500').replace('max_r_km=300.','max_r_km=1200.')
exec(compile(source,'existing_mixed_se_core','exec'),globals())
old=Path('output/jet30_matched_se_1200_plot500')
cache=old/'cache'
matches=json.loads((old/'matching_audit.json').read_text())
(OUT/'matching_audit.json').write_text(json.dumps(matches,indent=2))
summary=[]
for rec in matches:
    if not rec['accepted']:continue
    j,c=rec['jet_h'],rec['ctrl_h'];records=[]
    for offset in range(-3,4):
        jj,cc=j+offset,c+offset
        name=f'U30_J{jj}_C{cc}.npz';candidate=OUT/name
        if not candidate.exists():candidate=old/name
        print('WINDOW',j,c,'OFFSET',offset,flush=True)
        records.append(dict(np.load(candidate,allow_pickle=True)) if candidate.exists() else pair('U30',jj,cc))
    mean={k:np.mean([x[k] for x in records],axis=0) for k in records[0]}
    mean.update(jet_hours=np.arange(j-3,j+4),ctrl_hours=np.arange(c-3,c+4),residual_max=max(float(np.max(x['residuals'])) for x in records),closure_max=max(float(x['closure']) for x in records))
    np.savez_compressed(OUT/f"{rec['phase']}_J{j}_C{c}_window_mean.npz",**mean)
    mask=mean['r_km']<=500
    for region,zm in [('outflow_10_17km',(mean['z_km']>=10)&(mean['z_km']<=17)),('whole_0_20km',np.ones(mean['z_km'].size,bool))]:
        r=mean['r_km'][mask];z=mean['z_km'][zm]
        weight=np.gradient(z)[:,None]*r[None,:]*np.gradient(r)[None,:]
        row=dict(phase=rec['phase'],jet_h=j,ctrl_h=c,region=region,residual_max=mean['residual_max'],closure_max=mean['closure_max'])
        for k in ['operator','thermal','interaction','total']:
            u=mean[k+'_ur'][zm][:,mask];row[k+'_rms_ur']=float(np.sqrt(np.sum(weight*u*u)/np.sum(weight)))
        summary.append(row)
    for overlay in ['actual_jet_ur','actual_delta_ur']:
        fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
        limit=max([float(np.max(np.abs(mean[k+'_ur'][:,mask]))) for k in ['operator','thermal']]+[1e-6])
        for ax,k in zip(axes,['operator','thermal']):
            im=ax.contourf(mean['r_km'],mean['z_km'],mean[k+'_ur'],levels=np.linspace(-limit,limit,25),cmap='RdBu_r',extend='both')
            ax.contour(mean['r_km'],mean['z_km'],mean[overlay],levels=[-10,-5,-2,2,5,10,15],colors='k',linewidths=.65)
            ax.set(xlim=(0,500),ylim=(0,20),title=k+' change response',xlabel='Radius (km)')
        axes[0].set_ylabel('Height (km)')
        fig.suptitle(f"JET30 {j} h / CTRL {c} h | +/-3 h hourly response mean | black: {overlay}")
        fig.subplots_adjust(right=.85,wspace=.12);fig.colorbar(im,cax=fig.add_axes([.88,.16,.02,.65]),label='SE radial response (m/s)')
        fig.savefig(OUT/f"{rec['phase']}_J{j}_C{c}_{overlay}.png",dpi=180,bbox_inches='tight');plt.close(fig)
    print('DONE WINDOW',j,c,flush=True)
    with (OUT/'comparison_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
(OUT/'README.json').write_text(json.dumps(dict(cases=['standard JET30','CTRL'],solve_radius_km=1200,plot_radius_km=500,window='7 hourly offsets -3 through +3 around each previously matched central time; offsets retain original time lag, not independently rematched; phase-edge windows extend beyond phase bounds',averaging='Solve each hourly pair first, then average responses and actual winds; no phase-wide average',thermal='LC inverse applied to FQ_J minus FQ_C, own-state full thermal RHS; includes existing Q sources, not diabatic heating alone',operator='LJ inverse FC minus LC inverse FC, full CTRL RHS fixed',solver='Existing Bui core and nonuniform solver unchanged; homogeneous Dirichlet',f=args.f,eps_ratio=args.eps_ratio,metrics='Cylindrical-volume weighted RMS within 500 km, outflow 10-17 km and whole domain; diagnostic amplitude only, not causal intensity contribution',colour_scale='Common symmetric scale for operator and thermal within each window',evidence='Regularized balanced projection; inspect interaction and regularization dependence before mechanism claims'),indent=2))
print('COMPLETE',flush=True)
