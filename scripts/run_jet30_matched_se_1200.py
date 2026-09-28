"""JET30 versus CTRL: individual matched solves, 1200-km domain, 500-km plots."""
from pathlib import Path
import json
source=Path('scripts/run_strength_phase_mixed_se_25N.py').read_text()
source=source.split('for label,path in paths.items():')[0]
source=source.replace("output/strength_sensitivity_phase_mixed_se_25N","output/jet30_matched_se_1200_plot500")
source=source.replace('max_r_km=300.','max_r_km=1200.')
exec(compile(source,'reused_mixed_se_core','exec'),globals())
previous=Path('output/strength_sensitivity_phase_mixed_se_25N/matching_audit.json')
matches=[x for x in json.loads(previous.read_text()) if x['case']=='U30']
(OUT/'matching_audit.json').write_text(json.dumps(matches,indent=2))
for rec in matches:
    if not rec['accepted']:continue
    j,c=rec['jet_h'],rec['ctrl_h']
    print('START PAIR',rec,flush=True)
    result=pair('U30',j,c)
    for overlay in ['actual_jet_ur','actual_delta_ur']:
        fig,axes=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
        inside=result['r_km']<=500
        for ax,k in zip(axes,['operator','thermal','momentum']):
            field=result[k+'_ur'];limit=max(float(np.max(np.abs(field[:,inside]))),1e-6)
            im=ax.contourf(result['r_km'],result['z_km'],field,levels=np.linspace(-limit,limit,25),cmap='RdBu_r',extend='both')
            ax.contour(result['r_km'],result['z_km'],result[overlay],levels=[-10,-5,-2,2,5,10,15],colors='k',linewidths=.65)
            ax.set(xlim=(0,500),ylim=(0,20),title=k,xlabel='Radius (km)')
            fig.colorbar(im,ax=ax,label='SE radial response (m/s)',shrink=.8)
        axes[0].set_ylabel('Height (km)')
        fig.suptitle(f"{rec['phase']} h | JET30 {j} h / CTRL {c} h | black: {overlay}")
        fig.tight_layout();fig.savefig(OUT/f"{rec['phase']}_J{j}_C{c}_{overlay}.png",dpi=180);plt.close(fig)
    print('DONE PAIR',j,c,flush=True)
(OUT/'README.json').write_text(json.dumps(dict(solve_radius_km=1200,plot_radius_km=500,time_average=False,cases=['standard JET30','CTRL'],matches='Reused accepted U30 same-phase intensity/RMW matches',f=args.f,eps_ratio=args.eps_ratio,solver='Existing nonuniform flux-form, homogeneous Dirichlet boundaries',RHS='Own-state thermal and momentum RHS built once; literal fixed RHS factorial',velocity='Common CTRL density for decomposition',colour_scale='Each component uses its own symmetric scale within plotted 500 km; read colourbars before comparing amplitudes',evidence='Regularized balanced projection, not full model circulation'),indent=2))
print('COMPLETE',flush=True)
