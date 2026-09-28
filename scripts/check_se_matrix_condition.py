"""Read-only matrix audit; row-scaled estimated 1-norm condition lower bounds."""
import sys,json
from pathlib import Path
source=Path('scripts/run_complete_se_attribution.py').read_text();prefix=source[:source.index("names=['thermal'")].replace('OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)','OUT=None')
exec(compile(prefix,'existing_matrix_helpers','exec'),globals())
from scipy.sparse import diags
rows=[]
for h in [48,60,80,110,130]:
    for label in ['CTRL','U30']:
        s=state(label,h);r,z=s['r_km']*1000,s['z_km']*1000
        theta,_=invert_balanced_theta(s['ut'],s['theta'],r,z,args.f,outer_smooth_window=1)
        basic=build_basic_state(s['ut'],theta,s['rho'],r,z,args.f);_,op,reg=old.build_ctrl_operator(basic,r,z,args)
        m=matrix(op,r,z,s['rho']);scaled=(diags(1/np.abs(m.diagonal()))@m).tocsc();lu=spla.splu(scaled)
        inv=spla.LinearOperator(scaled.shape,matvec=lambda x:lu.solve(x),rmatvec=lambda x:lu.solve(x,trans='T'),matmat=lambda x:lu.solve(x),rmatmat=lambda x:lu.solve(x,trans='T'))
        estimate=float(spla.norm(scaled,1)*spla.onenormest(inv))
        row=dict(hour=h,case=label,row_scaled_condition_lower_bound=estimate,raw_nonelliptic=reg['raw_nonelliptic_fraction'],changed_fraction=reg['changed_coefficient_fraction'],min_regularized_discriminant=reg['min_regularized_discriminant'])
        rows.append(row);print(json.dumps(row),flush=True)
path=Path('output/se_boundary_fix_validation_new01/matrix_condition_audit.json')
assert not path.exists();path.write_text(json.dumps(dict(method='Estimated lower bound of row-scaled matrix 1-norm condition number; not physical causality',rows=rows),indent=2))
