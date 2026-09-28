"""Change response colour limits only, reuse prior interpolation and plotting."""
import sys
from pathlib import Path
source=Path('scripts/replot_se_interpolated.py').read_text()
old="limit=max([float(np.max(abs(mean[k+'_u'][mask]))) for k in ['thermal','momentum','inertial']]+[1e-6])"
new="limit=quantile_limit(mean,mask)"
assert old in Path('scripts/run_same_time_se_window3h.py').read_text()
adjustment="block=textwrap.dedent(block)\nblock=block[block.index('for wind in '):]\nblock=block.replace("+repr(old)+","+repr(new)+")"
source=source.replace('block=textwrap.dedent(block)',adjustment)
helper="""quantile_records=[]
def quantile_limit(mean,mask):
    values=mean['delta_ur'][mask];values=values[np.isfinite(values)]
    p5,p95=np.percentile(values,[5,95]);limit=max(abs(float(p5)),abs(float(p95)),1e-6)
    if not any(q['hour']==h for q in quantile_records):
        quantile_records.append(dict(hour=h,p5=float(p5),p95=float(p95),vmin=-limit,vmax=limit))
    return limit
"""
source=source.replace('for h in range(50,151,10):',helper+'for h in range(50,151,10):')
source=source.replace('original_colour_limits_preserved=True','original_colour_limits_preserved=False,response_limits="Per-time symmetric max(abs(P5),abs(P95)) of native actual JET-CTRL ur in view",forcing_colour_limits_preserved=True')
namespace={'__name__':'existing_replot_with_response_scale_adjustment'}
exec(compile(source,'reused_interpolated_replot_response_scale_only','exec'),namespace)
(namespace['OUT']/'response_colour_bounds.json').write_text(namespace['json'].dumps(namespace['quantile_records'],indent=2))
print(namespace['json'].dumps(namespace['quantile_records'],indent=2),flush=True)
