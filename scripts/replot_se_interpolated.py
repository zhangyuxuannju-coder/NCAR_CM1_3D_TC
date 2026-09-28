"""Reuse existing plotting blocks; interpolate only at display, never re-solve."""
import sys,json,textwrap
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scripts import plot_inertial_operator_evolution_matched as inertial
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
ROOT=Path('output/same_time_se_window3h_50_150_new01')
source=Path('scripts/run_same_time_se_window3h.py').read_text()
function=source[source.index('def panel('):source.index('completed=[]')]
insertion="""    native_r,native_z=r,z
    r=np.arange(native_r[0],500.001,2.0); z=np.arange(native_z[0],18.001,.1)
    rr,zz=np.meshgrid(r,z); points=np.column_stack([zz.ravel(),rr.ravel()])
    def interpolate(f):
        return RegularGridInterpolator((native_z,native_r),f,method='linear',bounds_error=True)(points).reshape(zz.shape)
    field=interpolate(field)
    local=dict(m); local[wind]=interpolate(m[wind])
    if flow:local[key[:-1]+'w']=interpolate(m[key[:-1]+'w'])
    m=local
"""
function=function.replace("    im=ax.pcolormesh(r,z,field,cmap=inertial.CMAP,shading='auto',vmin=-limit,vmax=limit)",insertion+"    im=ax.contourf(r,z,field,cmap=inertial.CMAP,levels=np.linspace(-limit,limit,31),extend='both')")
function=function.replace('[::4]','[::20]').replace('[::3]','[::18]')
exec(compile(function,'existing_panel_display_interpolation','exec'),globals())
block=source[source.index('    fig,axes=plt.subplots(2,4'):source.index('    completed.append(h)')]
block=textwrap.dedent(block)
for h in range(50,151,10):
    mean=dict(np.load(ROOT/f'window_{h:03d}.npz'))
    exec(compile(block,'reused_existing_plot_layout','exec'),globals())
    print('PLOTTED',h,flush=True)
(OUT/'README.json').write_text(json.dumps(dict(source=str(ROOT),reused_plotting='run_same_time_se_window3h.py panel and layouts',interpolation='Linear RegularGridInterpolator on actual stored r/z, no smoothing or extrapolation',display_dr_km=2,display_dz_km=.1,display_domain='r 6–500 km, z 0.025–17.925 km; axes unchanged 0–500,0–18',native_dr_km=12,native_shape=[49,100],no_new_SE_solves=True,original_colour_limits_preserved=True,arrow_spacing='approximately36 km radial,2 km vertical',warning='Display interpolation does not improve physical resolution, solver discretization, or abnormal amplitude reliability; linear display across100km forcing cutoff is not a change to the stored RHS'),indent=2))
print('COMPLETE',flush=True)
