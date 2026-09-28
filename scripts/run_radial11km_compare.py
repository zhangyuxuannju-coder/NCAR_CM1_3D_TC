"""Reuse previous radial animation; height interpolation, circular view, case stacking."""
import sys,json,subprocess
from pathlib import Path
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False)
source=Path('/tmp/animate_horizontal_radial.py').read_text()
source=source.replace('iz=int(np.argmin(abs(z-a.height_km)))','iz=max(0,min(int(np.searchsorted(z,a.height_km))-1,len(z)-2)); iz1=iz+1; weight=(a.height_km-z[iz])/(z[iz1]-z[iz]); original_heights=[float(z[iz]),float(z[iz1])]; z=z.copy(); z[iz]=a.height_km')
for variable in ['u','v']:
    old=f'np.asarray(ds["{variable}"][it,iz],float)'
    new=f'((1-weight)*np.asarray(ds["{variable}"][it,iz],float)+weight*np.asarray(ds["{variable}"][it,iz1],float))'
    source=source.replace(old,new)
source=source.replace('ux=np.nanmean(sp,axis=1)','sp=np.where(np.hypot(xx,yy)<=500,sp,np.nan); ux=np.nanmean(sp,axis=1)')
source=source.replace('fixed_colorbar_m_s=[0,a.speed_max]','fixed_colorbar_m_s=[-a.speed_max,a.speed_max],interpolated_from_height_km=original_heights,radial_mask_km=500')
source=source.replace('"-c:v","libx264"','"-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2","-c:v","libx264"')
source=source.replace('if __name__=="__main__": main()','')
cases={'CTRL':'cm1out_25N_CTRL_R2_Thompson_OML100_240h.nc','JET30':'cm1out_25N_JET_R2_Thompson_OML100_240h.nc'}
for label,name in cases.items():
    sys.argv=['existing_radial_animation','--input','/data/zhangyx/DATA/'+name,'--output',str(OUT/(label+'.mp4')),'--height-km','11','--start-hour','50','--end-hour','150','--interval-hours','1','--fps','8','--speed-max','20','--x-half-width-km','500','--y-south-km','500','--y-north-km','500','--jet-offset-km','888','--keep-frames']
    namespace={'__name__':'reused_radial_animation'}
    exec(compile(source,'existing_horizontal_radial_with_display_adjustments','exec'),namespace)
    namespace['main']()
ff='/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/ffmpeg'
subprocess.run([ff,'-n','-i',str(OUT/'CTRL.mp4'),'-i',str(OUT/'JET30.mp4'),'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(OUT/'CTRL_JET30_radial_11km_50_150h.mp4')],check=True)
(OUT/'README.json').write_text(json.dumps(dict(inputs=cases,data_directory='/data/zhangyx/DATA',height_km=11,height='Linear interpolation of destaggered horizontal wind between adjacent scalar levels',range_h=[50,150],interval_h=1,fps=8,radius_km=500,fixed_scale_m_s=[-20,20],definition='Original storm-centred Cartesian wind projected onto radial direction; no storm translation subtraction; centre from spatially smoothed surface pressure minimum',reuse='/tmp/animate_horizontal_radial.py, only height/view/metadata and encoder display adjustments; old code/results untouched',frames_retained=True),indent=2))
print('COMPLETE',flush=True)
