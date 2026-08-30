#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--projection',choices=['raw','balanced'],default='raw');a=p.parse_args()
x=np.load(a.input);r=x['r_km'];z=x['z_km']
prefix='stability_class' if a.projection=='raw' else 'stability_class_balanced_projection'
c=x[f'{prefix}_noJET']==0;j=x[f'{prefix}_JET']==0;good=c&j
rows=[]
for r0 in np.arange(6,181,12):
 for r1 in np.arange(r0+60,301,12):
  for z0 in np.arange(0,10.1,1):
   for z1 in np.arange(max(z0+4,10),18.1,1):
    m=(r[None]>=r0)&(r[None]<=r1)&(z[:,None]>=z0)&(z[:,None]<=z1)
    n=int(m.sum())
    if n and np.all(good[m]): rows.append(dict(r_min=float(r0),r_max=float(r1),z_min=float(z0),z_max=float(z1),grid_points=n,rz_area=float((r1-r0)*(z1-z0))))
rows=sorted(rows,key=lambda q:(q['rz_area'],q['grid_points']),reverse=True)
Path(a.output).write_text(json.dumps({'hour':float(x['hour']),'projection':a.projection,'top_rectangles':rows[:30]},indent=2),encoding='utf-8')
print(json.dumps({'hour':float(x['hour']),'projection':a.projection,'top_rectangles':rows[:10]},indent=2))
