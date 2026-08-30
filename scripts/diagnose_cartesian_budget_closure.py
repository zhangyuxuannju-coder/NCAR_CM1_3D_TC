from netCDF4 import Dataset
import numpy as np
import sys

SUFFIXES = ("hadv", "vadv", "cor", "pgrad", "hidiff", "vidiff", "hturb", "vturb", "rdamp")

def stats(obs, model, mask):
    x, y = obs[mask], model[mask]
    use = np.isfinite(x) & np.isfinite(y)
    x, y = x[use], y[use]
    corr = np.corrcoef(x, y)[0, 1]
    slope = np.sum(x*y) / np.sum(y*y)
    rmsx = np.sqrt(np.mean(x*x)); rmsy = np.sqrt(np.mean(y*y))
    residual = np.sqrt(np.mean((x-y)**2)) / rmsx
    scaled = np.sqrt(np.mean((x-slope*y)**2)) / rmsx
    return dict(corr=float(corr), model_obs_rms=float(rmsy/rmsx),
                optimal_model_scale=float(slope), normalized_residual=float(residual),
                scaled_residual=float(scaled))

for path in sys.argv[1:]:
    with Dataset(path) as ds:
        t = np.asarray(ds.variables["time"][:], float)
        i = int(np.argmin(abs(t-72*3600)))
        k = int(np.argmin(abs(np.asarray(ds.variables["zh"][:])-12.0)))
        ps = np.asarray(ds.variables["psfc"][i])
        iy, ix = np.unravel_index(np.nanargmin(ps), ps.shape)
        xc = float(ds.variables["xh"][ix]); yc=float(ds.variables["yh"][iy])
        results = {}
        for comp, xname, yname in (("u","xf","yh"),("v","xh","yf")):
            q0=np.asarray(ds.variables[comp][i,k],float)
            x=np.asarray(ds.variables[xname][:],float); y=np.asarray(ds.variables[yname][:],float)
            xx,yy=np.meshgrid(x,y); mask=np.hypot(xx-xc,yy-yc)<=500
            results[comp]={}
            for hw in (1,2,4,8,12):
                im=i-hw; ip=i+hw
                qm=np.asarray(ds.variables[comp][im,k],float)
                qp=np.asarray(ds.variables[comp][ip,k],float)
                obs=(qp-qm)/(t[ip]-t[im])
                samples=[]
                for j in range(im,ip+1):
                    samples.append(sum((np.asarray(ds.variables[f"{comp}b_{s}"][j,k],float)
                                        for s in SUFFIXES if f"{comp}b_{s}" in ds.variables),
                                       np.zeros_like(q0)))
                model=np.trapezoid(np.stack(samples),x=t[im:ip+1],axis=0)/(t[ip]-t[im])
                results[comp][f"centered_{2*hw}h"] = stats(obs,model,mask)
        print(path, "z_km", float(ds.variables["zh"][k]), "center", xc, yc)
        print(results)
