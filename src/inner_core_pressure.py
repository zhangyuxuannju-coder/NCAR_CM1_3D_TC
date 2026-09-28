"""Resolved dry-air transport diagnostics; unavailable terms remain missing."""
import numpy as np
from scipy.integrate import quad
G=9.81

def circle_cell_weights(xf,yf,cx,cy,radius):
    weights=np.zeros((len(yf)-1,len(xf)-1))
    for j in np.where((yf[:-1]<cy+radius)&(yf[1:]>cy-radius))[0]:
        ly,hy=yf[j]-cy,yf[j+1]-cy
        for i in np.where((xf[:-1]<cx+radius)&(xf[1:]>cx-radius))[0]:
            lo,hi=max(xf[i]-cx,-radius),min(xf[i+1]-cx,radius)
            def height(xx):
                yy=np.sqrt(max(0.,radius**2-xx**2))
                return max(0.,min(hy,yy)-max(ly,-yy))
            points=[0.]
            for yy in (ly,hy):
                if abs(yy)<radius:
                    xx=np.sqrt(radius**2-yy**2); points.extend([-xx,xx])
            weights[j,i]=quad(height,lo,hi,points=sorted(p for p in set(points) if lo<p<hi),epsabs=.01)[0]
    if not np.isclose(weights.sum(),np.pi*radius**2,rtol=1e-7):
        raise ValueError('Incomplete disk coverage or area quadrature failure')
    return weights

def side_terms(rho,ur,dz,radius):
    mr,mu=rho.mean(1),ur.mean(1)
    mean=-2*G/radius*36*np.sum(mr*mu*dz)
    eddy=-2*G/radius*36*np.sum(((rho-mr[:,None])*(ur-mu[:,None])).mean(1)*dz)
    total=-2*G/radius*36*np.sum((rho*ur).mean(1)*dz)
    if not np.isclose(mean+eddy,total,atol=1e-10): raise ValueError('mean+eddy failed')
    return mean,eddy,total

def window_average(values,seconds):
    return np.trapezoid(values,seconds)/(seconds[-1]-seconds[0])
