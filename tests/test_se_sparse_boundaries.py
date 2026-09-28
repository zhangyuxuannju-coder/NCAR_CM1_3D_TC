import unittest
from unittest.mock import patch
import numpy as np
from src._se_pipeline_single import solve_se_sparse,psi_to_uw

class SparseBoundaryTests(unittest.TestCase):
    def test_matrix_matches_consistent_ghost_stencil(self):
        rng=np.random.default_rng(42);nr,nz=6,7;dr,dz=2.,3.
        fields=[rng.normal(size=(nr,nz)) for _ in range(5)];p=rng.normal(size=(nr,nz));box={}
        def capture(m,b):box['m']=m;return np.zeros(b.size)
        with patch('scipy.sparse.linalg.spsolve',capture):solve_se_sparse(*fields,np.zeros_like(p),dr,dz)
        def v(i,j):
            if j<0:return 0.
            if j==nz:j=nz-2
            if i<0:i=1
            if i==nr:i=nr-2
            return p[i,j]
        expected=np.zeros_like(p)
        for i in range(nr):
            for j in range(nz):
                xx=(v(i+1,j)+v(i-1,j)-2*v(i,j))/dr**2
                yy=(v(i,j+1)+v(i,j-1)-2*v(i,j))/dz**2
                xy=(v(i+1,j+1)-v(i+1,j-1)-v(i-1,j+1)+v(i-1,j-1))/(4*dr*dz)
                x=(v(i+1,j)-v(i-1,j))/(2*dr);y=(v(i,j+1)-v(i,j-1))/(2*dz)
                expected[i,j]=sum(f[i,j]*d for f,d in zip(fields,[xx,xy,yy,x,y]))
        np.testing.assert_allclose((box['m']@p.ravel()).reshape(p.shape),expected,atol=1e-14)
    def example(self,nz=9):
        nr=6;H=10.;dz=H/nz;z=np.arange(1,nz+1)*dz
        q=np.broadcast_to(z*(2*H-z),(nr,nz));one=np.ones_like(q);zero=np.zeros_like(q)
        p=solve_se_sparse(one,zero,one,zero,zero,-2*one,1.,dz)
        return p,q,dz
    def test_exact_quadratic_solution(self):
        p,q,dz=self.example();np.testing.assert_allclose(p[:,1:-1],q,rtol=1e-12,atol=1e-12)
    def test_top_mirror_and_bottom_zero(self):
        p,q,dz=self.example();np.testing.assert_array_equal(p[:,-1],p[:,-3]);np.testing.assert_array_equal(p[:,0],0.)
    def test_winds_respect_psi_boundary(self):
        p,q,dz=self.example();r=np.arange(1,p.shape[0]+1,dtype=float);u,w=psi_to_uw(p,np.ones_like(p),r,1.,dz)
        np.testing.assert_allclose(u[:,-2],0.,atol=1e-13);np.testing.assert_array_equal(w[[0,-1]],0.)
        np.testing.assert_allclose(w[:,1:-1],0.,atol=1e-12)
        self.assertGreater(np.max(np.abs(u[0,1:-2])),0.)
    def test_linearity_and_small_residual(self):
        rng=np.random.default_rng(3);one=np.ones((6,8));zero=np.zeros_like(one);b=rng.normal(size=one.shape);c=rng.normal(size=one.shape)
        p=solve_se_sparse(one,zero,one,zero,zero,b,1.,1.);q=solve_se_sparse(one,zero,one,zero,zero,c,1.,1.);total=solve_se_sparse(one,zero,one,zero,zero,b+c,1.,1.)
        np.testing.assert_allclose(p+q,total,atol=1e-12)
        box={}
        def capture(m,rhs):box['m']=m;return np.zeros(rhs.size)
        with patch('scipy.sparse.linalg.spsolve',capture):solve_se_sparse(one,zero,one,zero,zero,b,1.,1.)
        self.assertLess(np.linalg.norm(box['m']@p[:,1:-1].ravel()-b.ravel())/np.linalg.norm(b),1e-12)
    def test_zero_rhs(self):
        one=np.ones((5,6));zero=np.zeros_like(one);np.testing.assert_array_equal(solve_se_sparse(one,zero,one,zero,zero,zero,1.,1.),0.)
    def test_optional_inner_axis_dirichlet_removes_axis_wind_singularity(self):
        nr,nz=6,8;one=np.ones((nr,nz));zero=np.zeros_like(one)
        rhs=np.broadcast_to(np.linspace(1.,2.,nz),(nr,nz)).copy()
        p=solve_se_sparse(one,zero,one,zero,zero,rhs,1.,1.,inner_axis_dirichlet=True)
        np.testing.assert_array_equal(p[0,1:-1],0.)
        u,w=psi_to_uw(p,np.ones_like(p),np.arange(nr,dtype=float)+.5,1.,1.)
        np.testing.assert_array_equal(u[0],0.)
        np.testing.assert_array_equal(w[0],0.)
        self.assertTrue(np.all(np.isfinite(p)))

if __name__=='__main__':unittest.main()
