# exact semi-analytic calculation of planetary microlensing magnifications
# https://arxiv.org/abs/2207.12412
# author: Keming Zhang

import numpy as np
from time import time
from solver import multi_quartic

print_time = False
n_iter_max = 50

def mag_map_exact(s, q, x_grid, y_grid, laguerre=True, eps=1e-14, eps_check=1e-4):
    # calculate magnification maps
    coords = np.meshgrid(x_grid, y_grid)
    mag = magnification(coords[0].flatten(), coords[1].flatten(), s, q, laguerre, eps, eps_check)
    return mag.reshape(len(y_grid),len(x_grid))

def magnification(xs, ys, s, q, laguerre=True, eps=1e-14, eps_check=1e-4):
    # calculate light-curve using exact semi-analytic approach
    # xs and ys are 1D arrays of source trajectory
    
    # source trajectory in complex coordinates
    # note that zeta
    t0 = time()
    zeta = xs + ys * 1j
    zetac = np.conj(zeta)
    
    # z_stationary is the unperturbed location of the weakly perturbed image
    # i.e. one of the two solutions to "zeta = z-1/z_conj" that has Real(z)<0
    z_stationary = zeta / 2
    z_stationary2 = np.sqrt(zeta*zetac*(4+zeta*zetac)) / 2 / zetac
    major = np.real(zeta) > 0
    z_stationary[major] -= z_stationary2[major]
    z_stationary[~major] += z_stationary2[~major]
    
    if print_time:
        print('t=%.4fs for solving unperturbed location'%(time()-t0))
    
    # now let's use laguerre's or Newton's method to find one quintic root
    # using z_stationary as the initial guess
    t0 = time()
    f = None
    if laguerre:
        for i in range(n_iter_max):
            if f is None:
                f = poly(s, q, zeta, zetac, z_stationary)
            else:
                f[mask] = poly(s, q, zeta[mask], zetac[mask], z_stationary[mask])
            mask = np.abs(f) > eps
            if mask.sum() == 0:
                break
            df = dpoly(s, q, zeta[mask], zetac[mask], z_stationary[mask])
            ddf = ddpoly(s, q, zeta[mask], zetac[mask], z_stationary[mask])

            G = df/f[mask]
            G2 = G**2
            H = G2 - ddf/f[mask]
            p = np.sqrt(4*(5*H-G2))
            denom1 = G + p
            denom2 = G - p
            m = np.abs(denom1) < np.abs(denom2)
            denom1[m] = denom2[m]
            a = 5 / denom1
            z_stationary[mask] -= a


    # Newton's method
    else:
        for i in range(n_iter_max):
            if f is None:
                f = poly(s, q, zeta, zetac, z_stationary)
            else:
                f[mask] = poly(s, q, zeta[mask], zetac[mask], z_stationary[mask])
            mask = np.abs(f) > eps
            if mask.sum()==0:
                break
            df = dpoly(s, q, zeta[mask], zetac[mask], z_stationary[mask])
            z_stationary[mask] -= f[mask] / df
    if print_time:
        print('t=%.4fs for root-refinement'%(time()-t0))

    t0 = time()
    # after refinement z_stationary becomes roots of the 5th order polynomial
    # allowing us to transform the lens equation to a 4th order polynomial
    zs = solve_four(s, q, zeta, zetac, z_stationary, eps_check)
    if print_time:
        print('t=%.4fs for solving quartic'%(time()-t0))

    t0 = time()
    # add back the weakly perturbed image solution
    zs = np.c_[np.array(zs).T,z_stationary].T

    # calculate the jacobian at each image location
    jac = 1-np.abs(1/zs**2+q/(zs-s)**2)**2
    mag = np.abs(1/jac)
    
    # invalid image locations are marked with -999
    # and does not contribute to the total magnification
    mag[zs == -999] = 0
    mag = mag.sum(0)
    if print_time:
        print('t=%.4fs for evaluating magnification'%(time()-t0))
    return mag

def poly(s, q, zeta, zetac, z):
    # lens equation: quintic polynomial
    return -(s-z)**2*(-1+s*z)*zeta+q**2*z**2*(-s+zeta)-\
            q*(s-z)*z*(2*zeta+s*(-1+z**2-z*zeta))+(s-\
            z)*z*((1+q)*z*(z-2*zeta)+s**2*z*(z-zeta)+\
            s*((-1+q)*z-z**3+2*zeta+\
            z**2*zeta))*zetac-(s-\
            z)**2*z**2*(z-zeta)*zetac**2

def dpoly(s, q, zeta, zetac, z):
    # 1st order derivative
    return -((s**3-2*z-4*s**2*z+s*(2+3*z**2))*zeta)+\
            2*q**2*z*(-s+zeta)+\
            q*(4*z*zeta+s**2*(1-3*z**2+2*z*zeta)+\
            s*(-2*z+4*z**3-2*zeta-3*z**2*zeta))+(-2*(1+\
            q)*z**2*(2*z-3*zeta)+s**3*z*(3*z-2*zeta)+\
            s*z*(6*z+5*z**3-4*(2+q)*zeta-4*z**2*zeta)+\
            2*s**2*((-1+q)*z-4*z**3+zeta+\
            3*z**2*zeta))*zetac-(s-z)*z*(3*s*z-\
            5*z**2-2*s*zeta+4*z*zeta)*zetac**2


def ddpoly(s, q, zeta, zetac, z):
    # 2nd order derivative required by laguerre's method
    return -2*(q**2*(s-zeta)+(-1-2*s**2+3*s*z)*zeta+
            q*(s+3*s**2*z-6*s*z**2-2*zeta-s**2*zeta+
            3*s*z*zeta)+(6*(1+q)*z*(z-zeta)+
            s**3*(-3*z+zeta)-s**2*(-1+q-12*z**2+6*z*zeta)+
            2*s*(-3*z-5*z**3+(2+q)*zeta+
            3*z**2*zeta))*zetac+(2*z**2*(5*z-
            3*zeta)+s**2*(3*z-zeta)+
            6*s*z*(-2*z+zeta))*zetac**2)


def check_img_5(s,q, z, zeta, eps_check=1e-4):
    # check if polynomial roots are solution to the lens equation
    zeta0=z-1/np.conj(z)-q/(np.conj(z)-s)
    return np.abs(zeta0-zeta)<eps_check

def solve_four(s, q, zeta, zetac, z0, eps_check=1e-4):
    # evaluate reduced quartic polynomial in closed-form

    coeff0 = -((s**3-2*z0-4*s**2*z0+s*(2+3*z0**2))*zeta)+\
            2*q**2*z0*(-s+zeta)+\
            q*(4*z0*zeta+s**2*(1-3*z0**2+2*z0*zeta)+\
            s*(-2*z0+4*z0**3-2*zeta-3*z0**2*zeta))+(-2*(1+\
            q)*z0**2*(2*z0-3*zeta)+s**3*z0*(3*z0-2*zeta)+\
            s*z0*(6*z0+5*z0**3-4*(2+q)*zeta-4*z0**2*zeta)+\
            2*s**2*((-1+q)*z0-4*z0**3+zeta+\
            3*z0**2*zeta))*zetac-(s-z0)*z0*(3*s*z0-\
            5*z0**2-2*s*zeta+4*z0*zeta)*zetac**2
                
    coeff1 = (1+2*s**2-3*s*z0)*zeta+q**2*(-s+zeta)+\
            q*(2*zeta+s**2*(-3*z0+zeta)+\
            s*(-1+6*z0**2-3*z0*zeta))+(-6*(1+q)*z0*(z0-zeta)+\
            s**3*(3*z0-zeta)+s**2*(-1+q-12*z0**2+6*z0*zeta)+\
            2*s*(3*z0+5*z0**3-(2+q)*zeta-\
            3*z0**2*zeta))*zetac+(6*s*z0*(2*z0-zeta)\
            +s**2*(-3*z0+zeta)+\
            2*z0**2*(-5*z0+3*zeta))*zetac**2
    
    coeff2 = -s*(zeta+q*(s-4*z0+zeta))+(s**3-\
            2*(1+q)*(2*z0-zeta)+s**2*(-8*z0+2*zeta)+\
            2*s*(1+5*z0**2-2*z0*zeta))*zetac-(s**2-\
            8*s*z0+10*z0**2+2*s*zeta-\
            4*z0*zeta)*zetac**2
                
    coeff3 = q*s-(1+q+2*s**2-5*s*z0+\
            s*zeta)*zetac+(2*s-\
            5*z0+zeta)*zetac**2

    coeff4 = s*zetac-zetac**2
    
    coeff = np.array([coeff0, coeff1, coeff2, coeff3, coeff4])

    img = np.array(multi_quartic(*list(coeff[::-1])))

    img += z0
    mask = check_img_5(s, q, img, zeta, eps_check)

    img[~mask] = -999
    return img
