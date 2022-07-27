# exact semi-analytic calculation of planetary microlensing magnifications
# https://arxiv.org/abs/2207.12412
# author: Keming Zhang

import numpy as np
from solver import multi_quartic

def magnification(xs, ys, s, q, n_max=10, laguerre=False, eps=1e-12, eps_check=1e-4):
    # calculate light-curve using exact semi-analytic approach
    # xs and ys are 1D arrays of source trajectory
    zeta = xs + ys * 1j
    zetac = np.conj(zeta)
    z_stationary = zeta / 2
    z_stationary2 = (zeta*zetac*(4+zeta*zetac))**0.5 / 2 / zetac
    major = np.real(zeta) > 0
    z_stationary[major] -= z_stationary2[major]
    z_stationary[~major] += z_stationary2[~major]

    if laguerre:
        for i in range(n_max):
            f = poly(s, q, zeta, zetac, z_stationary)
            mask = np.abs(f) > eps
            if mask.sum() == 0:
                    break
            df = dpoly(s, q, zeta[mask], zetac[mask], z_stationary[mask])
            ddf = ddpoly(s, q, zeta[mask], zetac[mask], z_stationary[mask])

            G = df/f[mask]
            H = G**2 - ddf/f[mask]
            denom1 = G + (4*(5*H-G**2))**0.5
            denom2 = G - (4*(5*H-G**2))**0.5
            m = np.abs(denom1) < np.abs(denom2)
            denom1[m] = denom2[m]
            a = 5 / denom1
            z_stationary[mask] -= a

    # Newton's method
    else:
        for i in range(n_max):
            f = poly(s, q, zeta, zetac, z_stationary)
            mask = np.abs(f) > eps
            if mask.sum()==0:
                break
            df = dpoly(s, q, zeta[mask], zetac[mask], z_stationary[mask])
            z_stationary[mask] -= f[mask] / df
    zs = solve_four(s, q, zeta, zetac, z_stationary, eps_check)
    zs = np.c_[np.array(zs).T,z_stationary].T

    jac = 1-np.abs(1/zs**2+q/(zs-s)**2)**2
    mag = np.abs(1/jac)
    mag[zs == -999] = 0
    mag = mag.sum(0)
    return mag

def mag_map_exact(s, q, x_grid, y_grid, n_max=5, laguerre=True, eps=1e-12, eps_check=1e-4):
    # calculate magnification maps
    coords = np.meshgrid(x_grid, y_grid)
    mag = magnification(coords[0].flatten(), coords[1].flatten(), s, q, n_max, laguerre, eps, eps_check)
    return mag.reshape(len(y_grid),len(x_grid))

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


def check_img_5(s,q, z, zeta, eps=1e-4):
    # check if polynomial roots are solution to the lens equation
    zeta0=z-1/np.conj(z)-q/(np.conj(z)-s)
    return np.abs(zeta0-zeta)<eps

def solve_four(s, q, zeta, zetac, z0, eps_check=1e-4):
    # evaluate reduced quartic polynomial in closed-form
    s = s + np.zeros_like(z0)
    q = q + np.zeros_like(z0)

    coeff = np.array([q*s**2-2*q*s*z0-2*q**2*s*z0-3*q*s**2*z0**2+4*q*s*z0**3-
                2*s*zeta-2*q*s*zeta-s**3*zeta+2*z0*zeta+
                4*q*z0*zeta+2*q**2*z0*zeta+4*s**2*z0*zeta+
                2*q*s**2*z0*zeta-3*s*z0**2*zeta-3*q*s*z0**2*zeta-
                2*s**2*z0*zetac+2*q*s**2*z0*zetac+
                6*s*z0**2*zetac+3*s**3*z0**2*zetac-
                4*z0**3*zetac-4*q*z0**3*zetac-
                8*s**2*z0**3*zetac+5*s*z0**4*zetac+
                2*s**2*zeta*zetac-
                8*s*z0*zeta*zetac-
                4*q*s*z0*zeta*zetac-
                2*s**3*z0*zeta*zetac+
                6*z0**2*zeta*zetac+
                6*q*z0**2*zeta*zetac+
                6*s**2*z0**2*zeta*zetac-
                4*s*z0**3*zeta*zetac-
                3*s**2*z0**2*zetac**2+8*s*z0**3*zetac**2-
                5*z0**4*zetac**2+
                2*s**2*z0*zeta*zetac**2-
                6*s*z0**2*zeta*zetac**2+
                4*z0**3*zeta*zetac**2,-q*s-q**2*s-3*q*s**2*z0+
                6*q*s*z0**2+zeta+2*q*zeta+q**2*zeta+2*s**2*zeta+
                q*s**2*zeta-3*s*z0*zeta-3*q*s*z0*zeta-
                s**2*zetac+q*s**2*zetac+
                6*s*z0*zetac+3*s**3*z0*zetac-
                6*z0**2*zetac-6*q*z0**2*zetac-
                12*s**2*z0**2*zetac+10*s*z0**3*zetac-
                4*s*zeta*zetac-2*q*s*zeta*zetac-
                s**3*zeta*zetac+6*z0*zeta*zetac+
                6*q*z0*zeta*zetac+
                6*s**2*z0*zeta*zetac-
                6*s*z0**2*zeta*zetac-
                3*s**2*z0*zetac**2+12*s*z0**2*zetac**2-
                10*z0**3*zetac**2+s**2*zeta*zetac**2-
                6*s*z0*zeta*zetac**2+
                6*z0**2*zeta*zetac**2,-q*s**2+4*q*s*z0-s*zeta-
                q*s*zeta+2*s*zetac+s**3*zetac-
                4*z0*zetac-4*q*z0*zetac-
                8*s**2*z0*zetac+10*s*z0**2*zetac+
                2*zeta*zetac+2*q*zeta*zetac+
                2*s**2*zeta*zetac-
                4*s*z0*zeta*zetac-s**2*zetac**2+
                8*s*z0*zetac**2-10*z0**2*zetac**2-
                2*s*zeta*zetac**2+
                4*z0*zeta*zetac**2,
                q*s-zetac-q*zetac-
                2*s**2*zetac+5*s*z0*zetac-
                s*zeta*zetac+2*s*zetac**2-
                5*z0*zetac**2+zeta*zetac**2,
                s*zetac-zetac**2])

    img = np.array(multi_quartic(*list(coeff[::-1])))
    img += z0
    mask = check_img_5(s, q, img, zeta, eps_check)

    img[~mask] = -999
    return img