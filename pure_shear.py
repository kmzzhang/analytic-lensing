# pure-shear (approximate) calculation of planetary microlensing magnifications
# https://arxiv.org/abs/2207.12412
# author: Keming Zhang

import numpy as np
from solver import multi_quartic

def magnification(xs, ys, s, q):
    # calculate light-curve using pure-shear approximation
    # xs and ys are 1D arrays of source trajectory
    u = (xs**2 + ys**2)**0.5
    mag_pspl = (u**2+2)/np.abs(u)/(u**2+4)**0.5
    
    z0s = ((xs**2+4)**0.5+xs) / 2
    shear = 1/z0s**2
    
    coords = xs + ys * 1j - (s - 1/s)
    coords /= q**0.5
    
    img = cr_image_multi(shear, coords)
    dmag = cr_mag(shear, img)
    dmag -= np.abs(1/(shear**2-1))
    mag = np.array(dmag) + mag_pspl
    return mag

def mag_map_shear(s, q, x_grid, y_grid):
    # calculate magnification maps
    coords = np.meshgrid(x_grid, y_grid)
    mag = magnification(coords[0].flatten(), coords[1].flatten(), s, q)
    return mag.reshape(len(y_grid),len(x_grid))

def cr_image_multi(g, x):
    cr_coeff = np.array([g, -x + 2*g* np.conj(x), -2* g**2 - x* np.conj(x) + g *np.conj(x)**2,
                         g * x + np.conj(x) - 2 *g**2 *np.conj(x), -g + g**3])
    img = np.array(multi_quartic(*list(cr_coeff[::-1])))
    mask = check_img(g, img, x)
    img[~mask] = -999
    return img

def cr_mag_img(s, q, x_grid, y_grid):
    coords = np.meshgrid(x_grid, y_grid)
    u = (coords[0].flatten()**2 + coords[1].flatten()**2)**0.5
    mag_pspl = (u**2+2)/np.abs(u)/(u**2+4)**0.5
    
    z0s = ((coords[0].flatten()**2+4)**0.5+coords[0].flatten()) / 2
    shear = 1/z0s**2
    
    coords = coords[0].flatten() + coords[1].flatten() * 1j - (s - 1/s)
    coords /= q**0.5
    
    img = cr_image_multi(shear, coords)
    
    dmag = cr_mag(shear, img)
    dmag -= np.abs(1/(shear**2-1))
    mag = np.array(dmag) + mag_pspl
    return mag.reshape(len(y_grid),len(x_grid)), img.reshape(4,len(y_grid),len(x_grid))

def check_img(gamma, z, zeta):
    zeta0=z-1/np.conj(z)+gamma*np.conj(z)
    return np.abs(zeta0-zeta)<1e-6

def cr_mag(gamma, z):
    jac = 1-np.abs(1/z**2+gamma)**2
    mag = np.abs(1/jac)
    mag[z == -999] = 0
    mag = mag.sum(0)
    return mag
