import numpy as np

def li_ma_significance(N_on, N_off, alpha=0.2):
    if N_on + N_off == 0:
        return 0

    p_on = N_on / (N_on + N_off)
    p_off = N_off / (N_on + N_off)

    if p_on ==0:
        return 0

    t1 = N_on * np.log(((1 + alpha) / alpha) * p_on)
    t2 = N_off * np.log((1 + alpha) * p_off)

    ts = (t1 + t2)

    significance = np.sqrt(ts*2)

    return significance

def theta_mm_to_theta_squared_deg(theta):
    pixelsize = 9.5 #mm
    fov_per_pixel = 0.11 #degree
    return (theta*(fov_per_pixel/pixelsize))**2
