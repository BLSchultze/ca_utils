import numpy as np
import scipy.interpolate as interp1d


def temporally_align_traces(x, y, fs=None, kind='linear', bounds_error=False, fill_value='extrapolate'):
    """Resample and align list of unevenly sampled traces along time axis.

    Args:
        x - list of [T,] np.arrays of time stamps for each item in y
        y - list of [T, ...] np.arrays of fluoresence values
        fs=None - target sampling rate, optional - will infer fs from x values
    Returns:
        x_new - single np.array containing common time base
        y_new - list of np.arrays of fluoresence values for the new time base

    """
    if fs is None:
        tmin = None
        tmax = None
        nb_steps = None
        time_base = np.linspace(tmin, tmax, nb_steps)  # infer from x

    time_base = None  # make new time base
    y_new = []
    for xx, yy in zip(x, y):
        interpolate = interp1d(xx, yy, axis=0, kind=kind, bounds_error=bounds_error, fill_value=fill_value)
        y_new.append(interpolate(time_base))

    return time_base, y_new
