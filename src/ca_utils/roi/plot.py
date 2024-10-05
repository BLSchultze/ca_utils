"""Plot rois and traces."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import itertools
import xarray as xr
import seaborn as sns
from typing import Optional, List


def set_colors(nb_colors: int, set_cycle: bool = True):
    """Generate HUSL palette from seaborn with specified number of colors.

    Args:
        nb_colors (int): Number of colors in palette.
        set_cycle (bool, optional): Set the default color cycle in matplotlib with the new palette. Defaults to True.

    Returns:
        matplotlib.colors.ListedColormap
    """
    palette = sns.color_palette("husl", n_colors=nb_colors)  # a list of RGB tuples
    if set_cycle:
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=palette)

    cmap = matplotlib.colors.ListedColormap(palette)
    return cmap


def plot_rois(rois: xr.DataArray, layer: int = 0, background: Optional[xr.DataArray] = None, auto_colors=True):
    """_summary_

    Args:
        rois (xr.DataArray): ROIs.
        layer (int): Z-layer to show ROIs for. Defaults to 0 (first layer).
        background (_type_, optional): _description_. Defaults to None.
        auto_colors (bool, optional): Use HUSL palette. Defaults to True.
    """
    if auto_colors:
        cmap = set_colors(len(rois.roi))
    else:
        cmap = None
    msk = np.full(rois.shape[1:3], np.nan)
    for cnt in range(len(rois.roi)):
        msk[rois[layer, ..., cnt]] = cnt + 2
    if background is not None:
        plt.imshow(background, cmap="Greys")
    plt.imshow(msk, cmap=cmap, alpha=0.5)
    plt.xlabel(str(rois.dims[1]))
    plt.ylabel(str(rois.dims[2]))


def extract_one_trial(stack: xr.DataArray, rois: xr.DataArray):
    """Extract fluorescence trace for a single trial
    Args:
        stack - stacks (with dims [T, Z, X, Y, C])
        rois - N rois (with dims [Z, X, Y, N])
    Returns:
        xr.DataArray with traces [T, C, N]
    """
    nb_layers = rois.shape[0]
    nb_rois = rois.shape[-1]

    nb_times = stack.shape[0]
    nb_channels = stack.shape[-1]
    trace = xr.DataArray(
        np.zeros((nb_times, nb_channels, nb_rois)),
        name="traces",
        dims=["time", "channel", "roi"],
        coords={"time": stack.time, "channel": stack.channel},
        attrs={"stim_info": stack.attrs["stim_info"].copy()},
    )
    for roi in range(nb_rois):
        for chan, layer in itertools.product(range(nb_channels), range(nb_layers)):
            stack.data[:, layer, ..., chan]
            rois.data[layer, ..., roi]
            trace[:, chan, roi] += np.mean(stack[:, layer, ..., chan] * rois[layer, ..., roi], axis=(-2, -1))
    return trace


def extract_traces(session, rois):
    """Extract fluorescence traces for ROIs for all trials in session

    Args:
        session (ca_utils.io.Session):
        rois (xr.DataArray): Rois [Z, X, Y, N]

    Returns:
        xr.DataArray with traces [T, C, N]
    """
    traces = []
    for trial_number in range(session.nb_trials):
        trial = session.stack(trial_number)
        trace = extract_one_trial(trial, rois)
        traces.append(trace)
    return traces


def dff(traces: xr.DataArray, f0_seconds: List[float]):
    """Compute dF/F

    Args:
        traces (List[xr.DataArray]):list of traces. Each item [T,C,N].
        f0_seconds (List[float]): start and stop seconds of time span to calculate F0 from.

    Returns:
        List[xr.DataArrau]: List of normalized traces.
    """
    traces_dff = []
    for trial_number in range(len(traces)):
        trace = traces[trial_number]
        f0 = trace.sel(time=slice(*f0_seconds)).mean(dim="time")
        dff = (trace - f0) / f0
        dff.attrs = trace.attrs.copy()
        traces_dff.append(dff)
    return traces_dff


def smooth(traces, sigma: float = 4.0):
    """Smooth traces with Gaussian kernel.

    Args:
        traces (List[xr.DataArray]):list of traces. Each item [T,C,N].
        sigma (float): Width of Gaussian kernel in time steps. Higher values lead to more smoothing. Defaults to 4.0.

    Returns:
        List[xr.DataArrau]: List of smoothed traces.
    """
    traces_smooth = []
    for trial_number in range(len(traces)):
        trace_smooth = traces[trial_number].copy()  # copy and replace data to keep all axes info
        trace_smooth.data = gaussian_filter(traces[trial_number], sigma=sigma, axes=0, mode="nearest")
        traces_smooth.append(trace_smooth)
    return traces_smooth
