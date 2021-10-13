import numpy as np
import numba
from typing import List, Optional


@numba.jit(nopython=True, parallel=True)
def parallel_interpolate(pixel_times, pixel_values, new_times):
    new_stack_traces = np.zeros((pixel_times.shape[0], len(new_times)))
    for cnt in numba.prange(pixel_times.shape[0]):
        new_stack_traces[cnt, :] = np.interp(new_times, pixel_times[cnt, ...], pixel_values[cnt, ...])
    return new_stack_traces


def reslice_time(stack: np.ndarray, frameonset_ms: List[float], frameoffset_ms: List[float],
                 new_times: Optional[List[float]] = None,
                 new_fps: Optional[float] = None):
    """Reslice time

    Args:
        stack (np.ndarray): 4D stack as a time series of 3D volumes - [volumes, zpos, xpos, ypos] - [xpos, ypos] defines a single frame, [zpos, xpos, ypos] defines a single stack
        frameonset_ms (List[float]): time of the first pixel of each frame, in ms. Should be len (volume * xpos)
        frameoffset_ms (List[float]): time of the last pixel of each frame, in ms. Should be len (volume * xpos)
        new_times (List[float]): List of volume times. Optional.
        new_fps (float): New frame/volume rate. New volumes times will start at midpoints of first and last frame.
                         Optional (ignored if new_times is set)

    Raises:
        ValueError if neither new_times nor new_fps is set.

    Returns:
        np.ndarray: new stack shape [new times, zpos, xpos, ypos]
        np.ndarray: new times start

    """
    if stack.ndim!=4:
        raise ValueError("Stack is not 4D (shape is {stack.shape}. Should be [vol,z,x,y].")

    if new_times is None and new_fps is None:
        raise ValueError("Need to provide either new_times or new_fps.")

    # generate `new_times` based on `new_fps` from mid time points of first and last frame.
    if new_times is None and new_fps is not None:
        frameinterval_ms = np.array(frameoffset_ms) - np.array(frameonset_ms)
        frame_mid_points = np.array(frameonset_ms) + frameinterval_ms/2
        new_times = np.arange(frame_mid_points[0], frame_mid_points[-1], 1000/new_fps)

    # infer time of each pixel based on frame on- and offset times
    px_per_frame = stack.shape[2] * stack.shape[3]
    pixel_times = np.linspace(frameonset_ms, frameoffset_ms, px_per_frame)
    pixel_times = np.reshape(pixel_times.T, stack.shape)

    new_stack = interpnd(pixel_times, stack, new_times)
    return new_stack, new_times


def reslice_time_and_zpos(stack: np.ndarray, frameonset_ms: List[float], frameoffset_ms: List[float],
                          new_times: Optional[List[float]] = None,
                          new_fps: Optional[float] = None,
                          frames_zpos = None,
                          new_nb_zpos: Optional[int] = None,
                          new_zpos: Optional[List[float]] = None):
    """Reslice time

    Args:
        stack (np.ndarray): 4D stack as a time series of 3D volumes - [volumes, zpos, xpos, ypos] - [xpos, ypos] defines a single frame, [zpos, xpos, ypos] defines a single stack
        frameonset_ms (List[float]): time of the first pixel of each frame, in ms. Should be len (volume * xpos)
        frameoffset_ms (List[float]): time of the last pixel of each frame, in ms. Should be len (volume * xpos)
        new_times (List[float]): List of volume times. Optional.
        new_fps (float): New frame/volume rate. New volumes times will start at midpoints of first and last frame.
                         Optional (ignored if new_times is set)
        frames_zpos: Optional = None: zpos for each frame and pixel (same shape as stack)
        new_zpos (List[float], optional): New z positions
        new_nb_zpos: Optional[int] = None): Number of new z positions - will start at 5-percentile and and at 95-percentile of the original stack values.
                                            Ignored if new_zpos is set.

    Raises:
        ValueError if neither new_times nor new_fps is set.

    Returns:
        np.ndarray: new stack shape [new times, zpos, xpos, ypos]
        np.ndarray: new times start at min(frameonset_ms) and at max(frameoffset_ms)

    """
    if stack.ndim!=4:
        raise ValueError(f"Stack is not 4D (shape is {stack.shape}. Should be [vol,z,x,y].")

    if new_times is None and new_fps is None:
        raise ValueError("Need to provide either new_times or new_fps.")


    # generate `new_times` based on `new_fps` from mid time points of first and last frame.
    if new_times is None and new_fps is not None:
        frameinterval_ms = np.array(frameoffset_ms) - np.array(frameonset_ms)
        frame_mid_points = np.array(frameonset_ms) + frameinterval_ms/2
        new_times = np.arange(frame_mid_points[0], frame_mid_points[-1], 1000/new_fps)

    # infer time of each pixel based on frame on- and offset times
    pixel_values = stack
    px_per_frame =  pixel_values.shape[2] *  pixel_values.shape[3]
    pixel_times = np.linspace(frameonset_ms, frameoffset_ms, px_per_frame)
    pixel_times = np.reshape(pixel_times.T,  pixel_values.shape)

    new_pixel_values = interpnd(pixel_times, pixel_values, new_times)

    if frames_zpos is not None and (new_nb_zpos is not None or new_zpos is not None):
        # resample zpos to new time grid
        pixel_zpos = frames_zpos.reshape(pixel_values.shape)
        new_pixel_zpos = interpnd(pixel_times, pixel_zpos, new_times)

        # generate new zpos
        if new_zpos is None:
            z_start, z_end = np.percentile(new_pixel_zpos[:, ::10], [5, 95])
            new_zpos = np.linspace(z_start, z_end, new_nb_zpos)

        new_pixel_values = interpnd(new_pixel_zpos, new_pixel_values, new_zpos, axis=1)
        return new_pixel_values, new_times, new_zpos
    else:
        return new_pixel_values, new_times


def interpnd(X, Y, new_x, axis=0):
    if X.shape != Y.shape:
        raise ValueError("X and Y must have same shape. X.shape is {X.shape} and Y.shape is {Y.shape}")

    ndim = X.ndim
    if axis != 0:
        new_axes_order = np.arange(ndim)
        new_axes_order = new_axes_order[new_axes_order != axis]
        new_axes_order = (axis, *new_axes_order)
        X = X.transpose(new_axes_order)
        Y = Y.transpose(new_axes_order)

    # interp along first dim
    xx = np.reshape(X, (X.shape[0], -1)).T
    yy = np.reshape(Y, (Y.shape[0], -1)).T

    # cast all to float - otherwise numba errors
    xx = xx.astype(float)
    yy = yy.astype(float)
    new_x = new_x.astype(float)

    # reslice time
    new_yy = parallel_interpolate(xx, yy, new_x)
    del xx
    del yy

    # reshape to [new_volumes, zpos, xpos, ypos]
    new_Y = np.reshape(new_yy, (*Y.shape[1:], -1))
    new_Y = new_Y.transpose((3,0,1,2))
    del new_yy
    if axis != 0:
        restored_axes = np.arange(ndim)[np.argsort(new_axes_order)]
        new_Y = new_Y.transpose(restored_axes)
    return new_Y