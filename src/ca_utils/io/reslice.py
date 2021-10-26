import numpy as np
import numba
from typing import List, Optional, Tuple, Union


@numba.jit(nopython=True, parallel=True)
def parallel_interpolate(pixel_times: np.ndarray, pixel_values: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        pixel_times (np.ndarray): [description]
        pixel_values (np.ndarray): [description]
        new_times (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    new_stack_traces = np.zeros((pixel_times.shape[0], len(new_times)))
    for cnt in numba.prange(pixel_times.shape[0]):
        new_stack_traces[cnt, :] = np.interp(
            new_times, pixel_times[cnt, ...], pixel_values[cnt, ...])
    return new_stack_traces


def reslice_time(stack: np.ndarray, frameonset_ms: List[float], frameoffset_ms: List[float],
                 new_times: Optional[List[float]] = None,
                 new_fps: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Reslice time

    Args:
        stack (np.ndarray): 4D stack as a time series of 3D volumes [volumes, zpos, xpos, ypos]
                            [xpos, ypos] defines a single frame, [zpos, xpos, ypos] defines a single stack
        frameonset_ms (List[float]): time of the first pixel of each frame, in ms. Should be len (volume * xpos)
        frameoffset_ms (List[float]): time of the last pixel of each frame, in ms. Should be len (volume * xpos)
        new_times (List[float]): List of volume times. Optional.
        new_fps (float): New frame/volume rate. New volumes times will start at midpoints of first and last frame.
                         Optional (ignored if new_times is set)

    Raises:
        ValueError if neither new_times nor new_fps is set.
        ValueError if stack is not 4D.

    Returns:
        np.ndarray: new stack shape [new times, zpos, xpos, ypos]
        np.ndarray: new times start

    """
    if stack.ndim != 4:
        raise ValueError(f"Stack is not 4D (shape is {stack.shape}. Should be [vol,z,x,y].")

    if new_times is None and new_fps is None:
        raise ValueError("Need to provide either new_times or new_fps.")

    # generate `new_times` based on `new_fps` from mid time points of first and last frame.
    if new_times is None and new_fps is not None:
        frameinterval_ms = np.array(frameoffset_ms) - np.array(frameonset_ms)
        frame_mid_points = np.array(frameonset_ms) + frameinterval_ms/2
        new_times = np.arange(
            frame_mid_points[0], frame_mid_points[-1], 1000/new_fps)

    # infer time of each pixel based on frame on- and offset times
    px_per_frame = stack.shape[2] * stack.shape[3]
    pixel_times = np.linspace(frameonset_ms, frameoffset_ms, px_per_frame)
    pixel_times = np.reshape(pixel_times.T, stack.shape)

    new_stack = interpnd(pixel_times, stack, new_times)
    return new_stack, new_times


def reslice_time_and_zpos(stack: np.ndarray, frameonset_ms: List[float], frameoffset_ms: List[float],
                          new_times: Optional[List[float]] = None,
                          new_fps: Optional[float] = None,
                          frames_zpos: Optional[np.ndarray] = None,
                          new_nb_zpos: Optional[int] = None) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reslice time and optionally also zpos

    Args:
        stack (np.ndarray): 4D stack as a time series of 3D volumes [volumes, zpos, xpos, ypos]
                            [xpos, ypos] defines a single frame, [zpos, xpos, ypos] defines a single stack
        frameonset_ms (List[float]): time of the first pixel of each frame, in ms. Should be len (volume * xpos)
        frameoffset_ms (List[float]): time of the last pixel of each frame, in ms. Should be len (volume * xpos)
        new_times (List[float]): List of volume times. Optional.
        new_fps (float): New frame/volume rate. New volumes times will start at midpoints of first and last frame.
                         Optional (ignored if new_times is set)
        frames_zpos (Optional): zpos for each frame and pixel (same shape as stack). If not provided/None, will not do zpos re-slicing. Defaults to None.
        new_nb_zpos (Optional[int]) = None): Number of new z positions - will start at 2-percentile and and at 98-percentile of the original stack values.
                                            Ignored if new_zpos is set.

    Raises:
        ValueError if neither new_times nor new_fps is set.
        ValueError if stack is not 4D.

    Returns:
        np.ndarray: new stack shape [new times, zpos, xpos, ypos]
        np.ndarray: new times start at min(frameonset_ms) and at max(frameoffset_ms)
        np.ndarray: new z pos (Optional)

    """
    if stack.ndim != 4:
        raise ValueError(f"Stack is not 4D (shape is {stack.shape}. Should be [vol,z,x,y].")

    if new_times is None and new_fps is None:
        raise ValueError("Need to provide either new_times or new_fps.")

    # generate `new_times` based on `new_fps` from mid time points of first and last frame.
    if new_times is None and new_fps is not None:
        frameinterval_ms = np.array(frameoffset_ms) - np.array(frameonset_ms)
        frame_mid_points = np.array(frameonset_ms) + frameinterval_ms/2
        new_times = np.arange(
            frame_mid_points[0], frame_mid_points[-1], 1000/new_fps)

    # infer time of each pixel based on frame on- and offset times
    pixel_values = stack
    px_per_frame = pixel_values.shape[2] * pixel_values.shape[3]
    pixel_times = np.linspace(frameonset_ms, frameoffset_ms, px_per_frame)
    pixel_times = np.reshape(pixel_times.T,  pixel_values.shape)

    new_pixel_values = interpnd(pixel_times, pixel_values, new_times)

    if frames_zpos is not None and new_nb_zpos is not None:
        # make linear indices
        new_pixel_zpos_idx = np.ones_like(new_pixel_values)
        for z in range(new_pixel_values.shape[1]):
            new_pixel_zpos_idx[:, z, :, :] = z

        # generate new zpos
        new_zpos_idx = np.linspace(0, new_pixel_values.shape[1], new_nb_zpos)

        new_pixel_values = interpnd(
            new_pixel_zpos_idx, new_pixel_values, new_zpos_idx, axis=1)

        # resample zpos to new time grid
        pixel_zpos = frames_zpos.reshape(pixel_values.shape)
        old_pixel_zpos = interpnd(pixel_times, pixel_zpos, new_times)
        new_zpos_values = interpnd(new_pixel_zpos_idx, old_pixel_zpos, new_zpos_idx, axis=1)
        new_zpos = np.median(new_zpos_values, axis=(0, 2, 3))
        return new_pixel_values, new_times, new_zpos
    else:
        return new_pixel_values, new_times


def interpnd(X: np.ndarray, Y: np.ndarray, new_x: np.ndarray, axis=0) -> np.ndarray:
    """[summary]

    Args:
        X (np.ndarray): [description]
        Y (np.ndarray): [description]
        new_x (np.ndarray): new x values along the specified axis
        axis (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: [description]
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape. X.shape is {X.shape} and Y.shape is {Y.shape}")

    # check the len(new_x)==X.shape[axis]

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

    # # reshape to [new_volumes, zpos, xpos, ypos]
    new_Y = np.reshape(new_yy, (*Y.shape[1:], -1))
    new_Y = new_Y.transpose((3, 0, 1, 2))
    del new_yy
    if axis != 0:
        restored_axes = np.arange(ndim)[np.argsort(new_axes_order)]
        new_Y = new_Y.transpose(restored_axes)
    return new_Y
