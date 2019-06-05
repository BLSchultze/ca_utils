"""Draw rois.

TODO:
[x] modularize
[x] vectorize trace extraction
[x] make ROIs stacks of same dim as imaging volume
[ ] automate plotting
[ ] offer to append or overwrite existing annotation
"""
import matplotlib
matplotlib.use('TKAgg', warn=False, force=True)  # changing backend of matplotlib for pyplot to work
import matplotlib.pyplot as plt
import numpy as np
import ca_utils.io as ca
# from ca_utils.roi.draw import roipoly
from .roipoly import roipoly
import deepdish as dd
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import itertools
import os
import defopt


def plot_rois(background, layer, rois=None):
    plt.imshow(background[layer, ...])
    if rois is not None and len(rois) > 0:  # overlay existing ROIs
        roi = np.sum(np.stack(rois, axis=-1), axis=-1)
        plt.imshow(roi[layer, ...], alpha=0.2)


def smooth_planes(stack, sigma=1.5, px_axes=(2, 3)):
    # iterate over all non-px axes and smooth
    dims = [d for d in list(range(stack.ndims)) if d not in px_axes]  # iterate over those to get indicidual images
    for whatever in whatevers:
        whatever = gaussian_filter(whatever, sigma)
    return whatever

def get_mean_img(stack, smooth_sigma=None):
    # prepare img to draw ROIs in
    mean_img = np.mean(stack, axis=0)
    if smooth_sigma is not None:
        for cnt in range(nb_layers):
            mean_img[cnt, ...] = gaussian_filter(mean_img[cnt, ...], smooth_sigma)
    return mean_img


def draw(mean_img):
    maskROI = []  # init empty list of masks
    nb_layers = mean_img.shape[0]
    layer = 0  # initial layer to draw ROIs in
    # DRAW ROIs
    while True:
        key_press = input(f"press number (0..{nb_layers-1}) to select layer, any key to draw another ROI in layer {layer}, or q to quit and save your ROIs: ")
        if key_press == 'q':  # stop drawing ROIs upon 'q' press
            break
        else:
            try:
                new_layer = int(key_press)
                if layer != new_layer and new_layer < nb_layers:
                    layer = new_layer
            except ValueError:
                print(f"   keeping layer {layer}")

            plot_rois(mean_img, layer, maskROI)

            # draw rois with mouse
            this_roi = np.zeros_like(mean_img, dtype=np.bool)  # init empty ROI stack
            ROI = roipoly(roicolor='r')  # init plot window
            this_roi[layer, ...] = ROI.getMask(mean_img[layer, ...])  # set mask for current layer
            maskROI.append(this_roi)
    return maskROI


# EXTRACT TRACES
def extract_traces(stacks, masks):
    """
    Args:
        stacks - list of M stacks (dims T,V,X,Y,C)
        masks - list of N masks (dims V,X,Y)
    Returns:
        list of N traces M
    """
    masks = np.stack(maskROI, axis=-1)
    traces = []
    nb_layers = masks.shape[0]
    nb_masks = masks.shape[-1]
    for stack in tqdm(stacks):
        nb_times = stack.shape[0]
        nb_channels = stack.shape[-1]
        trace = np.zeros((nb_times, nb_channels, nb_masks))

        for msk in range(nb_masks):
            for chan, layer in itertools.product(range(nb_channels), range(nb_layers)):
                trace[:, chan, msk] += np.mean(stack[:, layer, ..., chan] * masks[layer, ..., msk], axis=(-2, -1))
        traces.append(trace)
    return traces

def main(root: str, datename: str, session: int):
    smooth_sigma = 1.5

    cur_rec = f'{datename}_{session}'
    filename = f'{root}/dat/{datename}/{cur_rec}'
    savename = f'{root}/res/{cur_rec}_rois.h5'

    print(f'reading metadata for {cur_rec}')
    s = ca.Session(filename)
    print(f'   loading {s.nb_trials} stacks')
    stacks = [s.stack(trial, split_channels=True, split_volumes=True) for trial in tqdm(range(s.nb_trials))]

    gCaMP_channel = 0
    gCaMPdata = stacks[0][..., gCaMP_channel]
    mean_img = get_mean_img(gCaMPdata)
    maskROI = draw(mean_img)
    print(f'   extracting fluorescence traces for all {s.nb_trials} trials:')
    traces = extract_traces(stacks, maskROI)

    # SAVE ROIs and TRACES
    print(f'saving ROIs and traces to {savename}.')
    key_press = 'y'
    if os.path.exists(savename):
        key_press = input(f"{savename} exists - overwrite [y]/n?")
    if key_press != 'n'
        dd.io.save(savename, {'ROIs': maskROI, 'traces': traces, 'background': mean_img})


if __name__ == "__main__":
    defopt.run(main)
    # change filename and layer (of z-stack)
    root = '/Volumes/ukme04/#Common/2P'  # JAN
    # root = '/home/local/UG-STUDENT/elsa.steinfath/ukme04/#Common/2P'  # ELSA
    # root = 'Z:/#Common/2P'
    datename = '20190417'
    session = '022'
