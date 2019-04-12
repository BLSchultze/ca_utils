import numpy as np
from scipy.signal import find_peaks
from ScanImageTiffReader import ScanImageTiffReader
import h5py
from glob import glob
import logging
from pprint import pprint
from collections import namedtuple
from scanimagetiffile import ScanImageTiffFile
Trial = namedtuple('Trial', ['file_names', 'frames_first', 'frames_last', 'nb_frames', 'frame_width', 'frame_height', 'nb_channels', 'channel_names', 'frame_rate_hz', 'volume_rate_hz', 'nb_slices', 'frame_zindex'])


def parse_stim_log(logfile_name):
    """Reconstruct playlist from log file.

    Args:
        logfilename
    Returns:
        dict with playlist entries
    """
    with open(logfile_name, 'r') as f:
        logs = f.read()
    log_lines = logs.strip().split('\n')
    session_log = []
    for current_line in log_lines:
        head, _, dict_str = current_line.partition(': ')
        if dict_str.startswith('cnt:'):  # indicates playlist line in log file
            dict_items = dict_str.strip().split('; ')
            dd = dict()
            for dict_item in dict_items:
                key, val = dict_item.strip(';').split(': ')
                val = val.replace('nan', 'np.nan')
                try:
                    dd[key.strip()] = eval(val.strip())
                except (ValueError, NameError):
                    dd[key.strip()] = val.strip()
            session_log.append(dd)
    return session_log


def parse_files(path):
    """Load all tif files in the path.

    Args:
        path
    Returns:
        list of ScanImageTiffFile objects
    """
    recordings = glob(path + '_*.tif')
    recordings.sort()
    logging.info(f'found {len(recordings)} recordings:')
    files = [ScanImageTiffFile(recording) for recording in recordings]
    return files


def parse_trial_files(path):
    """Get file info for each trial

    Args:
        path
    Returns:
        list of Trial objects:
            Trial(file_names, frames_first, frames_last, nb_frames, frame_width, frame_height, nb_channels, channel_names)
    """
    files = parse_files(path)
    # assemble file numbers/names and frame-numbers for each trial
    trial_starttime = np.concatenate([f.description['nextFileMarkerTimestamps_sec'] for f in files])  # start time of the current trial for each frame
    file_index = np.concatenate([f.description['acquisitionNumbers'] for f in files]) - 1  # file number for each frame, -1 for 0-based indexing
    frame_numbers = np.concatenate([f.description['frameNumbers'] for f in files]) - 1  # running number of frames in session, -1 for 0-based indexing
    trial_uni, trial_index = np.unique(trial_starttime, return_inverse=True)
    nb_trials = len(trial_uni)

    file_onsets = np.where(np.diff(file_index) > 0)[0].astype(np.uintp)+1  # plus 1 since we want the first frame *after* the change
    file_onsets = np.pad(file_onsets, (1, 0), mode='constant', constant_values=(0, len(file_index)))  # append first frame as first file onset
    file_offsets = np.pad(file_onsets[1:], (0, 1), mode='constant', constant_values=(0, len(file_index)))  # append last frame as last file onset

    # probably don't need this if we only care about the first and last frame for that trial from each file
    frame_index = np.zeros((len(file_index,)), dtype=np.uintp)  # within trial frame number
    for onset, offset in zip(file_onsets, file_offsets):
        frame_index[onset:offset] = np.arange(0, offset-onset)

    trials = []
    # add info about which frame from which files belong to that trial
    for trial_number in range(nb_trials):
        idx = np.where(trial_index == trial_number)[0]  # get global frame numbers for current trial
        frm = frame_index[idx]  # get within-file frame numbers for current trial
        uni_files = np.unique(file_index[idx])  # which files contribute to the current trial

        file_names = [files[ii].name for ii in uni_files]
        framenumbers = [frm[file_index[idx] == ii] for ii in uni_files]
        frames_first = [f[0] for f in framenumbers]
        frames_last = [f[-1] + 1 for f in framenumbers]  # +1 since we we use it as a range (exclusive bounds), otherwise we would miss the last frame
        nb_frames = sum([int(last - first) for first, last in zip(frames_first, frames_last)])
        # add some metadata to the trial info from the first file in each trial
        frame_width = int(files[uni_files[0]].metadata['hRoiManager.linesPerFrame'])
        frame_height = int(files[uni_files[0]].metadata['hRoiManager.linesPerFrame'])
        frame_rate_hz = files[uni_files[0]].metadata['hRoiManager.scanFrameRate']
        volume_rate_hz = files[uni_files[0]].metadata['hRoiManager.scanVolumeRate']
        nb_channels = int(len(files[uni_files[0]].metadata['hChannels.channelSave']))
        channel_names = ['gcamp', 'tdtomato'][:nb_channels]
        nb_slices = files[uni_files[0]].metadata['hStackManager.numSlices']
        frame_zindex = np.mod(np.array(frame_numbers[idx[::nb_channels]]), nb_slices)
        trials.append(Trial(file_names, frames_first, frames_last, nb_frames, frame_width, frame_height, nb_channels, channel_names, frame_rate_hz, volume_rate_hz, nb_slices, frame_zindex))
    return trials


def parse_daq(ypos, zpos, next_trigger, sound=None):
    """Get timing of frames and trials from ca recording.d_ypos.

    Args:
        ypos - y-position of the pixel scanner (resets indicate frame onsets)
        zpos - z-position of the piezo scanner
        next_trigger - recording of the next file trigger from scanimage to partition trials
        sound (optional) - sound recording (requires single channel - sum over left and right channel)

    Returns dict with the following keys:
        frame_onset_samples - sample number for the onset of each frame (inferred from the y-pos reset)
        frame_offset_samples - sample number for the offset of each frame (inferred from the y-pos reset)
        trial_onset_samples - onset of each trial (inferred from peaks in the next_trigger signals)
        trial_offset_samples - offset of each trial (last sample number added as offset for final trial)
        sound_onset_samples - onset of first sound event in the trial (None if no sound provided)
        sound_offset_samples - offset of last sound event in the trial (None if no sound provided)
        frame_avgzpos - average zpos for each frame
    """
    d_ypos = -np.diff(ypos)
    frame_offset_samples, _ = find_peaks(d_ypos, height=np.max(d_ypos) / 2)  # samples at which each frame has been stopped being acquired (y pos resets)
    frame_onset_samples = np.empty_like(frame_offset_samples)
    frame_onset_samples[1:] = frame_offset_samples[:-1] + 1  # samples after frame offset
    tmp = find_peaks(-d_ypos[:frame_onset_samples[1]], height=np.max(-d_ypos) / 2)  #  detect first frame onset
    frame_onset_samples[0] = tmp[0] - 1
    d_nt = np.diff(next_trigger)
    trial_onset_samples, _ = find_peaks(d_nt, height=np.max(d_nt) / 2)

    # from these construct trial offset samples:
    trial_offset_samples = np.append(trial_onset_samples[1:], len(next_trigger))  # add last sample as final offset

    # detect sound onsets and offset from DAQ recording of the sound
    sound_onset_samples = None
    sound_offset_samples = None
    if sound is not None:
        # use these to infer within trial sound onset
        sound_onset_samples = np.zeros((len(trial_onset_samples),), dtype=np.uintp)
        sound_offset_samples = np.zeros((len(trial_onset_samples),), dtype=np.uintp)
        for cnt, (trial_start_sample, trial_end_sample) in enumerate(zip(trial_onset_samples, trial_offset_samples)):
            trial_sound = sound[trial_start_sample:trial_end_sample]
            thres = np.max(trial_sound) / 10
            sound_onset_samples[cnt] = int(trial_start_sample + np.argmax(trial_sound >= thres))

            trial_sound = sound[sound_onset_samples[cnt]:trial_end_sample]
            sound_offset_samples[cnt] = int(sound_onset_samples[cnt] + len(trial_sound) - 1 - np.argmax(trial_sound[::-1] >= thres))

    # get avg z-pos for each frame
    frame_avgzpos = np.zeros_like(frame_onset_samples, dtype=np.float)
    for cnt, (on, off) in enumerate(zip(frame_onset_samples, frame_offset_samples)):
        frame_avgzpos[cnt] = np.mean(zpos[on:off])

    d = dict()
    d['frame_offset_samples'] = frame_offset_samples
    d['frame_onset_samples'] = frame_onset_samples
    d['trial_onset_samples'] = trial_onset_samples
    d['trial_offset_samples'] = trial_offset_samples
    d['sound_onset_samples'] = sound_onset_samples
    d['sound_offset_samples'] = sound_offset_samples
    d['frame_avgzpos'] = frame_avgzpos
    return d


def samples2frames(frame_samples, samples):
    """Get next frame after sample.

    Args:
        frame_samples: sample numbers of the frames
        samples: single sample number or list of samples
    Returns:
        list of frame at or after samples
    """
    try:
        len(samples)
    except TypeError:
        samples = np.array([samples, ], dtype=np.uintp)
    frames = np.zeros_like(samples)
    frame_samples = np.append(frame_samples, np.max(samples) + 1)  # why +1?

    for cnt, sample in enumerate(samples):
        frames[cnt] = np.argmax(frame_samples > sample)
    return frames


def find_nearest(arr, val):
    """Find index of occurrence of item in arr closest to val."""
    return (np.abs(np.array(arr) - val)).argmin()


def parse_trial_timing(daq_file_name):
    """Parse DAQ file to get frame precise timestamps and sound onset/offset information."""
    # parse DAQ data for synchronization
    with h5py.File(daq_file_name, 'r') as f:
        data = f['samples']
        ypos = data[:, 0]  # y pos of the scan pixel
        zpos = data[:, 1]  # z pos of the scan pixel
        next_trigger = data[:, -2]
        sound = np.sum(data[:, 3:5], axis=1)  # pool left and right channel
        try:
            fs = f.attrs['rate']
            logging.info(f'using saved sampling rate of: {fs}Hz')
        except KeyError:
            fs = 10_000
            logging.warning(f'sampling rate of recording not found in DAQ file - defaulting to {fs}Hz')
    d = parse_daq(ypos, zpos, next_trigger, sound)

    # get frame times for each trial
    nb_trials = len(d['trial_onset_samples'])
    Log = namedtuple('Log', ['frametimes_ms', 'frame_avgzpos', 'stimonset_ms', 'stimoffset_ms', 'stimonset_frame', 'stimoffset_frame'])
    logs = [None]*nb_trials
    for cnt in range(nb_trials):
        d['trial_offset_samples'][cnt] = d['trial_offset_samples'][cnt]
        d['trial_onset_samples'][cnt] = d['trial_onset_samples'][cnt]

        trial_onset_frame = samples2frames(d['frame_offset_samples'], d['trial_onset_samples'][cnt])[0]
        trial_offset_frame = samples2frames(d['frame_offset_samples'], d['trial_offset_samples'][cnt])[0]
        frametimes = (d['frame_offset_samples'][trial_onset_frame:trial_offset_frame] - d['trial_onset_samples'][cnt])/fs*1000
        frametimes_ms = frametimes.tolist()
        frame_avgzpos = d['frame_avgzpos'][trial_onset_frame:trial_offset_frame]
        stimonset_ms = float((d['sound_onset_samples'][cnt] - d['trial_onset_samples'][cnt])/fs*1000)
        stimoffset_ms = float((d['sound_offset_samples'][cnt] - d['trial_onset_samples'][cnt])/fs*1000)
        stimonset_frame = find_nearest(frametimes_ms, stimonset_ms)
        stimoffset_frame = find_nearest(frametimes_ms, stimoffset_ms)
        logs[cnt] = Log(frametimes_ms, frame_avgzpos, stimonset_ms, stimoffset_ms, stimonset_frame, stimoffset_frame)
    return logs
