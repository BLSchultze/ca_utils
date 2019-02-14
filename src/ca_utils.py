import numpy as np
from scipy.signal import find_peaks
from ScanImageTiffReader import ScanImageTiffReader
import h5py
from glob import glob
import logging

def parse_metadata(file):
    """Parses the meta data embedded in tif-file to a dict.

    Args:
        file - ScanImageTiffReader object
    Returns:
        metadata - dict
    """

    meta_lines = file.metadata().strip().split('\n')
    metadata = dict()
    for meta_line in meta_lines:
        try:
            key, val = meta_line.split(' = ')
            key = key.split('.')[-1]
            try:
                metadata[key] = eval(val)
            except (ValueError, NameError):
                metadata[key] = val
        except:
            pass
    return metadata


def parse_session_log(logfile_name):
    """Reconstructs playlist from log file.

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

        # process head
        #     timestamp, hostname = head.split(' ')
        #     timestamp = datetime.strptime(timestamp, '%Y-%m-%d,%H:%M:%S.%f')

        if dict_str[:4] == 'cnt:':
            dict_items = dict_str.strip().split('; ')
            dd = dict()
            for dict_item in dict_items:
                key, val = dict_item.strip(';').split(': ')
                try:
                    dd[key.strip()] = eval(val.strip())
                except (ValueError, NameError):
                    dd[key.strip()] = val.strip()
            session_log.append(dd)
    return session_log


def parse_daq(ypos, next_trigger, sound=None):
    """Get timing of frames and trials from ca recording.d_ypos

    Args:
        ypos - ypose of the pixel scanner (resets indicate frame onsets
        next_trigger - recording of the next file trigger from scanimage
        sound (optional) - sound recording (requires single channel - sum over left and right channel)

    Returns dict with the following keys:
        frame_onset_samples - sample number for the onset of each frame (inferred from the y-pos reset)
        trial_onset_samples - onset of each trial (inferred from peaks in the next_trigger signals)
        trial_offset_samples - offset of each trial (last sample number added as offset for final trial)
        sound_onset_samples - onset of first sound event in the trial (None if no sound provided)
        sound_offset_samples - offset of last sound event in the trial (None if no sound provided)
    """
    d_ypos = -np.diff(ypos)
    frame_onset_samples, _ = find_peaks(d_ypos, height=np.max(d_ypos) / 2)  # these are the samples at which each frame begins (y pos resets)

    d_nt = np.diff(next_trigger)
    trial_onset_samples, _ = find_peaks(d_nt, height=np.max(d_nt) / 2)

    # from these construct trial offset samples:
    trial_offset_samples = np.append(trial_onset_samples[1:], len(next_trigger))  # add last sample as final offset

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
    d = dict()
    d['frame_onset_samples'] = frame_onset_samples
    d['trial_onset_samples'] = trial_onset_samples
    d['trial_offset_samples'] = trial_offset_samples
    d['sound_onset_samples'] = sound_onset_samples
    d['sound_offset_samples'] = sound_offset_samples
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
    except:
        samples = np.array([samples, ], dtype=np.uintp)
    frames = np.zeros_like(samples) + len(frame_samples)
    frame_samples = np.append(frame_samples, np.max(samples) + 1)

    for cnt, sample in enumerate(samples):
        frames[cnt] = np.argmax(frame_samples >= sample)
    return frames


def find_nearest(arr, val):
    return (np.abs(np.array(arr) - val)).argmin()


def parse_session(root, expt_id, fs=10000):
    """
    Args:
        root - dir containing the files for that recording
        expt_id - experiment ID (datename_session, e.g. 20190122_001)
        fs=10000 - sampling rate in Hz, defaults to 10000Hz
    Returns:
        list of dicts, one for each tif file
    """
    # fs should be in the log or a daq file attr!!
    path = root + expt_id
    # global metadata
    recordings = glob(path + '_*.tif')
    recordings.sort()
    logging.info(f'found {len(recordings)} recordings:')

    # parse stim log
    logfile_name = path + '_daq.log'
    session_log = parse_session_log(logfile_name)

    # parse DAQ data for synchronization
    with h5py.File(path + '_daq.h5', 'r') as f:
        data = f['samples']
        ypos = data[:,0]  # y pos of the scan pixel
        next_trigger = data[:,-2]
        sound = np.sum(data[:,3:5], axis=1)  # pool left and right channel
        try:
            fs = f.attrs['rate']
            logging.info(f'using saved sampling rate of: {fs}Hz')
        except KeyError:
            logging.info(f'sampling rate of recording not saved - defaulting to {fs}Hz')
        d = parse_daq(ypos, next_trigger, sound)

    md_list = ['linePeriod', 'linesPerFrame', 'pixelsPerLine', 'scanFrameRate',
               'scanVolumeRate', 'framesPerSlice', 'numSlices', 'stackZStepSize']

    # get frame times for each trial
    for cnt in range(len(recordings)):
        idx = cnt
        file = ScanImageTiffReader(recordings[idx])
        metadata = parse_metadata(file)

        session_log[cnt]['tif_filename'] = recordings[idx]
        for md in md_list:
            session_log[cnt][md] = metadata[md]

        trial_onset_frame = samples2frames(d['frame_onset_samples'], d['trial_onset_samples'][cnt])[0]
        trial_offset_frame = samples2frames(d['frame_onset_samples'], d['trial_offset_samples'][cnt])[0]
        frame_range = range(trial_onset_frame, trial_offset_frame)
        frame_times = (d['frame_onset_samples'][frame_range] - d['trial_onset_samples'][cnt])/fs*1000
        session_log[cnt]['time_ms'] = frame_times.tolist()

        session_log[cnt]['sound_onset_ms'] = float((d['sound_onset_samples'][cnt] - d['trial_onset_samples'][cnt])/fs*1000)
        session_log[cnt]['sound_offset_ms'] = float((d['sound_offset_samples'][cnt] - d['trial_onset_samples'][cnt])/fs*1000)

        session_log[cnt]['sound_onset_frame'] = find_nearest(session_log[cnt]['time_ms'], session_log[0]['sound_onset_ms'])
        session_log[cnt]['sound_offset_frame'] = find_nearest(session_log[cnt]['time_ms'], session_log[0]['sound_offset_ms'])

        session_log[cnt]['sound_onset_ms_from_playlist'] = float(min(session_log[cnt]['silencePre']))
        session_log[cnt]['sound_offset_ms_from_playlist'] = float(d['trial_offset_samples'][cnt]/fs*1000 - min(session_log[cnt]['silencePost']) - d['trial_onset_samples'][cnt]/fs*1000)

        # "annotate" frames:
        # frames alternate between GCAMP and TDTOMATO
        # M slices, N volumes

    return session_log
