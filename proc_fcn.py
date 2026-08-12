"""LFP preprocessing utilities used by the ripple-detection pipeline."""

import numpy as np


def downsample_data (data, sf, d_sf):

    # Dowsampling
    if sf > d_sf:
        downsampled_pts = np.linspace(0, data.shape[0]-1, int(np.round(data.shape[0]/sf*d_sf))).astype(int)
        downsampled_data = data[downsampled_pts, :]

    # Upsampling
    elif sf < d_sf:
        print(f"Original sampling rate below {d_sf} Hz!")
        return None
    
    elif sf==d_sf:
        print("No downsaple is required")
        downsampled_data=data


    # Change from int16 to float16 if necessary
    # int16 ranges from -32,768 to 32,767
    # float16 has ±65,504, with precision up to 0.0000000596046
    if downsampled_data.dtype != 'float16':
        downsampled_data = np.array(downsampled_data, dtype="float16")

    return downsampled_data


def z_score_normalization(data):
    channels = range(np.shape(data)[1])

    for channel in channels:
        # Since data is in float16 type, we make it smaller to avoid overflows
        # and then we restore it.
        # Mean and std use float64 to have enough space
        # Then we convert the data back to float16
        dmax = np.amax(data[:, channel])
        dmin = abs(np.amin(data[:, channel]))
        dabs = dmax if dmax>dmin else dmin
        m = np.mean(data[:, channel] / dmax, dtype='float64') * dmax
        s = np.std(data[:, channel] / dmax, dtype='float64') * dmax
        s = 1 if s == 0 else s # If std == 0, change it to 1, so data-mean = 0
        data[:, channel] = ((data[:, channel] - m) / s).astype('float16')
    
    return data


def generate_overlapping_windows(data, window_size, stride, sf):
    window_pts = int(window_size * sf)
    stride_pts = int(stride * sf)
    r = range(0, data.shape[0], stride_pts)

    new_data = np.empty((len(list(r)), window_pts, data.shape[1]))

    cont = 0
    for idx in r:
        win = data[idx:idx+window_pts, :]

        if (win.shape[0] < window_pts):
            continue

        new_data[cont,:,:]  = win

        cont = cont+1

    return new_data

# Detection functions


def process_LFP(LFP,sf,d_sf,channels):
    
    ''' 
    def process_LFP(LFP,sf,d_sf,channels)

    This function processes the LFP before calling the detection algorithm.
    1. It extracts the desired channels from the original LFP, and interpolates where there is a value of -1.
    2. Downsamples the LFP to d_sf Hz.
    3. Normalizes each channel separately by z-scoring them.

    Mandatory inputs:
        LFP: 		(np.array: n_samples x n_channels) LFP recorded data.
        sf: 		(int) Original sampling frequency (in Hz).
        d_sf:		(int) Desired subsampling frequency (in Hz).
        channels: 	(np.array: n_channels) Indicates which channels will the pre processing be applied to. Counting starts in 0. 
                    If channels contains any -1, interpolation will be also applied. 
                    See channels of rippl_AI.predict(), or aux_fcn.interpolate_channels() for more information.
    Output:
    -------
        LFP_norm: normalized LFP (np.array: n_samples x len(channels)). It is undersampled to d_sf Hz, z-scored, 
                    and transformed to used the channels specified in channels.
    A Rubio, LCN 2023
    '''
    data=interpolate_channels(LFP,channels)
    print(f'Downsampling data from {sf} to {d_sf} Hz...')
    data = downsample_data(data, sf, d_sf)
    print("Shape of downsampled data:",data.shape)

    
    print('Normalizing data...')
    normalized_data=z_score_normalization(data)
    return normalized_data


def interpolate_channels(data, ch_map):
    
    interp_data = np.zeros((data.shape[0], len(ch_map)))
    for idx, ch in enumerate(ch_map):
        if ch>-1:
            interp_data[:,idx] = data[:,ch]
        else:
            pre_ch_idx = np.where(np.array(ch_map[:idx])>-1)[0][-1]
            pre_ch = ch_map[pre_ch_idx]
            post_ch_idx = np.where(np.array(ch_map[idx+1:])>-1)[0][0]+idx+1
            post_ch = ch_map[post_ch_idx]
            ch_dist = post_ch_idx - pre_ch_idx
            interp_data[:,idx] = data[:, pre_ch] + ((idx-pre_ch_idx)/ch_dist) * \
                (data[:, post_ch] - data[:, pre_ch])
    return interp_data

# Retraining auxiliary functions
