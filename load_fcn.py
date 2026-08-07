"""Data-loading utilities for ripple-AI.

These functions read saved objects, recording metadata, ripple annotations,
channel maps, and raw LFP data from disk.
"""

import math
import os
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io

def fcn_load_pickle(name):
    '''
    [x] = fcn_load_pickle(name) loads the content of the pickle file to x
    '''
    with open(name, 'rb') as handle:
            return( pickle.load(handle) )


# Estas dos funciones están normalmente en bz_load_binary, se pueden mover allá


def loadChunk(fid, nChannels, channels, nSamples, precision):
    size = int(nChannels * nSamples * precision)
    nSamples = int(nSamples)

    data = fid.read(size)

    # fromstring to read the data as int16
    # reshape to give it the appropiate shape (nSamples x nChannels)
    data = np.fromstring(data, dtype=np.int16).reshape(nSamples, len(channels))
    data = data[:, channels]

    return data


def bz_LoadBinary(filename, nChannels, channels, sampleSize, verbose=False):

    if (len(channels) > nChannels):
        print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
        return

    #aqui iria CdE de filename
    with open(filename, "rb") as f:
        dataOffset = 0

        # Determine total number of samples in file
        fileStart = f.tell()
        if verbose:
            print("fileStart ", fileStart)
        status = f.seek(0, 2) # Go to the end of the file
        fileStop = f.tell()
        f.seek(0, 0) # Back to the begining
        if verbose:
            print("fileStop ", fileStop)

        # (floor in case all channels do not have the same number of samples)
        maxNSamplesPerChannel = math.floor(((fileStop-fileStart)/nChannels/sampleSize))
        nSamplesPerChannel = maxNSamplesPerChannel

        # For large amounts of data, read chunk by chunk
        maxSamplesPerChunk = 10000
        nSamples = int(nSamplesPerChannel*nChannels)

        if verbose:
            print("nSamples ", nSamples)

        if nSamples <= maxNSamplesPerChannel:
            data = loadChunk(f, nChannels, channels, nSamples, sampleSize)
        else:
            # Determine chunk duration and number of chunks
            nSamplesPerChunk = math.floor(maxSamplesPerChunk/nChannels)*nChannels
            nChunks = math.floor(nSamples/nSamplesPerChunk)

            if verbose:
                print("nSamplesPerChannel ", nSamplesPerChannel)
                print("nSamplesPerChunk ", nSamplesPerChunk)

            # Preallocate memory
            data = np.zeros((nSamplesPerChannel,len(channels)), dtype=np.int16)

            if verbose:
                print("size data ", np.size(data, 0))

            # Read all chuncks
            i = 0
            for j in range(nChunks):
                d = loadChunk(f, nChannels, channels, nSamplesPerChunk/nChannels, sampleSize)
                m = np.size(d, 0)

                if m == 0:
                    break

                data[i:i+m, :] = d
                i = i+m

            # If the data size is not a multiple of the chunk size, read the remainder
            remainder = nSamples - nChunks*nSamplesPerChunk
            if remainder != 0:
                d = loadChunk(f, nChannels, channels, remainder/nChannels, sampleSize)
                m = np.size(d, 0)

                if m != 0:
                    data[i:i+m, :] = d

    return data

# Functions used to load the raw LFP, select channels, load ripples, downsample and normalize


def load_lab_data(path):
    sf, expName, ref_channels, dead_channels = load_info(path)
    channels_map = load_channels_map(path)
    ripples=load_ripples(path)/sf
    channels, shanks, ref_channels = reformat_channels(channels_map, ref_channels)
    LFP = load_raw_data(path, expName, channels, verbose=True)
    return(LFP,ripples)


def load_info (path):
    try:
        mat = scipy.io.loadmat(os.path.join(path, "info.mat"))
    except:
        print("info.mat file does not exist.")
        sys.exit()

    sf = mat["fs"][0][0]
    expName = mat["expName"][0]

    ref_channels = {}
    ref_channels["so"] = mat["so"][0]
    ref_channels["pyr"] = mat["pyr"][0]
    ref_channels["rad"] = mat["rad"][0]
    ref_channels["slm"] = mat["slm"][0]


    if len(mat["chDead"]) <= 0:
        dead_channels = []
    else:
        dead_channels = [x-1 for x in (mat["chDead"][0]).astype(int)]

    return sf, expName, ref_channels, dead_channels


def load_ripples (path, verbose=False):
    try:
        dataset = pd.read_csv(os.path.join(path,"ripples.csv"), delimiter=' ', header=0, usecols = ["ripIni","ripEnd"])# "ripMiddle", "ripEnd", "type", "shank"])
    except:
        print(path+"ripples.csv file does not exist.")
        sys.exit()

    ripples = dataset.values
    ripples = ripples[np.argsort(ripples, axis=0)[:, 0], :]
    if verbose:
        print("Loaded ripples: ", len(ripples))

    return ripples


def load_channels_map (path):
    try:
        dataset = pd.read_csv(path+"/mapsCh.csv", delimiter=' ', header=0)
    except:
        print("ripples.csv file does not exist.")
        sys.exit()

    channels_map = dataset.values

    return channels_map


def reformat_channels (channels_map, ref_channels):
    channels = np.where(np.isnan(channels_map[:, 0]) == False, channels_map[:, 0], 0)
    channels = [x-1 for x in (channels).astype(int)]

    shanks = np.where(np.isnan(channels_map[:, 1]) == False, channels_map[:, 1], 0)
    shanks = [x-1 for x in (shanks).astype(int)]

    ref_channels["so"] = np.where(np.isnan(ref_channels["so"]) == False, ref_channels["so"], 0)
    ref_channels["so"] = [x-1 for x in ref_channels["so"].astype(int)]
    ref_channels["pyr"] = np.where(np.isnan(ref_channels["pyr"]) == False, ref_channels["pyr"], 0)
    ref_channels["pyr"] = [x-1 for x in ref_channels["pyr"].astype(int)]
    ref_channels["rad"] = np.where(np.isnan(ref_channels["rad"]) == False, ref_channels["rad"], 0)
    ref_channels["rad"] = [x-1 for x in ref_channels["rad"].astype(int)]
    ref_channels["slm"] = np.where(np.isnan(ref_channels["slm"]) == False, ref_channels["slm"], 0)
    ref_channels["slm"] = [x-1 for x in ref_channels["slm"].astype(int)]

    return channels, shanks, ref_channels


def load_raw_data (path, expName, channels, verbose=False):
    
    # There is .dat file
    is_dat = any([file.endswith(".dat") for file in os.listdir(path)])

    # There is .eeg file
    is_eeg = any([file.endswith(".eeg") for file in os.listdir(path)])
    
    # There is .mat file with the name of the last folder
    is_mat = any([os.path.basename(os.path.normpath(path))+".mat" in file for file in os.listdir(path)])

    if is_dat:
        name_dat = os.listdir(path)[np.where([file.endswith(".dat") for file in os.listdir(path)])[0][0]]
        if verbose:
            print(path+"/"+name_dat)
        data = bz_LoadBinary(path+"/"+name_dat, len(channels), channels, 2, verbose)

    elif is_eeg:
        name_eeg = os.listdir(path)[np.where([file.endswith(".eeg") for file in os.listdir(path)])[0][0]]
        if verbose:
            print(path+"/"+name_eeg)
        data = bz_LoadBinary(path+"/"+name_eeg, len(channels), channels, 2, verbose)

    elif is_mat:
        folder = path + "/" + os.path.basename(os.path.normpath(path))+".mat"
        if verbose:
            print(folder)
        try:
            mat = scipy.io.loadmat(folder)
            data = mat["fil"]
        except:
            mat = h5py.File(folder, 'r')
            data = np.array(mat["fil"]).T
    else:
        print('Not data found')

    return data


def load_data_fs(path, shank, verbose=False):
    # Read info.mat
    sf, expName, ref_channels, dead_channels = load_info(path)

    #Read mapsCh.csv
    channels_map = load_channels_map(path)

    # Reformat channels into correct values
    channels, shanks, ref_channels = reformat_channels(channels_map, ref_channels)
    # Read .dat
    data = load_raw_data(path, expName, channels, verbose=verbose)


    return data, sf
