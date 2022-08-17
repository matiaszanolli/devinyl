import math

import librosa
import numpy as np
from pysndfx import AudioEffectsChain
import python_speech_features
import scipy as sp


'''------------------------------------
NOISE REDUCTION USING POWER:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''
def reduce_noise_power(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.4
    threshold_l = round(np.median(cent))*0.2

    less_noise = AudioEffectsChain().lowshelf(gain=-34.0, frequency=threshold_l, slope=0.7).highshelf(gain=-16.0, frequency=threshold_h, slope=0.6).limiter(gain=8.0)
    y_clean = less_noise(y)

    return y_clean


'''------------------------------------
NOISE REDUCTION USING CENTROID ANALYSIS:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''

def reduce_noise_centroid_s(y, sr):
    
    s = np.abs(y)

    cent = librosa.feature.spectral_centroid(y=s, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5).limiter(gain=6.0)

    y_cleaned = less_noise(y)

    return y_cleaned

def reduce_noise_centroid_mb(y, sr):
    y = np.abs(y)
    
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5).highshelf(gain=-30.0, frequency=threshold_h, slope=0.5).limiter(gain=10.0)
    less_noise = AudioEffectsChain().lowpass(frequency=threshold_h).highpass(frequency=threshold_l)
    y_cleaned = less_noise(y)


    cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
    columns, channels, rows = cent_cleaned.shape
    boost_h = math.floor(rows/3*2)
    boost_l = math.floor(rows/6)
    boost = math.floor(rows/3)

    # boost_bass = AudioEffectsChain().lowshelf(gain=20.0, frequency=boost, slope=0.8)
    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5).lowshelf(gain=-20.0, frequency=boost_l, slope=0.8)
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted


'''------------------------------------
NOISE REDUCTION USING MFCC:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''
def reduce_noise_mfcc_down(y, sr):

    hop_length = 512

    ## librosa
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, reference, sr2, hop_length=hop_length, n_mfcc=13)
    # librosa.mel_to_hz(mfcc)

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.6).limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

def reduce_noise_mfcc_up(y, sr):

    hop_length = 512

    ## librosa
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, reference, sr2, hop_length=hop_length, n_mfcc=13)
    # librosa.mel_to_hz(mfcc)

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum_of_squares = []
    index = -1
    for r in mfcc:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = python_speech_features.base.mel2hz(mfcc[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5).highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.5)#.limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

'''------------------------------------
NOISE REDUCTION USING MEDIAN:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''

def reduce_noise_median(y, sr):
    y = sp.signal.medfilt(y,3)
    return (y)


def reduce_noise_no_reduction(y, sr):
    return (y)


'''------------------------------------
SILENCE TRIMMER:
    receives an audio matrix,
    returns an audio matrix with less silence and the amout of time that was trimmed
------------------------------------'''
def trim_silence(y):
    y_trimmed, index = librosa.effects.trim(y=y, top_db=20, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y=y) - librosa.get_duration(y=y_trimmed)

    return y_trimmed, trimmed_length


'''------------------------------------
AUDIO ENHANCER:
    receives an audio matrix,
    returns the same matrix after audio manipulation
------------------------------------'''
def enhance(y):
    apply_audio_effects = AudioEffectsChain().lowshelf(gain=10.0, frequency=260, slope=0.1).reverb(reverberance=25, hf_damping=5, room_scale=5, stereo_depth=50, pre_delay=20, wet_gain=0, wet_only=False)#.normalize()
    y_enhanced = apply_audio_effects(y)

    return y_enhanced