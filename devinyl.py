import os
import logging
import math
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import librosa
from pysndfx import AudioEffectsChain
import numpy as np
import matchering as mg
from matchering.core import Config
from matchering.results import Result
import python_speech_features
import soundfile as sf
import scipy as sp
from tqdm import tqdm

# Thanks to all of these projects and pages for making this possible.

# Main work on noise reduction with librosa: https://github.com/dodiku/noise_reduction/blob/master/noise.py
# http://python-speech-features.readthedocs.io/en/latest/
# https://github.com/jameslyons/python_speech_features
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#deltas-and-delta-deltas
# http://dsp.stackexchange.com/search?q=noise+reduction/

'''------------------------------------
FILE READER:
    receives filename,
    returns audio time series (y) and sampling rate of y (sr)
------------------------------------'''
def read_file(file_name):
    sample_path = file_name

    # generating audio time series and a sampling rate (int)
    y, sr = librosa.load(os.path.abspath(sample_path), mono=False)

    return y, sr

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

'''------------------------------------
OUTPUT GENERATOR:
    receives a destination path, file name, audio matrix, and sample rate,
    generates a wav file based on input
------------------------------------'''
def output_file(idx, destination ,filename, y, sr, reference_file, ext=""):
    destination = os.path.join(os.path.abspath(destination), Path(filename[:-4] + ext + '.wav').name)
    print('DESTINATION=', destination)
    print('SHAPE=', y.shape, y.dtype)
    with sf.SoundFile(destination, 'w', sr, 2, 'FLOAT') as f:
        f.write(np.hstack((y[0].reshape(-1, 1), y[1].reshape(-1,1))))
    postprocess(destination, 44100, reference_file)
    postclean(idx, destination)


def postprocess(source_file, sample_rate, reference_file):
    results = mg.core.process(
        target=source_file,
        reference=reference_file,
        results = [Result(source_file.replace('.wav', '.processed.wav'), subtype='FLOAT', use_limiter=True, normalize=False)],
        config=Config(internal_sample_rate=sample_rate)
    )


def postclean(idx, source_file):
    y, sr = read_file(source_file.replace('.wav', '.processed.wav'))
    filtered_y = random.choice(filters)(y, sr)
    with sf.SoundFile(source_file.replace('.wav', '.processed_filtered.wav'), 'w', sr, 2, 'FLOAT') as f:
        f.write(np.hstack((filtered_y[0].reshape(-1, 1), filtered_y[1].reshape(-1,1))))


def parse_args():
    parser = ArgumentParser(
        description="Simple Matchering 2.0 Command Line Application"
    )
    parser.add_argument("target", type=str, help="The track you want to master")
    parser.add_argument("reference", type=str, help='Some "wet" reference track')
    # parser.add_argument("result", type=str, help="Where to save your result")
    parser.add_argument(
        "-b",
        "--bit",
        type=int,
        choices=[16, 24, 32],
        default=16,
        help="The bit depth of your mastered result. 32 means 32-bit float",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="The file to which the logs will be written",
    )
    parser.add_argument(
        "--no_limiter",
        dest="no_limiter",
        action="store_true",
        help="Disables the limiter at the final stage of processing",
    )
    parser.add_argument(
        "--dont_normalize",
        dest="dont_normalize",
        action="store_true",
        help="Disables normalization, if --no_limiter is set. "
        "Can cause clipping if the bit depth is not 32",
    )
    return parser.parse_args()


def set_logger(handler, formatter, logger):
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def prepare_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("{asctime}: {levelname:>7}: {message}", style="{")

    if args.log:
        set_logger(logging.FileHandler(args.log), formatter, logger)

    set_logger(logging.StreamHandler(sys.stdout), formatter, logger)

    return args, logger

pre_filters = [reduce_noise_median for _ in range(7)]

filters = [reduce_noise_median for _ in range(7)]

# filters = [
#     reduce_noise_power,
#     reduce_noise_power,
#     reduce_noise_centroid_mb,
#     reduce_noise_mfcc_up,
#     reduce_noise_mfcc_down,
#     reduce_noise_median,
#     reduce_noise_no_reduction
# ]


def run(args, logger):
    logger.info('##############################################')
    logger.info('# DEVINYL - Recover vynils beyond recovering #')
    logger.info('##############################################')

    mg.log(
        warning_handler=logger.warning,
        info_handler=logger.info,
        debug_handler=logger.debug,
    )

    # reading a file
    y, sr = read_file(args.target)

    filtered_y_list = tqdm([filter(y, sr) for filter in pre_filters])

    # trimming silences
    trimmed_y_list = tqdm([trim_silence(to_trim)[0] for to_trim in filtered_y_list])
    
    suffixes = [
        '_pwr',
        '_ctr_s',
        '_ctr_mb',
        '_mfcc_up',
        '_mfcc_down',
        '_median',
        '_org'
    ]
    
    # generating output files
    for i in range(len(filters[:1])):
        output_file(i, './01_samples_trimmed_noise_reduced/', args.target, trimmed_y_list[i], sr, args.reference, suffixes[i])


if __name__ == "__main__":
    run(*prepare_logger(parse_args()))