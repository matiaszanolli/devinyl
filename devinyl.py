import os
import logging
import random
import sys
from argparse import ArgumentParser
from os.path import exists
from pathlib import Path
from urllib.parse import urlparse


import librosa
import numpy as np
import matchering as mg
import requests
import soundfile as sf
from matchering.core import Config
from matchering.results import Result

from effects import (reduce_noise_centroid_mb, reduce_noise_centroid_s, reduce_noise_median, reduce_noise_mfcc_down, 
                     reduce_noise_mfcc_up, reduce_noise_no_reduction, reduce_noise_power, trim_silence)

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
def read_file(file_path):

    def is_local(url):
        url_parsed = urlparse(url)
        if url_parsed.scheme in ('file', ''): # Possibly a local file
            return exists(url_parsed.path)
        return False
    
    if not is_local(file_path):
        response = requests.get(file_path)
        if file_path[-1] == '/':
            file_path = file_path[:-1]
        file_name = file_name.split('/')[-1]
        sample_path = os.path.abspath("./input/" + file_name)
        open(sample_path, "wb").write(response.content)
    else:
        sample_path = file_name

    # generating audio time series and a sampling rate (int)
    y, sr = librosa.load(os.path.abspath(sample_path), mono=False)

    return y, sr


'''------------------------------------
OUTPUT GENERATOR:
    receives a destination path, file name, audio matrix, and sample rate,
    generates a wav file based on input
------------------------------------'''
def output_file(idx, destination ,filename, y, sr, reference_file, use_limiter, normalize, ext=""):
    destination = os.path.join(os.path.abspath(destination), Path(filename[:-4] + ext + '.wav').name)
    print('DESTINATION=', destination)
    print('SHAPE=', y.shape, y.dtype)
    with sf.SoundFile(destination, 'w', sr, 2, 'FLOAT') as f:
        f.write(np.hstack((y[0].reshape(-1, 1), y[1].reshape(-1,1))))
    postprocess(destination, 44100, reference_file, use_limiter, normalize)
    postclean(idx, destination)


def postprocess(source_file, sample_rate, reference_file, use_limiter, normalize):
    mg.core.process(
        target=source_file,
        reference=reference_file,
        results = [Result(source_file.replace('.wav', '.processed.wav'), subtype='FLOAT', use_limiter=use_limiter, normalize=normalize)],
        config=Config(internal_sample_rate=sample_rate)
    )


def postclean(idx, source_file):
    y, sr = read_file(source_file.replace('.wav', '.processed.wav'))
    filtered_y = random.choice(filters)(y, sr)
    with sf.SoundFile(source_file.replace('.wav', '.processed_filtered.wav'), 'w', sr, 2, 'FLOAT') as f:
        f.write(np.hstack((filtered_y[0].reshape(-1, 1), filtered_y[1].reshape(-1,1))))


def parse_args():
    parser = ArgumentParser(
        description="DEVINYL - Recover vinyls beyond recovering"
    )
    parser.add_argument("input", type=str, help="The track (file or URL) you want to master")
    parser.add_argument("reference", type=str, help='Some reference track (file or URL)')
    # parser.add_argument("result", type=str, help="Where to save your result")
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

filters = [reduce_noise_centroid_s for _ in range(7)]

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

    filtered_y_list = [filter(y, sr) for filter in pre_filters]

    # trimming silences
    trimmed_y_list = [trim_silence(to_trim)[0] for to_trim in filtered_y_list]
    
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
        output_file(i, './output/', args.target, trimmed_y_list[i], sr, args.reference, suffixes[i], args.use_limiter, args.normalize)


if __name__ == "__main__":
    run(*prepare_logger(parse_args()))