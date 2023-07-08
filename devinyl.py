import os
import logging
import sys
import traceback
from argparse import ArgumentParser
from os.path import exists
from pathlib import Path
from urllib.parse import urlparse


import librosa
import noisereduce as nr
import numpy as np
import matchering as mg
import requests
import soundfile as sf
from matchering.core import Config
from matchering.results import Result
from tqdm import tqdm

from effects import (reduce_noise_centroid_mb, reduce_noise_centroid_s, reduce_noise_median, reduce_noise_mfcc_down, 
                     reduce_noise_mfcc_up, reduce_noise_no_reduction, reduce_noise_power, reduce_noise_power2, 
                     reduce_noise_wiener, reduce_noise_wiener_dec, reduce_noise_fft, reduce_noise_notch,
                     reduce_noise_dummy, reduce_noise_adaptive, reduce_noise_sample, trim_silence)

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

    def is_pathname_valid(pathname: str) -> bool:
        '''
        `True` if the passed pathname is a valid pathname for the current OS;
        `False` otherwise.
        '''
        # If this pathname is either not a string or is but is empty, this pathname
        # is invalid.
        try:
            if not isinstance(pathname, str) or not pathname:
                return False

            # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
            # if any. Since Windows prohibits path components from containing `:`
            # characters, failing to strip this `:`-suffixed prefix would
            # erroneously invalidate all valid absolute Windows pathnames.
            drive_name, pathname = os.path.splitdrive(pathname)

            # Directory guaranteed to exist. If the current OS is Windows, this is
            # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
            # environment variable); else, the typical root directory.
            root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
                if sys.platform == 'win32' else os.path.sep
            assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

            # Append a path separator to this directory if needed.
            root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

            # Test whether each path component split from this pathname is valid or
            # not, ignoring non-existent and non-readable path components.
            for pathname_part in pathname.split(os.path.sep):
                try:
                    os.lstat(drive_name + pathname_part)
                # If an OS-specific exception is raised, its error code
                # indicates whether this pathname is valid or not. Unless this
                # is the case, this exception implies an ignorable kernel or
                # filesystem complaint (e.g., path not found or inaccessible).
                #
                # Only the following exceptions indicate invalid pathnames:
                #
                # * Instances of the Windows-specific "WindowsError" class
                #   defining the "winerror" attribute whose value is
                #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
                #   fine-grained and hence useful than the generic "errno"
                #   attribute. When a too-long pathname is passed, for example,
                #   "errno" is "ENOENT" (i.e., no such file or directory) rather
                #   than "ENAMETOOLONG" (i.e., file name too long).
                # * Instances of the cross-platform "OSError" class defining the
                #   generic "errno" attribute whose value is either:
                #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
                #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
                except OSError as exc:
                    traceback.print_exc()
        # If a "TypeError" exception was raised, it almost certainly has the
        # error message "embedded NUL character" indicating an invalid pathname.
        except TypeError as exc:
            return False
        # If no exception was raised, all path components and hence this
        # pathname itself are valid. (Praise be to the curmudgeonly python.)
        else:
            return True
    
    if not is_pathname_valid(file_path.replace('\\', '/')):
        response = requests.get(file_path, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)
        if file_path[-1] == '/':
            file_path = file_path[:-1]
        file_name = file_path.split('/')[-1]
        sample_path = os.path.abspath("./input/" + file_name)
        with open(sample_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise requests.exceptions.ConnectionError(f"ERROR while downloading {file_name}.")
    else:
        sample_path = file_path

    # generating audio time series and a sampling rate (int)
    y, sr = librosa.load(os.path.abspath(sample_path), mono=False)

    return y, sr


'''------------------------------------
OUTPUT GENERATOR:
    receives a destination path, file name, audio matrix, and sample rate,
    generates a wav file based on input
------------------------------------'''
def output_file(destination ,filename, y, sr, ext, use_limiter, normalize, fast):
    # audio_stft = librosa.stft(y, n_fft=2048, hop_length=512)
    # noise_stft = librosa.stft(y_noise, n_fft=2048, hop_length=512)
    # audio_mag = np.abs(audio_stft)
    # noise_mag = np.abs(noise_stft)
    # noise_profile = np.median(noise_mag, axis=2, keepdims=True)
    # mask = np.divide(audio_mag, noise_profile)
    # audio_stft_denoised = np.multiply(audio_stft, mask)
    # y = librosa.istft(audio_stft_denoised, hop_length=512)

    destination = os.path.join(os.path.abspath(destination), Path(filename[:-4] + ext + '.wav').name)
    with sf.SoundFile(destination, 'w', sr, channels=2, format='FLAC') as f:
        f.write(np.hstack((y[0].reshape(-1, 1), y[1].reshape(-1,1))))
    
    postprocess(destination, 44100, './reference.wav', use_limiter, normalize, False)
    if not fast:
        postclean(destination)
#        postprocess(destination, 44100, reference_file, use_limiter, normalize, True)


def postprocess(source_file, sample_rate, reference_file, use_limiter, normalize, second_stage):
    
    mg.core.process(
        target=source_file if not second_stage else source_file.replace('.wav', '.postprocess.postclean.wav'),
        reference=reference_file,
        results=[Result(source_file.replace('.wav', '.postprocess.wav' if not second_stage else '.final.wav'), subtype='FLOAT', use_limiter=use_limiter, normalize=normalize)],
        config=Config(internal_sample_rate=sample_rate)
    )


def postclean(source_file):
    file_path = source_file.replace('.wav', '.postprocess.wav')

    # with sf.SoundFile(source_file.replace('.wav', '.noise.wav'), 'w', 44100, 2, 'FLOAT') as f:
    #     f.write(np.hstack((y_noise[0].reshape(-1, 1), y_noise[1].reshape(-1, 1))))

    # # If there's enough silence to gather background noise from, we gather the stats and run the filter
    # if len(y_noise[1]) > 30000:
    #     for _ in tqdm(range(1)):  # Wonderful tri-destilation
    #         filtered_y = nr.reduce_noise(y=filtered_y, y_noise=y_noise, sr=sr, prop_decrease=0.4, n_jobs=-1, sigmoid_slope_nonstationary=10)
    y, sr = read_file(file_path)
    filtered_y = filter(y, sr)

    with sf.SoundFile(source_file.replace('.wav', '.postprocess.postclean.wav'), 'w', sr, 2, 'FLOAT') as f:
        f.write(filtered_y)


def parse_args():
    parser = ArgumentParser(
        description="DEVINYL - Recover vinyls beyond recovering"
    )
    parser.add_argument("input", type=str, help="The track (file or URL) you want to master")
#     parser.add_argument("reference", type=str, help='Some reference track (file or URL)')
    # parser.add_argument("result", type=str, help="Where to save your result")
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="The file to which the logs will be written",
    )
    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        help="(NEW) Fast mode - Runs half of the process. Gets a decent result at half the processing time, "
             "yet the result is not as clean as doing the full process",
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


pre_filter = reduce_noise_no_reduction


filter = reduce_noise_sample


# filters = [
#     reduce_noise_power,
#     reduce_noise_power,
#     reduce_noise_centroid_mb,
#     reduce_noise_mfcc_up,
#     reduce_noise_mfcc_down,
#     reduce_noise_median,
#     reduce_noise_no_reduction,
#     reduce_noise_notch
# ]


def run(args, logger):
    logger.info('##############################################')
    logger.info('# DEVINYL - Recover vinyls beyond recovering #')
    logger.info('##############################################')

    mg.log(
        warning_handler=logger.warning,
        info_handler=logger.info,
        debug_handler=logger.debug,
    )

    # reading a file
    y, sr = read_file(args.input)

    filtered_y = pre_filter(y, sr)

    # trimming silences
    # y_trimmed, trimmed_length, noise_y = trim_silence(filtered_y)

    # noise_samples = len(filtered_y[1]) - len(y_trimmed[1])  # Also taken as the index of the place where the music starts
    #
    # # We don't want the section where the disc spins up, neither nothing too close to the edge
    # noise_idx_low = int(noise_samples * 0.3)
    # noise_idx_high = int(noise_samples * 0.9)
    #
    # # This should be our pure vinyl noise sample
    # noise_fragment = filtered_y[:, noise_idx_low:noise_idx_high]
    
    use_limiter = not args.no_limiter
    normalize = not args.dont_normalize
    # generating output files
    output_file('./output/', args.input, filtered_y, sr, 'devyniled', use_limiter, normalize, args.fast)


if __name__ == "__main__":
    run(*prepare_logger(parse_args()))
