import io
import math
import time
from datetime import timedelta as td

import librosa
import noisereduce as nr
import numpy as np
from pydub import AudioSegment
from pysndfx import AudioEffectsChain
import python_speech_features
import scipy as sp
from scipy import signal

from utils import pydub_to_np


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length) 


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)

def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    #print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = sp.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    return recovered_signal

def reduce_noise_dummy(y, sr):
    return y

'''------------------------------------
NOISE REDUCTION USING POWER:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''
def reduce_noise_power(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.4
    threshold_l = round(np.median(cent))*0.4

    less_noise = AudioEffectsChain()\
        .lowshelf(gain=-50.0, frequency=threshold_l, slope=0.5)\
        .highshelf(gain=-12.0, frequency=threshold_h, slope=0.7)\
        .limiter(gain=6.0)
    y_clean = less_noise(y)

    return y_clean


def reduce_noise_power2(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.4
    threshold_l = round(np.median(cent))*0.2

    less_noise = AudioEffectsChain()\
        .lowshelf(gain=-24.0, frequency=threshold_l, slope=0.7)\
        .highshelf(gain=-26.0, frequency=threshold_h, slope=0.6)\
        .limiter(gain=12.0)
    y_clean = less_noise(y)

    return y_clean

def reduce_noise_notch(y, sr):
# Create/view notch filter
    samp_freq = 1000  # Sample frequency (Hz)
    notch_freq = 60.0  # Frequency to be removed from signal (Hz)
    quality_factor = 30.0  # Quality factor
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    freq, h = signal.freqz(b_notch, a_notch, fs = samp_freq)

    # Create/view signal that is a mixture of two frequencies
    f1 = 17
    f2 = 60
    t = np.linspace(0.0, 1, 1_000)
    y_pure = np.sin(f1 * 2.0*np.pi*t) + np.sin(f2 * 2.0*np.pi*t) 

    # apply notch filter to signal
    y_notched = signal.filtfilt(b_notch, a_notch, y_pure)
    return y_notched

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

    less_noise = AudioEffectsChain()\
        .lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5)\
        .highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)\
        .limiter(gain=6.0)

    y_cleaned = less_noise(y)

    return y_cleaned

def reduce_noise_centroid_mb(y, sr):
    y = np.abs(y)
    
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain()\
        .lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5)\
        .highshelf(gain=-30.0, frequency=threshold_h, slope=0.5)\
        .limiter(gain=10.0)
    less_noise = AudioEffectsChain()\
        .lowpass(frequency=threshold_h)\
        .highpass(frequency=threshold_l)
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

    speech_booster = AudioEffectsChain()\
        .lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5)\
        .highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.5)#.limiter(gain=8.0)
    y_speach_boosted = speech_booster(y)

    return (y_speach_boosted)

'''------------------------------------
NOISE REDUCTION USING FAST FOURIER TRANSFORM:
    receives an audio matrix,
    returns the matrix after a noise reduction
------------------------------------'''
def reduce_noise_fft(y, sr):
    # Normalize the signal
    y /= 2**15

    # Apply Fourier transform
    freq_signal = np.fft.fft(y)

    # Create a binary mask to filter out the vinyl noise
    freqs = np.fft.fftfreq(len(y), 1.0/sr)
    mask = np.abs(freqs) > 1000 # set threshold to 1000Hz

    # Apply the filter
    freq_signal_filtered = freq_signal.copy()
    freq_signal_filtered[mask] = 0

    # Apply inverse Fourier transform
    filtered_signal = np.real(np.fft.ifft(freq_signal_filtered))

    return (filtered_signal)

def reduce_noise_adaptive(y, sr):
    # Define the filter length
    filter_length = 1024

    # Define the step size for the LMS algorithm
    step_size = 0.01

    # Define the number of iterations for the LMS algorithm
    n_iterations = 100

    # Initialize the adaptive filter coefficients
    coeffs_left = np.zeros(filter_length)
    coeffs_right = np.zeros(filter_length)

    # Define the reference signal for the LMS algorithm
    ref_left = np.zeros(filter_length)
    ref_right = np.zeros(filter_length)
    ref_left[filter_length // 2] = 1
    ref_right[filter_length // 2] = 1

    # Apply the LMS algorithm to the audio signal
    for n in range(n_iterations):
        # Get a segment of the audio signal and the reference signal
        start_idx = n * filter_length
        end_idx = start_idx + filter_length
        audio_segment_left = y[0, start_idx:end_idx]
        audio_segment_right = y[1, start_idx:end_idx]
        ref_segment_left = ref_left
        ref_segment_right = ref_right
        
        # Compute the filter output for each channel
        filter_output_left = np.dot(coeffs_left, audio_segment_left)
        filter_output_right = np.dot(coeffs_right, audio_segment_right)
        
        # Compute the error signal for each channel
        error_left = ref_segment_left - filter_output_left
        error_right = ref_segment_right - filter_output_right
        
        # Update the filter coefficients for each channel
        coeffs_left += step_size * np.conj(audio_segment_left) * error_left
        coeffs_right += step_size * np.conj(audio_segment_right) * error_right
    
    # Apply the adaptive filter to the audio signal
    filtered_audio_signal = np.zeros_like(y)
    for n in range(len(y) // filter_length):
        start_idx = n * filter_length
        end_idx = start_idx + filter_length
        audio_segment_left = y[0, start_idx:end_idx]
        audio_segment_right = y[1, start_idx:end_idx]
        
        # Compute the filter output for each channel
        filter_output_left = np.dot(coeffs_left, audio_segment_left)
        filter_output_right = np.dot(coeffs_right, audio_segment_right)
        
        # Subtract the filter output from the audio signal to remove the noise
        filtered_audio_segment_left = audio_segment_left - filter_output_left
        filtered_audio_segment_right = audio_segment_right - filter_output_right
        
        # Store the filtered audio signal
        filtered_audio_signal[0, start_idx:end_idx] = filtered_audio_segment_left
        filtered_audio_signal[1, start_idx:end_idx] = filtered_audio_segment_right
    
    return (filtered_audio_signal)

'''------------------------------------
NOISE REDUCTION USING MEDIAN:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
------------------------------------'''

def reduce_noise_median(y, sr):
    y = sp.signal.medfilt(y,3)
    return y


def reduce_noise_no_reduction(y, sr):
    return y

'''------------------------------------
NOISE REDUCTION USING WIENER:
    receives an audio matrix,
    returns the matrix after a wiener filtering on the audio wave.
------------------------------------'''

def reduce_noise_wiener(y, sr):
    # Add random noise for wiener filtering
    # noise_threshold = 0.25
    # y = np.random.normal(y, np.abs(noise_threshold * y))  # TODO: Pass bitrate as an argument
    for _ in range(3):
        y = sp.signal.wiener(y, 2, 0.1)
    return (y)


def reduce_noise_wiener_dec(y, sr):
    booster = AudioEffectsChain().gain(10.0).limiter(gain=4.0)  # It's just a tiny limiter, don't nag about the settings :P
    # Convert numpy array to pyDub AudioSegment
    y_boosted = booster(y)
    y = reduce_noise_power(y_boosted, sr)
    y = AudioSegment(data=y.tobytes(),
                     sample_width=2,
                     frame_rate=sr, channels=2)

    # wav_io = io.BytesIO()
    # sp.io.wavfile.write(wav_io, sr * 2, y)
    # wav_io.seek(0)
    # y = AudioSegment.from_wav(wav_io)
    [y1, y2] = [pydub_to_np(sample) for sample in y.split_to_mono()]
    S1 = librosa.feature.melspectrogram(y=y1[1], sr=sr, n_mels=64)
    S2 = librosa.feature.melspectrogram(y=y2[1], sr=sr, n_mels=64)
    
    # yf1 = f_high(y1, sr)
    # yf2 = f_high(y2, sr)
    
    Dp1 = librosa.pcen(
        S1 * (2**31), sr=sr, gain=1.1, hop_length=128, bias=2, power=0.5, time_constant=0.8, eps=1e-06, max_size=2)
    Dp2 = librosa.pcen(
        S2 * (2**31), sr=sr, gain=1.1, hop_length=128, bias=2, power=0.5, time_constant=0.8, eps=1e-06, max_size=2)
    
    yp1 = librosa.feature.inverse.mel_to_audio(Dp1)
    yp2 = librosa.feature.inverse.mel_to_audio(Dp2)
    y = np.array([yp1, yp2])

    # Add random noise for wiener filtering
    noise_threshold = 0.25
    y = np.random.normal(yp2, np.abs(noise_threshold * y))  # TODO: Pass bitrate as an argument
    for idx in range(5):
        y = sp.signal.wiener(y, 12 - idx * 2)
    return y


def reduce_noise_sample(y, sr):
    # Split stereo to mono
    left_channel = y[0, :]
    right_channel = y[1, :]
    left = True
    # Select a section of your audio where only noise is present
    noisy_left = left_channel[25000:35000]
    noisy_right = right_channel[25000:35000]

    for _ in range(4):
        # Perform noise reduction on each channel separately
        reduced_noise_left = nr.reduce_noise(left_channel, sr, y_noise=noisy_left, n_fft=1024, prop_decrease=0.3, time_mask_smooth_ms=12)
        reduced_noise_right = nr.reduce_noise(right_channel, sr, y_noise=noisy_right, n_fft=1024, prop_decrease=0.3, time_mask_smooth_ms=12)
        left_channel = reduced_noise_left
        right_channel = reduced_noise_right
        noisy_left = left_channel[25000:35000]
        noisy_right = right_channel[25000:35000]

    # Combine two channels
    processed_y = np.vstack((reduced_noise_left, reduced_noise_right)).T
    return processed_y


def f_high(y, sr):
    b, a = sp.signal.butter(10, 2000 / (sr / 2), btype='highpass')
    yf = sp.signal.lfilter(b, a, y)
    return yf


'''------------------------------------
SILENCE TRIMMER:
    receives an audio matrix,
    returns an audio matrix with less silence and the amout of samples that were trimmed
------------------------------------'''
def trim_silence(y):
    y_trimmed, noise_y = librosa.effects.trim(y=y, top_db=30, frame_length=2, hop_length=250)
    trimmed_length = len(y_trimmed[1])

    return y_trimmed, trimmed_length, noise_y


'''------------------------------------
AUDIO ENHANCER:
    receives an audio matrix,
    returns the same matrix after audio manipulation
------------------------------------'''
def enhance(y):
    apply_audio_effects = AudioEffectsChain()\
        .lowshelf(gain=10.0, frequency=260, slope=0.1)\
        .reverb(reverberance=25, hf_damping=5, room_scale=5, stereo_depth=50, pre_delay=20, wet_gain=0, wet_only=False)
        #.normalize()
    y_enhanced = apply_audio_effects(y)

    return y_enhanced