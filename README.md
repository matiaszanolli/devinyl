# DEVINYL - Recover vinyls beyond recovering

DEVINYL is a tool with a very simple purpose in mind: Restore vinyls. The older and more damage they are, the better the work it does.

It's designed around two main concepts: speed and compatibility. It works on every major platform and architecture, and by using the amazing [Numba](http://numba.pydata.org/) library, it runs blazingly fast over a JIT compiler without worrying about any of all usual Python process and threading restrictions. All your CPUs will get a piece of the cake!

So you basically need an input record and a reference audio. Both can be either local files or remote URLs. 
About the reference audio, it should preferably be something that covers most of the audio spectrum without saturating it, like some 90's rock or pop record for example. But that's the thing. There's not an exact formula for every song, so feel free to experiment and check out the variation on the results!

### Requirements

* Python >= 3.7
* Linux / Windows / macOS (not tested on this last one but should work)
* SoX binaries (available as packages on Linux via Apt / dnf / pacman, must download from website in Windows / macOS)
* ffmpeg binaries (same as above, though available in windows through chocolatey / Winget and macOS through brew)
* (Windows only) check that SoX and ffmpeg are accessible through terminal, otherwise add their respective location to your user's PATH env variable.

## Install - Requirements (Pip - Conda - Apt)

### Conda

Create a new environment for your devinyling to happen:

```bash
conda create --name devinyl python=3.7

conda activate devinyl
```

(Any Python version above 3.7 will do so far)

Then follow the pip instructions ahead.

### Pip

install all necesary dependencies:

```bash
pip install -r requirements.txt
```

### Usage

```
devinyl.py [-h] [--log LOG] [--no_limiter]
           [--dont_normalize]
           target_file reference_file

positional arguments:
    input                 The track you want to start from
    reference             Some reference track to enhance the base sound

optional arguments:
    -h, --help            show this help message and exit
    --fast                (NEW) Fast mode - Runs half of the process. Gets a decent result at half the processing time, yet the result is not as clean as doing the full process
    --no_limiter          Disables the limiter at the final stage of processing
    --dont_normalize      Disables normalization, if --no_limiter is set.
```
## Extras

### Thanks to these wonderful sites and repos!

[Noise reduction gist](https://github.com/dodiku/noise_reduction/blob/master/noise.py)

[Noise reduction topic](http://dsp.stackexchange.com/search?q=noise+reduction/)

[matchering](https://github.com/sergree/matchering)

[matchering-cli](https://github.com/sergree/matchering-cli)

[Anaconda](https://www.anaconda.com/products/individual#Downloads)

[FFmpeg](https://www.ffmpeg.org/download.html)


### If this helped you out, please buy me a coffee!

https://www.buymeacoffee.com/matiaszanolli
