# DEVINYL - Recover vinyls beyond recovering

This project started as a proof of concept that old vynils could be recovered at least a bit behind all the noise, crackling and clipping captured by the record player. It's still a work in progress, but since results seem to be decent enough to start working on more advanced features (like a lightweight Wiener filter capturing a portion of only noise, allowing a much cleaner filter).

It needs a reference audio. Preferably something that covers most of the audio spectrum without saturating will do fine, like most of 90's rock or pop records for example. But that's the thing. There's not an exact formula for every song, so feel free to experiment and check out the variation on the results!

### Requirements

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

### Pip

Then follow the pip instructions ahead.

install all necesary dependencies:

```bash
pip install -r requirements.txt
```

### Usage

```
devinyl.py [-h] [-b {16,24,32}] [--log LOG] [--no_limiter]
           [--dont_normalize]
           target_file reference_file

positional arguments:
    target                The track you want to start from
    reference             Some reference track to enhance the base sound

optional arguments:
    -h, --help            show this help message and exit
    -b {16,24,32}, --bit {16,24,32}
                        The bit depth of your mastered result. 32 means 32-bit
                        float
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
