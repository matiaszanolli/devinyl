# DEVINYL - Recover vinyls beyond recovering

DEVINYL is a tool with a very simple purpose in mind: Restore vinyls. The older and more damaged they are, the better the work it does.

### How does it work?

  Contrary to v1, DeVinyl v2 is as simple as it can get. Most vinyl records have at least a ~2 second audio gap before actually starting each song, specially old 78rpm records. So taking a sample from seconds 1 to 2 you can be pretty sure you're getting a pure noise sample (I'll make this adjustable in a future release for your custom tracks). From there, by using the awesome open source (SoX)[https://sourceforge.net/projects/sox/] library, we can create a noise profile from the source track and with said profile, just remove the majority of the noise of the song, including heavy hissing and clicking and without creating noticeable artifacts.

### Requirements

* Any modern OS 
* SoX binaries (available as packages on Linux via Apt / dnf / pacman, must download from website in Windows / macOS: (https://sourceforge.net/projects/sox/)[https://sourceforge.net/projects/sox/])
* ffmpeg binaries (same as above, though available in windows through chocolatey / Winget and macOS through brew)
* (Windows only) check that SoX and ffmpeg are accessible through terminal, otherwise add their respective location to your user's PATH environment variable.

### Usage

Linux/macOS
```
./devinyl.sh source_file
```

Windows (Powershell)
```
./devinyl.ps1 source_file
```

The clean track will be generated as `<source_track>_clean.flac`
## Extras

### Additional reference sites

These links helped me a lot better understanding the topics of audio processing and noise reduction in general: 

[Noise reduction gist](https://github.com/dodiku/noise_reduction/blob/master/noise.py)

[Noise reduction topic](http://dsp.stackexchange.com/search?q=noise+reduction/)

[noisereduce library](https://github.com/timsainb/noisereduce)

[matchering](https://github.com/sergree/matchering)

[matchering-cli](https://github.com/sergree/matchering-cli)

[Anaconda](https://www.anaconda.com/products/individual#Downloads)

[FFmpeg](https://www.ffmpeg.org/download.html)


### If this helped you out, please buy me a coffee!

https://www.buymeacoffee.com/matiaszanolli

### Or follow my channel in YouTube:

https://www.youtube.com/@TechforMusicAI
