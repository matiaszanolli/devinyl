#!/bin/bash

# Check if ffmpeg and sox are installed
if ! command -v ffmpeg &> /dev/null
then
    echo "FFMPEG is not installed."
    exit
fi

if ! command -v sox &> /dev/null
then
    echo "SOX is not installed."
    exit
fi

# Check if a file has been passed for cleaning
if [ $# -eq 0 ]
then
    echo "No file given to clean up."
    exit
fi

# Variables
INPUTFILE=$1
OUTPUTFILE="${INPUTFILE%.*}"
DECOMPRESSED="$OUTPUTFILE.wav"

# Decompress the audio file
echo "Decompressing the audio file..."
ffmpeg -i $INPUTFILE $DECOMPRESSED

# Listen to first half second of video
echo "Listening to first half second of video to get noise level..."
ffmpeg -i $DECOMPRESSED -ss 00:00:00.0 -t 00:00:00.5 noiseaud.wav

# Generate noise profile
echo "Making noise profile..."
sox noiseaud.wav -n noiseprof noise.prof

# Clean the audio file with the profile created
echo "Cleaning the audio with the profile..."
sox $DECOMPRESSED $OUTPUTFILE-clean.wav noisered noise.prof 0.21
ffmpeg -i $OUTPUTFILE-clean.wav $OUTPUTFILE-clean.flac

echo "Cleaning up..."
# Clean up temp files
rm noiseaud.wav noise.prof $OUTPUTFILE-clean.wav $DECOMPRESSED
