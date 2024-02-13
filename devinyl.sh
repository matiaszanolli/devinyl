#!/bin/bash

# Based on this guide: http://www.zoharbabin.com/how-to-do-noise-reduction-using-ffmpeg-and-sox/

# Check that FFMPEG and sox are installed
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
if [ "$#" -eq 0 ]; then
    echo "No file given to clean up."
    exit
fi

# Variables
INPUTFILE="$1"
OUTPUTFILE="${INPUTFILE%.*}"
DECOMPRESSED="${OUTPUTFILE}.wav"

# Decompress the audio file
echo "Decompressing the audio file..."
ffmpeg -i "$INPUTFILE" "$DECOMPRESSED"

# Listen to first half second of audio
echo "Listening to first half second of audio to get noise level..."
ffmpeg -i "$DECOMPRESSED" -ss 00:00:01.0 -t 00:00:02.0 noiseaudpre.wav

# Generate noise profile
echo "Making noise profile..."
sox noiseaudpre.wav -n noiseprof noisepre.prof

# Clean the audio file with the profile created
echo "Cleaning the audio with the profile..."
sox "$DECOMPRESSED" "${OUTPUTFILE}-preclean.wav" noisered noisepre.prof 0.21

# Listen to first half second of audio
echo "Listening to first half second of audio to get noise level..."
ffmpeg -i "${OUTPUTFILE}-preclean.wav" -ss 00:00:01.0 -t 00:00:02.0 noiseaud.wav

# Generate second noise profile
echo "Making second noise profile..."
sox noiseaud.wav -n noiseprof noise.prof

# Clean the audio file with the second profile
echo "Cleaning the audio with the second profile..."
sox "${OUTPUTFILE}-preclean.wav" "${OUTPUTFILE}-clean.wav" noisered noise.prof 0.21

# Convert the cleaned audio to FLAC
ffmpeg -i "${OUTPUTFILE}-clean.wav" "${OUTPUTFILE}-clean.flac"

echo "Cleaning up..."
# Clean up temp files
rm noiseaudpre.wav noiseaud.wav noisepre.prof noise.prof "${OUTPUTFILE}-preclean.wav" "${OUTPUTFILE}-clean.wav" "$DECOMPRESSED"