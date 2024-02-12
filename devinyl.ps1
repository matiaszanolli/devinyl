# Based on this guide: http://www.zoharbabin.com/how-to-do-noise-reduction-using-ffmpeg-and-sox/
# Script converted to PowerShell

# Check that FFMPEG and sox are installed
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "FFMPEG is not installed."
    exit
}

if (-not (Get-Command sox -ErrorAction SilentlyContinue)) {
    Write-Host "SOX is not installed."
    exit
}

# Check if a file has been passed for cleaning
if ($args.Count -eq 0) {
    Write-Host "No file given to clean up."
    exit
}

# Variables
$INPUTFILE = $args[0]
$OUTPUTFILE = $INPUTFILE -replace '\.[^.]+$', ''
$DECOMPRESSED = $OUTPUTFILE + '.wav'


# Decompress the audio file
Write-Host "Decompressing the audio file..."
ffmpeg -i $INPUTFILE $DECOMPRESSED

# Listen to first half second of video
Write-Host "Listening to first half second of video to get noise level..."
ffmpeg -i $DECOMPRESSED -ss 00:00:00.0 -t 00:00:00.5 noiseaud.wav

# Generate noise profile
Write-Host "Making noise profile..."
sox noiseaud.wav -n noiseprof noise.prof

# Clean the audio file with the profile created
Write-Host "Cleaning the audio with the profile..."
sox $DECOMPRESSED $OUTPUTFILE-clean.wav noisered noise.prof 0.21
ffmpeg -i $OUTPUTFILE-clean.wav $OUTPUTFILE-clean.flac

Write-Host "Cleaning up..."
# Clean up temp files
Remove-Item noiseaud.wav, noise.prof, $OUTPUTFILE-clean.wav, $DECOMPRESSED