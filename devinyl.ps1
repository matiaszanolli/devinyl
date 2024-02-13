# Based on this guide: http://www.zoharbabin.com/how-to-do-noise-reduction-using-ffmpeg-and-sox/

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

# Listen to first half second of audio
Write-Host "Listening to first half second of audio to get noise level..."
ffmpeg -i $DECOMPRESSED -ss 00:00:01.0 -t 00:00:02.0 noiseaudpre.wav

# Generate noise profile
Write-Host "Making noise profile..."
sox noiseaudpre.wav -n noiseprof noisepre.prof

# Clean the audio file with the profile created
Write-Host "Cleaning the audio with the profile..."
sox $DECOMPRESSED $OUTPUTFILE-preclean.wav noisered noisepre.prof 0.21

# Listen to first half second of audio
Write-Host "Listening to first half second of audio to get noise level..."
ffmpeg -i $OUTPUTFILE-preclean.wav -ss 00:00:01.0 -t 00:00:02.0 noiseaud.wav

# Generate second noise profile
Write-Host "Making second noise profile..."
sox noiseaud.wav -n noiseprof noise.prof

# Clean the audio file with the second profile
Write-Host "Cleaning the audio with the second profile..."
sox $OUTPUTFILE-preclean.wav $OUTPUTFILE-clean.wav noisered noise.prof 0.21

# Convert the cleaned audio to FLAC
ffmpeg -i $OUTPUTFILE-clean.wav $OUTPUTFILE-clean.flac

Write-Host "Cleaning up..."
# Clean up temp files
Remove-Item noiseaudpre.wav, noiseaud.wav, noisepre.prof, noise.prof, $OUTPUTFILE-preclean.wav, $OUTPUTFILE-clean.wav, $DECOMPRESSED