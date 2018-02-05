import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

data, sample_rate = librosa.load('./Robert_Hood-Chase.ogg')

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(data)

# Beat track on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sample_rate)

# Compute MFCC features from the raw signal
mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, hop_length=hop_length, n_mfcc=13)

# And the first-order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

# Compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sample_rate)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

# Finally, stack all beat-synchronous features together
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])



# print('bpm: {:.2f}'.format(tempo))

# beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

# print('saving beat times...')
# librosa.output.times_csv('beat_times.csv', beat_times)
