import matplotlib.pyplot as plt
import librosa
import librosa.display

data, sample_rate = librosa.load('./Robert_Hood-Chase.ogg')

tempo, beat_frames = librosa.beat.beat_track(y=data, sr=sample_rate)

print('bpm: {:.2f}'.format(tempo))

beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

print('saving beat times...')
librosa.output.times_csv('beat_times.csv', beat_times)
