import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
audio_path = 'data.wav'
y, sr = librosa.load(audio_path)

# Use the onset detection function with backtrack to find beat frames
# Adjust the backtrack value to control sensitivity (higher values exclude minor beats)
backtrack_value = 1  # Adjust this value as needed
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=backtrack_value)

# Convert beat frames to times
beat_times = librosa.frames_to_time(onset_frames, sr=sr)

# Plot the audio waveform and detected beats
plt.figure(figsize=(10, 6))
librosa.display.waveshow(y, sr=sr)
plt.vlines(beat_times, -1, 1, color='r', alpha=0.7, label='Beats')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform with Detected Beats')
plt.legend()
plt.tight_layout()
plt.show()

# Print the detected beat times
print("Detected beat times:", beat_times)
