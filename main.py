#!/usr/bin/python3

import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def find_maximum_around_peak(data, peak_location, search_range):
    """
    Find the maximum value within a specified search range around a given peak location.

    Parameters:
        data (numpy.ndarray): Input data.
        peak_location (int): Index of the located peak.
        search_range (int): Number of indices to search around the peak location.

    Returns:
        float: Maximum value found within the search range.
        int: Index of the maximum value within the search range.
    """
    start_index = max(0, peak_location - search_range)
    end_index = min(len(data), peak_location + search_range + 1)
    max_value = np.max(data[start_index:end_index])
    max_index = np.argmax(data[start_index:end_index]) + start_index
    return max_value, max_index


def create_envelope(signal, window_size):
    """
    Create an envelope for the input signal using a moving average.

    Parameters:
        signal (numpy.ndarray): Input signal.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: Envelope of the signal.
    """
    absolute_signal = np.abs(signal)
    envelope = moving_average(absolute_signal, window_size)
    return envelope

def moving_average(data, window_size):
    """
    Apply a moving average smoothing to the input data.

    Parameters:
        data (numpy.ndarray): Input data to be smoothed.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: Smoothed data.
    """
    window = np.ones(window_size) / float(window_size)
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data







def replace_negatives_with_neighbors(lst):
    new_lst = lst.copy()  # Create a copy of the original list
    for i in range(len(lst)):
        if lst[i] < 0:
            # Find the nearest non-negative neighbors
            left_neighbor = next((x for x in reversed(lst[:i]) if x >= 0), None)
            right_neighbor = next((x for x in lst[i:] if x >= 0), None)

            # Replace negative value with the nearest neighbor
            if left_neighbor is not None and right_neighbor is not None:
                if abs(lst[i] - left_neighbor) <= abs(lst[i] - right_neighbor):
                    new_lst[i] = left_neighbor
                else:
                    new_lst[i] = right_neighbor
            elif left_neighbor is not None:
                new_lst[i] = left_neighbor
            elif right_neighbor is not None:
                new_lst[i] = right_neighbor
    return new_lst


filename = './TEST_8Transients_2.wav'
wave_file = wave.open(filename, 'rb')

frame_rate = wave_file.getframerate()
num_frames = wave_file.getnframes()
amplitude_data = np.frombuffer(wave_file.readframes(num_frames), dtype=np.int64)
numchannel = 3
channeloffset = 2
amplitude_data = amplitude_data[channeloffset::numchannel]

normalized_amplitude = amplitude_data / np.max(np.abs(amplitude_data))
wave_file.close()
normalized_amplitude = replace_negatives_with_neighbors(normalized_amplitude)
envelope_smoothness = 100
envel = create_envelope(np.abs(normalized_amplitude),envelope_smoothness);
norm_envelope = envel / np.max(np.abs(envel))
# Set the threshold value
threshold = 0.25
exclusion = 30
# Find peak maxima above the threshold
peaks_roughly, _ = find_peaks(norm_envelope, prominence=threshold,width = exclusion)
time = np.arange(0, len(normalized_amplitude)) / frame_rate
peaks = []
for peak in peaks_roughly:
	search_range = 150
	max_value, max_index = find_maximum_around_peak(np.abs(normalized_amplitude), peak, search_range)
	peaks.append(max_index)
peaks = np.array(peaks)
timearray = peaks/frame_rate
differences = np.diff(timearray)
differences = np.append(differences, 0)

combined_array = np.column_stack((timearray, differences))

output_filename = filename[:-4]+".csv"
np.savetxt(output_filename, combined_array, delimiter=",", header="Times,differences", comments="")


plt.figure(figsize=(10, 4))
plt.plot(time, normalized_amplitude)
#plt.plot(time, norm_envelope)
# Plot the peaks with red dots
plt.plot(time[peaks], normalized_amplitude[peaks], 'ro', markersize=2, label='Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform Plot')
plt.legend()
plt.show()

#frames = list(firstframes)
#print(frames)
#plt.plot(frames)
#plt.show()
#return
#for i in range(0,framenumber):
#	firstframes = waveform_raw.readframes(1)
#	numbers.append(int.from_bytes(firstframes, byteorder='big'))
	#print(number)

#plt.plot(numbers)
#plt.show()


