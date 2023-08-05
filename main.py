#!/usr/bin/python3

import wave
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
from scipy.signal import find_peaks

filename = ''
output_filename = ''
thresh = ''
envelope_smoothness = ''
exclusion = ''
float_prec = ''
verbose = ''

parser = argparse.ArgumentParser(description='Map transient times')
parser.add_argument('-f', '--file', dest='filename', type=str, action='store', help='File to open')
parser.add_argument('-o', '--out', dest='output_filename', type=str, action='store', help='Filename to write output values to')
parser.add_argument('-t', '--threshold', dest='thresh', default='0.25', type=float, action='store', help='DEFAULT=0.25 Peak detection threshold, lower is rougher')
parser.add_argument('-c', '--number-channels', dest='num_channels', type=int, action='store', help='DEFAULT=3 Number of channels, 2=MONO, 3=STEREO, etc')
parser.add_argument('-s', '--channel-offset', dest='off_channel', type=int, action='store', help='DEFAULT=2 Channel offset, channel to analyze.')
parser.add_argument('-e', '--envelope-smoothness', dest='envelope_smoothness', default='100', type=int, action='store', help='DEFAULT=100 Amount of rounding around the envelope')
parser.add_argument('-x', '--exclusion', dest='exclusion', default='30', type=int, action='store', help='DEFAULT=30 Exclusion threshold')
parser.add_argument('-p', '--precision', dest='float_prec', default='6', type=int, action='store', help='DEFAULT=6 Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438')
parser.add_argument('-v', '--verbose', help="Set debug logging", action='store_true')
args = parser.parse_args()

def main():

    if args.verbose:
        print(args)
        # Set logging level - https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # User configuration values
    envelope_smoothness = args.envelope_smoothness
    exclusion = args.exclusion
    filename = args.filename
    output_filename = args.output_filename
    threshold = args.thresh
    float_prec = args.float_prec

    # Open wav file
    wave_file = wave.open(filename, 'rb')

    frame_rate = wave_file.getframerate()
    num_frames = wave_file.getnframes()
    logging.info("Analyzing {}, {}Hz and {} frames.".format(filename, frame_rate, num_frames))
    amplitude_data = np.frombuffer(wave_file.readframes(num_frames), dtype=np.int64)
    numchannel = 3
    channeloffset = 2
    amplitude_data = amplitude_data[channeloffset::numchannel]

    normalized_amplitude = amplitude_data / np.max(np.abs(amplitude_data))
    wave_file.close()
    normalized_amplitude = replace_negatives_with_neighbors(normalized_amplitude)
    envel = create_envelope(np.abs(normalized_amplitude),envelope_smoothness);
    norm_envelope = envel / np.max(np.abs(envel))
    # Set the threshold value
    #threshold = 0.25
    # Find peak maxima above the threshold
    peaks_roughly, _ = find_peaks(norm_envelope, prominence=threshold,width = exclusion)
    logging.info("Found {} rough peaks, refining...".format(len(peaks_roughly)))
    time = np.arange(0, len(normalized_amplitude)) / frame_rate *1000
    peaks = []
    for peak in peaks_roughly:
        search_range = 150
        max_value, max_index = find_maximum_around_peak(np.abs(normalized_amplitude), peak, search_range)
        peaks.append(max_index)
    peaks = np.array(peaks)
    logging.info("Refined to {} peaks, calculating times...".format(len(peaks)))
    timearray = peaks * 1000/frame_rate
    differences = np.diff(timearray)
    differences = np.append(differences, 0)

    combined_array = np.column_stack((timearray, differences))
    logging.debug("{}\n{}".format(timearray, differences))

    output_filename = filename[:-4]+".csv"
    logging.info("Saving output values to {}".format(output_filename))
    np_fmt = "%1.{}f".format(float_prec)

    np.savetxt(output_filename, combined_array, delimiter=",", header="Times,differences", fmt='%1.6f', comments="")

    #print(timearray*1000)
    #print(differences*1000)
    print(timearray)
    print(differences)
    plt.figure(figsize=(10, 4))
    plt.plot(time, normalized_amplitude)
    #plt.plot(time, norm_envelope)
    # Plot the peaks with red dots
    plt.plot(time[peaks], normalized_amplitude[peaks], 'ro', markersize=2, label='Peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Waveform Plot')
    plt.legend()
    plt.show()

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


if __name__ == '__main__':
    main()



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

