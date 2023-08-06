#!/usr/bin/python3

import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from matplotlib.widgets import Slider
from matplotlib.backend_bases import key_press_handler
import argparse
import logging

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

    wave_file.close()

    numchannel = 3
    channeloffset = 2
    amplitude_data = amplitude_data[channeloffset::numchannel]

    normalized_amplitude = amplitude_data / np.max(np.abs(amplitude_data))

    normalized_amplitude = replace_negatives_with_neighbors(normalized_amplitude)
    envel = create_envelope(np.abs(normalized_amplitude),envelope_smoothness);
    norm_envelope = envel / np.max(np.abs(envel))
    # Find peak maxima above the threshold
    peaks_roughly, _ = find_peaks(norm_envelope, prominence=threshold,width = exclusion)
    logging.info("Found {} rough peaks, refining...".format(len(peaks_roughly)))
    time = np.arange(0, len(normalized_amplitude)) / frame_rate * 1000
    peaks = []
    for peak in peaks_roughly:
        search_range = 150
        max_value, max_index = find_maximum_around_peak(np.abs(normalized_amplitude), peak, search_range)
        peaks.append(max_index)
    peaks = np.array(peaks)
    logging.info("Refined to {} peaks, calculating times...".format(len(peaks)))
    timearray = peaks/frame_rate*1000
    differences = np.diff(timearray)
    differences = np.append(differences, 0)

    signal = normalized_amplitude
    segment_width = 1000

    # Create and plot the segments centered around the peaks
    #plt.figure(0, figsize=(10, 6))
    fig_centered, ax_centered = plt.subplots()
    fig_waveform, ax_waveform = plt.subplots()  # Use 1 for a single set of axes
    ax_peak_diff = ax_waveform.twinx()

    combined_array = np.column_stack((timearray, differences))

    output_filename = filename[:-4]+".csv"
    logging.info("Saving output values to {}".format(output_filename))
    np_fmt = "%1.{}f".format(float_prec)
    np.savetxt(output_filename, combined_array, delimiter=",", header="Times[ms],differences[ms]", fmt=np_fmt, comments="")

    #update_centered(time, signal, 400, peaks, ax_centered, normalized_amplitude)
    #update_peakdiff(time, signal, 150, peaks, ax_peak_diff, normalized_amplitude)
    #update_waveform(time, signal, 400, peaks, ax_waveform, normalized_amplitude, frame_rate)

    #plt.show(block=True)


    ''''
    # Add slider for zooming
    ax_slider_centered = plt.axes([0.2, 0.03, 0.65, 0.03])
    slider_centered = Slider(ax_slider_centered, 'Zoom', valmin=0, valmax=1000, valinit=400, valstep=10)

    slider_centered.on_changed(update_centered)

    #plt.figure(1, figsize=(10, 6))
    fig_peakdiff, ax_peakdiff = plt.subplots()  # Use 1 for a single set of axes
    # Add slider for zooming
    ax_slider_peakdiff = plt.axes([0.2, 0.03, 0.65, 0.03])
    slider_peakdiff = Slider(ax_slider_peakdiff, 'Zoom', valmin=0, valmax=1000, valinit=150, valstep=10)

    slider_peakdiff.on_changed(update_peakdiff)

    #plt.figure(2, figsize=(10, 6))
    fig_waveform, ax_waveform = plt.subplots()  # Use 1 for a single set of axes

    # Add slider for zooming
    waveformseg_width = 500 #[ms]
    ax_slider_waveform = plt.axes([0.2, 0.03, 0.65, 0.03])
    slider_waveform = Slider(ax_slider_waveform, 'Scroll', valmin=0, valmax=max(time-waveformseg_width), valinit=1, valstep=100)
    ax_peak_diff = ax_waveform.twinx()

    slider_waveform.on_changed(update_waveform)

    # Connect key press event to on_key function for all plots

    fig_centered.canvas.mpl_connect('key_press_event', on_key)
    fig_peakdiff.canvas.mpl_connect('key_press_event', on_key)
    fig_waveform.canvas.mpl_connect('key_press_event', on_key)
    '''


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

def update_centered(time, signal, val, peaks, ax_centered, normalized_amplitude):
    # Update centered segments plot here
    log_val = np.exp(val / 100)

    # Update centered segments plot here
    segment_width = int(log_val)
    ax_centered.clear()
    #ax_centered.plot(time, normalized_amplitude)
    peak = ''
    for peak in peaks:
        start = max(1, peak - segment_width)
        end = min(len(time), peak + segment_width)
        segment = signal[start:end]
        centered_x = time[start:end] - time[peak]
        ax_centered.plot(centered_x, segment, label='Transients centered on Maximum')
        ax_centered.plot(0, normalized_amplitude[peak], 'ro', markersize=4, label='Peaks')
    ax_centered.set_xlabel('Time [ms]')
    ax_centered.set_ylabel('Amplitude [a.u.]')
    ax_centered.set_title('Similarness plot')
    plt.subplots_adjust(bottom=0.25)
    plt.draw()

def update_peakdiff(time, signal, val, peaks, ax_peakdiff, normalized_amplitude):
    # Update peakdiff segments plot here
    log_val = np.exp(val / 100)

    # Update peakdiff segments plot here
    segment_width = int(log_val)
    ax_peakdiff.clear()

    for i in range(len(peaks) - 2):
        start = peaks[i]  # Start index at the current peak
        end = peaks[i + 2]  # End index at the next peak
        segment = signal[start:end]
        peakdiff_x = time[start:end] - time[start]  # Adjust x-axis values relative to the start
        ax_peakdiff.plot(peakdiff_x, segment)
        ax_peakdiff.plot(time[peaks[i+1] - peaks[i]], normalized_amplitude[peaks[i+1]], 'ro', markersize=4, label='Peaks')


    # Set labels and title
    ax_peakdiff.set_xlabel('Time [ms]')
    ax_peakdiff.set_ylabel('Amplitude [a.u.]')
    ax_peakdiff.set_title('Tightness plot')


    # Get the current x-axis limits
    current_xlim = ax_peakdiff.get_xlim()

    # Calculate the center of the data
    data_center = (current_xlim[0] + current_xlim[1]) / 2

    # Calculate the new x-axis limits based on the data center and segment width
    new_xlim = (data_center - segment_width, data_center + segment_width)

    # Set the new x-axis limits
    ax_peakdiff.set_xlim(new_xlim)

    # Display the legend and adjust the layout
    plt.subplots_adjust(bottom=0.25)

    plt.draw()  # Add this line to refresh the plot


def update_waveform(time, signal, val, peaks, ax_waveform, normalized_amplitude, frame_rate):

    waveformseg_width = 500 #[ms]
    #ax_slider_waveform = plt.axes([0.2, 0.03, 0.65, 0.03])
    #slider_waveform = Slider(ax_slider_waveform, 'Scroll', valmin=0, valmax=max(time-waveformseg_width), valinit=1, valstep=100)
    ax_peak_diff = ax_waveform.twinx()

    ax_waveform.clear()
    #ax_peak_diff = ax_waveform.twinx()
    ax_peak_diff.clear()
    # Calculate the visible x-axis range based on the slider value
    visible_start = val
    visible_end = val + waveformseg_width

    # Find the indices of the waveform data that correspond to the visible range
    visible_indices = np.where((time >= visible_start) & (time <= visible_end))

    # Downsample the waveform data for plotting
    downsample_factor = 1
    downsampled_indices = visible_indices[0][::downsample_factor]

    # Plot the downsampled waveform within the visible range
    ax_waveform.plot(time[downsampled_indices], normalized_amplitude[downsampled_indices], 'b')

    # Calculate the indices of the peaks in the downsampled data
    peaks_downsampled = np.intersect1d(downsampled_indices, peaks)

    # Plot the peaks as red dots on the downsampled waveform

    ax_waveform.plot(time[peaks], normalized_amplitude[peaks], 'ro', markersize=4, label='Peaks')

    ax_waveform.set_xlabel('Time [ms]')
    ax_waveform.set_ylabel('Amplitude [a.u.]')
    ax_waveform.set_title('Waveform/Consistency Plot')

    # Create a second y-axis for peak differences

    # Plot the peak differences as a bar diagram
    peak_differences = np.diff(peaks/frame_rate*1000)
    peak_middles = ((time[peaks[:-1]]+time[peaks[1:]])/2)
    ax_peak_diff.bar(peak_middles, peak_differences, width=peak_differences, align='center', alpha=0.5, color='green', label='Time between transients [ms]')

    ax_peak_diff.legend(loc='upper right')  # Move the legend to the right side
    visible_peaks = np.where((peak_middles >= visible_start) & (peak_middles <= visible_end))
    # Set x-axis limits to the edges of the visible waveform plot
    ax_waveform.set_xlim(visible_start, visible_end)
    ax_peak_diff.set_xlim(visible_start, visible_end)
    ax_peak_diff.set_ylabel('Time [ms]')
    ax_peak_diff.yaxis.set_label_position('right')
    ax_waveform.set_ylim(0, 1)
    scaling_factor = np.mean(peak_differences[visible_peaks])
    resolution = 0.01
    ax_peak_diff.set_ylim((1-resolution)*scaling_factor, (1+resolution)*scaling_factor)
    plt.subplots_adjust(bottom=0.25)
    plt.draw()

'''
def on_key(event):
    """Handler for key press events."""
    if event.key == 'left':
        # Decrease slider value
        if event.inaxes == ax_slider_centered:
            slider_centered.set_val(slider_centered.val - slider_centered.valstep)
            fig_centered.canvas.draw_idle()
        elif event.inaxes == ax_slider_peakdiff:
            slider_peakdiff.set_val(slider_peakdiff.val - slider_peakdiff.valstep)
            fig_peakdiff.canvas.draw_idle()
        elif event.inaxes == ax_slider_waveform:
            slider_waveform.set_val(slider_waveform.val - slider_waveform.valstep)
            fig_waveform.canvas.draw_idle()
    elif event.key == 'right':
        # Increase slider value
        if event.inaxes == ax_slider_centered:
            slider_centered.set_val(slider_centered.val + slider_centered.valstep)
            fig_centered.canvas.draw_idle()
        elif event.inaxes == ax_slider_peakdiff:
            slider_peakdiff.set_val(slider_peakdiff.val + slider_peakdiff.valstep)
            fig_peakdiff.canvas.draw_idle()
        elif event.inaxes == ax_slider_waveform:
            slider_waveform.set_val(slider_waveform.val + slider_waveform.valstep)
            fig_waveform.canvas.draw_idle()
'''

if __name__ == '__main__':
    main()
