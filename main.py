#!/usr/bin/python3

import wave
import numpy as np
from bokeh.plotting import figure, show, output_file, reset_output
from bokeh.models import LinearAxis, Range1d
from scipy import integrate
from scipy.signal import find_peaks
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

    #

    combined_array = np.column_stack((timearray, differences))

    output_filename = filename[:-4]+".csv"
    logging.info("Saving output values to {}".format(output_filename))
    np_fmt = "%1.{}f".format(float_prec)
    np.savetxt(output_filename, combined_array, delimiter=",", header="Times[ms],differences[ms]", fmt=np_fmt, comments="")
    
    fig_center = figure(title='Similarness plot', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]')
    fig_center.output_backend = 'webgl'
    plot_centered(fig_center,signal,time,peaks)
    logging.info("center_plot exported")
    
    fig_peakdiff = figure(title='Tightness plot', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]')
    fig_peakdiff.output_backend = 'webgl'
    plot_peakdiff(fig_peakdiff,signal,time,peaks)
    logging.info("peakdiff exported")
    
    fig_waveform = figure(title='Consistency/Waveform plot', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]')
    fig_waveform.output_backend = 'webgl'
    plot_waveform(fig_waveform,signal,time,peaks,frame_rate)
    logging.info("waveform exported")
    
    differences = np.diff(timearray)
    peak_amp = normalized_amplitude[peaks]
    fig_stat = figure(title='stat plot', x_axis_label='Transient Time difference[ms]', y_axis_label='Number of elements in Bin')
    fig_stat.output_backend = 'webgl'
    plot_stat(fig_stat,differences[:-1],peak_amp)
    logging.info("stat exported")

def plot_centered(fig, signal, time, peaks):

    zoomval = 400
    log_val = np.exp(zoomval / 100)

    # Update centered segments plot here
    segment_width = int(log_val)
    
    for peak in peaks:
        start = max(1, peak - segment_width)
        end = min(len(time), peak + segment_width)
        segment = signal[start:end]
        centered_x = time[start:end] - time[peak]
        fig.line(centered_x, segment)
        fig.circle(0, signal[peak], size=10, fill_color='red')

    #reset_output()
    output_file("centered_plot.html")
    show(fig)
    
    
def plot_peakdiff(fig, signal, time, peaks):
    zoomval = 400
    log_val = np.exp(zoomval / 100)

    # Update peakdiff segments plot here
    segment_width = int(log_val)

    
    for i in range(len(peaks) - 2):
        start = peaks[i]  # Start index at the current peak
        end = peaks[i + 2]  # End index at the next peak
        segment = signal[start:end]
        peakdiff_x = time[start:end] - time[start]  # Adjust x-axis values relative to the start
        fig.line(peakdiff_x, segment)
        fig.circle(time[peaks[i+1] - peaks[i]], signal[peaks[i+1]], size=10, fill_color='red') 

    data_center = round(np.mean(time[peaks[i+1] - peaks[i]]))
    fig.x_range.start = data_center - segment_width
    fig.x_range.end = data_center + segment_width
    #reset_output()
    output_file("peakdiff_plot.html")
    show(fig)
    
def plot_waveform(fig, signal, time, peaks,frame_rate):
    visible_start = 0
    visible_end = max(time)
    visible_indices = np.where((time >= visible_start) & (time <= visible_end))
    fig.line(time, signal, legend_label='Waveform')
    fig.circle(time[peaks], signal[peaks], legend_label='Waveform', color = 'red')
    fig.y_range = Range1d(start=0, end=1)
    peak_differences = np.diff(peaks/frame_rate*1000)
    peak_middles = ((time[peaks[:-1]]+time[peaks[1:]])/2)
    resolution = 0.01

    visible_peaks = np.where((peak_middles >= visible_start) & (peak_middles <= visible_end))
    scaling_factor = np.mean(peak_differences[visible_peaks])
    fig.extra_y_ranges = {"peak_diff_range": Range1d(start=(1-resolution)*scaling_factor, end=(1+resolution)*scaling_factor)}
    fig.add_layout(LinearAxis(y_range_name="peak_diff_range", axis_label="Time between transients [ms]"), 'right')  # Add the right y-axis
    fig.vbar(x=peak_middles, top=peak_differences, width=np.mean(peak_differences), y_range_name="peak_diff_range", color = 'green', fill_alpha=0.5, legend_label='transient differences')
    #fig.extra_y_ranges = {"peak_diff_range": Range1d(start=(1-resolution)*scaling_factor, end=(1+resolution)*scaling_factor)}



    fig.x_range.start = visible_start
    fig.x_range.end = visible_end
    y_range_start = 0  # Define the start value for the y-axis range on the left
    y_range_end = 1   # Define the end value for the y-axis range on the left
    fig.y_range = Range1d(start=y_range_start, end=y_range_end)  # Set the y-range of the left y-axis

    #reset_output()
    output_file("waveform_plot.html")
    show(fig)
    
def plot_stat(fig, x_data, y_data):
    mean_x = np.mean(x_data)
    std_x = np.std(x_data)
    
    stddeviations = 5
    max_dist= max(abs(mean_x-min(x_data)),abs(mean_x+max(x_data)))
    x_curve = np.linspace(mean_x-stddeviations*std_x, mean_x+stddeviations*std_x, 1000)
    y_curve = np.linspace(min(y_data), max(y_data), 1000)
    
    # Calculate the unnormalized Gaussian values
    x_gaussian = np.exp(-0.5 * ((x_curve - mean_x) / std_x)**2) / (std_x * np.sqrt(2 * np.pi))
    
    #x_gaussian = x_gaussian * x_normalization_factor  # Corrected line
    area_under_curve= np.trapz(x_gaussian, x_curve)
    #y_gaussian = np.exp(-0.5 * ((y_curve - mean_y) / std_y)**2) / (std_y * np.sqrt(2 * np.pi))
    x_gaussian = x_gaussian / area_under_curve
    # Plot the curves
    num_bins = 6
    hist, bin_edges = np.histogram(x_data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_bins = np.mean(bin_centers)
    std_bins = np.std(bin_centers)
    binsize=bin_centers[1]-bin_centers[0]
    max_count_index = np.argmax(hist)
    num_elements_in_highest_bin = hist[max_count_index]
    fig.quad(top=hist, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], fill_color="blue", line_color="white", alpha=0.7)
    row_spacing = max(x_gaussian)/(num_elements_in_highest_bin)
    fig.y_range = Range1d(start=0, end=num_elements_in_highest_bin)
    fig.circle(x_data,np.zeros(len(x_data))+0.05, size=10, fill_color='red')
    
    fig.extra_y_ranges = {"gaussian_range": Range1d(start=0, end=max(x_gaussian))}
    fig.add_layout(LinearAxis(y_range_name="gaussian_range", axis_label="Probability Density [a.u.]"), 'right')  # Add the right y-axis
    fig.line(x_curve, x_gaussian, y_range_name="gaussian_range",)
    
    output_file("stat_plot.html")
    show(fig)

    
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


