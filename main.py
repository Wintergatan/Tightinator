#!/usr/bin/python3

import wave
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import LinearAxis, Range1d
from bokeh.layouts import row, column
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
npeaks = ''
nbins = ''


parser = argparse.ArgumentParser(description='Map transient times')
parser.add_argument('-f', '--file', dest='filename', type=str, action='store', help='File to open')
parser.add_argument('-o', '--out', dest='output_filename', type=str, action='store', help='Filename to write output values to')
parser.add_argument('-t', '--threshold', dest='thresh', default='0.25', type=float, action='store', help='DEFAULT=0.25 Peak detection threshold, lower is rougher')
parser.add_argument('-c', '--number-channels', dest='num_channels', type=int, action='store', help='DEFAULT=3 Number of channels, 2=MONO, 3=STEREO, etc')
parser.add_argument('-s', '--channel-offset', dest='off_channel', type=int, action='store', help='DEFAULT=2 Channel offset, channel to analyze.')
parser.add_argument('-e', '--envelope-smoothness', dest='envelope_smoothness', default='100', type=int, action='store', help='DEFAULT=100 Amount of rounding around the envelope')
parser.add_argument('-x', '--exclusion', dest='exclusion', default='30', type=int, action='store', help='DEFAULT=30 Exclusion threshold')
parser.add_argument('-p', '--precision', dest='float_prec', default='6', type=int, action='store', help='DEFAULT=6 Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438')
parser.add_argument('-n', '--npeaks', dest='npeaks', default='3', type=int, action='store', help='DEFAULT=3 Number of valid Peaks from which the leftmost is selected for better lining up between transients.')
parser.add_argument('-b', '--bins', dest='nbins', default='9', type=int, action='store', help='DEFAULT=9 Number of Bins used for the gaussian curve.')

parser.add_argument('-v', '--verbose', help="Set debug logging", action='store_true')

args = parser.parse_args()




def main():

    if args.verbose:
        print(args)
        # Set logging level - https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    nbins = 9
    npeaks = 3
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

    numchannel = 2

    amplitude_data = amplitude_data[::numchannel]

    normalized_amplitude = amplitude_data / np.max(np.abs(amplitude_data))
    normalized_amplitude = replace_negatives_with_neighbors(normalized_amplitude)
    envel = create_envelope(np.abs(normalized_amplitude),envelope_smoothness);
    norm_envelope = envel / np.max(np.abs(envel))
    # Find peak maxima above the threshold
    peaks_roughly, _ = find_peaks(norm_envelope, prominence=threshold,width = exclusion)
    #print(peaks_roughly)
    logging.info("Found {} rough peaks, refining...".format(len(peaks_roughly)))
    time = np.arange(0, len(normalized_amplitude)) / (frame_rate/4) * 1000
    peaks = []
    peaktimes = []
    for peak in peaks_roughly:
        search_range = 500
        max_value, max_index = find_maximum_around_peak(np.abs(normalized_amplitude), peak, search_range,npeaks)
        #max_time, max_index, max_value = find_fwhm_center(time,np.abs(normalized_amplitude),peak,search_range)
        if(len(peaks) >0):
            if(peaks[-1] != max_index):
                peaks.append(max_index)
                peaktimes.append(time[max_index])
        else:
            peaks.append(max_index)
            peaktimes.append(time[max_index])
    #print(peaks)
    #print(peaktimes)
    peaks = np.array(peaks)
    logging.info("Refined to {} peaks, calculating times...".format(len(peaks)))
    timearray = peaktimes
    differences = np.diff(timearray)
    differences = np.append(differences, 0)

    signal = normalized_amplitude
    segment_width = 1000

    diffs = differences[:-1]
    diff_std = []
    for i in range(len(diffs)-100):
        diff_std.append(np.std(diffs[i:i+99]))
    start_of_best_series=np.argmin(diff_std)
    
    best_peaks = peaks[start_of_best_series:start_of_best_series+99]
    best_series_times = timearray[start_of_best_series:start_of_best_series+100]
    best_diffs = diffs[start_of_best_series:start_of_best_series+99]
    best_series_amps = signal[start_of_best_series:start_of_best_series+99]
    
    combined_array = np.column_stack((timearray, differences))
    output_filename = filename[:-4]+".csv"
    logging.info("Saving output values to {}".format(output_filename))
    np_fmt = "%1.{}f".format(float_prec)
    np.savetxt(output_filename, combined_array, delimiter=",", header="Times[ms],differences[ms]", fmt=np_fmt, comments="")
    
    full_width = 2000
    plot_height = 600
    
    fig_center = figure(title='Similarness plot - most consistent Beast', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=int(np.floor(full_width/2)), height=plot_height)
    fig_center.output_backend = 'webgl'
    center_fig = plot_centered(fig_center,signal,time, best_peaks, best_series_amps)
    logging.info("center_plot figure created")
    
    fig_peakdiff = figure(title='Tightness plot - most consistent Beast', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
    fig_peakdiff.output_backend = 'webgl'
    peakdiff_fig = plot_peakdiff(fig_peakdiff,signal,time,best_peaks)
    logging.info("peakdiff figure created")
    
    fig_waveform = figure(title='Consistency/Waveform plot', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
    fig_waveform.output_backend = 'webgl'
    waveform_fig = plot_waveform(fig_waveform,signal,time,peaks,peaktimes,frame_rate,best_series_times)
    logging.info("waveform figure created")
    
    differences = np.diff(timearray)
    peak_amp = normalized_amplitude[peaks]
    fig_stat = figure(title='Statistics plot - most consistent Beast', x_axis_label='Transient Time difference [ms]', y_axis_label='Number of Elements in Bin', width=int(np.floor(full_width/2)), height=plot_height)
    fig_stat.output_backend = 'webgl'
    stat_fig = plot_stat(fig_stat,best_diffs,peak_amp,nbins)
    logging.info("stat figure created")
    
    layout = column(waveform_fig, peakdiff_fig, row(fig_center,fig_stat))
    output_file("summary.html", title="Summary Page")
    show(layout)

def plot_centered(fig, signal, time, peaks, best_series_amps):

    zoomval = 400
    log_val = np.exp(zoomval / 100)

    # Update centered segments plot here
    segment_width = int(log_val)
    cutoff = 0.01
    for peak in peaks:
        start = max(1, peak - segment_width)
        end = min(len(time), peak + segment_width)
        segment = signal[start:end]
        centered_x = time[start:end] - time[peak]
        fig.line(centered_x[segment >cutoff], segment[segment>cutoff], alpha=0.5, legend_label='centered Waveform')
        fig.circle(0, signal[peak], size=10, fill_color='red', legend_label='detected peak')
    fig.x_range.start = min(centered_x[segment >cutoff])
    fig.x_range.end = max(centered_x[segment >cutoff])
    fig.y_range.start = 0
    fig.y_range.end = 1
    return fig
    
    
def plot_peakdiff(fig, signal, time, peaks):
    cutoff = 0.01
    diff_times = []
    for i in range(len(peaks) - 2):
        start = peaks[i]  # Start index at the current peak
        end = peaks[i + 2]  # End index at the next peak
        segment = signal[start:end]
        peakdiff_x = time[start:end] - time[start]  # Adjust x-axis values relative to the start
        diff_times.append(time[peaks[i+1] - peaks[i]])
        fig.line(peakdiff_x[segment >cutoff], segment[segment >cutoff], alpha=0.5, legend_label='Peak Waveform')
        fig.circle(diff_times[i], signal[peaks[i+1]], size=10, fill_color='red', legend_label='detected Peak') 
    zoomfact = 5
    data_center = round(np.mean(diff_times))
    segment_width = zoomfact*round(np.std(diff_times))
    fig.x_range.start = data_center - segment_width
    fig.x_range.end = data_center + segment_width
    fig.y_range.start = 0
    fig.y_range.end = 1
    return fig
    
def plot_waveform(fig, signal, time, peaks,peaktimes,frame_rate,best_series_times):
    cutoff = 0.01

    signal_cut = signal[signal >cutoff]
    time_cut = time[signal >cutoff]/1000
    time_xax = time/1000
    visible_start = 0
    visible_end = max(time_xax)
    visible_indices = np.where((time_xax >= visible_start) & (time_xax <= visible_end))
    fig.line(time_cut, signal_cut, legend_label='Waveform')
    fig.circle(time_xax[peaks], signal[peaks], legend_label='Detected Peaks', color = 'red')
    fig.y_range = Range1d(start=0, end=1)
    peak_differences = np.diff(peaktimes)
    peak_differences_best = np.diff(best_series_times)
    #peak_differences_bpm = np.where(peak_differences == 0, -1, peak_differences) # a very dirty fix
    peak_bpm = (60*1000)/peak_differences
    peak_bpm_best = (60*1000)/peak_differences_best
    x_coordinate = best_series_times[0]/1000

    fig.line(x=[x_coordinate,x_coordinate], y=[0,1], line_width=2, line_dash="dashed", line_color="black", legend_label= 'Segment of most consistent Beats')
    x_coordinate = max(best_series_times)/1000

    fig.line(x=[x_coordinate,x_coordinate], y=[0,1], line_width=2, line_dash="dashed", line_color="black")
    peak_middles = ((time_xax[peaks[:-1]]+time_xax[peaks[1:]])/2)
    resolution = 0.01
    visible_peaks = np.where((peak_middles >= visible_start) & (peak_middles <= visible_end))
    diff_mean = np.mean(peak_bpm_best)
    diff_stdev = np.std(peak_bpm_best)
    zoom_factor = 10
    fig.extra_y_ranges = {"peak_diff_range": Range1d(diff_mean-zoom_factor*diff_stdev, diff_mean+zoom_factor*diff_stdev)}
    fig.add_layout(LinearAxis(y_range_name="peak_diff_range", axis_label="BPM [Hz]"), 'right')  # Add the right y-axis
    fig.vbar(x=peak_middles, top=peak_bpm, width=1/peak_bpm, y_range_name="peak_diff_range", color = 'green', fill_alpha=0.75, legend_label='BPM')

    #fig.extra_y_ranges = {"peak_diff_range": Range1d(start=(1-resolution)*scaling_factor, end=(1+resolution)*scaling_factor)}



    fig.x_range.start = visible_start
    fig.x_range.end = visible_end
    y_range_start = 0  # Define the start value for the y-axis range on the left
    y_range_end = 1   # Define the end value for the y-axis range on the left
    fig.y_range = Range1d(start=y_range_start, end=y_range_end)  # Set the y-range of the left y-axis
    return fig

    
def plot_stat(fig, x_data, y_data,nbins):
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
    num_bins = nbins
    hist, bin_edges = np.histogram(x_data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_bins = np.mean(bin_centers)
    std_bins = np.std(bin_centers)
    binsize=bin_centers[1]-bin_centers[0]
    max_count_index = np.argmax(hist)
    num_elements_in_highest_bin = hist[max_count_index]
    fig.quad(top=hist, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], fill_color="blue", line_color="white", alpha=0.7, legend_label='Number of Elements per Bin')
    row_spacing = max(x_gaussian)/(num_elements_in_highest_bin)
    fig.y_range = Range1d(start=0, end=num_elements_in_highest_bin)
    fig.circle(x_data,np.zeros(len(x_data))+0.05, size=10, fill_color='red', legend_label='Transient Time Difference')
    
    fig.extra_y_ranges = {"gaussian_range": Range1d(start=0, end=max(x_gaussian))}
    fig.add_layout(LinearAxis(y_range_name="gaussian_range", axis_label="Probability Density [a.u.]"), 'right')  # Add the right y-axis
    fig.line(x_curve, x_gaussian, y_range_name="gaussian_range", color= 'red', line_width = 2.5, legend_label='Gaussian distribution')
    return fig

    
def find_maximum_around_peak(data, peak_location, search_range, npeaks):
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
    numpeaks = int(npeaks)
    start_index = max(0, peak_location - search_range)
    end_index = min(len(data), peak_location + search_range + 1)
    max_values = np.partition(data[start_index:end_index], -numpeaks)[-numpeaks:]  # Get the two highest values
    max_indices = np.where(np.isin(data[start_index:end_index], max_values))[0] + start_index
    max_index = min(max_indices)  # Select the leftmost index
    return max_values[0], max_index


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
            new_lst[i] = np.abs(lst[i])
    return new_lst



if __name__ == '__main__':
    main()


