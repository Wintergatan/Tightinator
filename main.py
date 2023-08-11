#!/usr/bin/python3

import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import LinearAxis, Range1d, Label
from bokeh.layouts import row, column
from scipy import integrate
from scipy.io import wavfile
from scipy.signal import find_peaks
import argparse
import logging

filename = ''
output_filename = ''
downsample_rate = ''
thresh = ''
channel = ''
envelope_smoothness = ''
exclusion = ''
float_prec = ''
verbose = ''
npeaks = ''
nbins = ''
len_series = ''
web_mode = False
work_dir = ''
x_wide = ''
y_high = ''

parser = argparse.ArgumentParser(description='Map transient times')
parser.add_argument('-f', '--file', dest='filename', type=str, action='store', help='File to open')
parser.add_argument('-o', '--out', dest='output_filename', type=str, action='store', help='Filename to write output values to')
parser.add_argument('-d', '--downsample-rate', dest='downsample_rate', default='8', type=int, action='store', help='DEFAULT=8 Amount by which to reduce resolution. Higher resolution means longer compute.')
parser.add_argument('-t', '--threshold', dest='thresh', default='0.25', type=float, action='store', help='DEFAULT=0.25 Peak detection threshold, lower is rougher.')
parser.add_argument('-c', '--channel', dest='channel', default='1', type=int, action='store', help='DEFAULT=1 Channel to get the Waveform from.')
parser.add_argument('-en', '--envelope-smoothness', dest='envelope_smoothness', default='100', type=int, action='store', help='DEFAULT=100 Amount of rounding around the envelope.')
parser.add_argument('-ex', '--exclusion', dest='exclusion', default='30', type=int, action='store', help='DEFAULT=30 Exclusion threshold.')
parser.add_argument('-r', '--precision', dest='float_prec', default='6', type=int, action='store', help='DEFAULT=6 Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438')
parser.add_argument('-p', '--number-peaks', dest='npeaks', default='3', type=int, action='store', help='DEFAULT=3 Number of valid peaks from which the left-most is selected for better lining up between transients.')
parser.add_argument('-b', '--bins', dest='nbins', default='0', type=int, action='store', help='DEFAULT=0 Number of bins used for the gaussian curve.')
parser.add_argument('-l', '--length', dest='len_series', default='100', type=int, action='store', help='DEFAULT=100 The length of the series of most consistent beats.')
parser.add_argument('--work-dir', dest='work_dir', action='store', help='Directory structure to work under.' )
parser.add_argument('-x', '--x-width', dest='x_wide', default='2000', type=int, action='store', help='DEFAULT=2000 Fixed width for graphs.')
parser.add_argument('-y', '--plot-height', dest='y_high', default='600', type=int, action='store', help='DEFAULT=600 Fixed height for single plot.')
parser.add_argument('-v', '--verbose', help="Set debug logging", action='store_true')

args = parser.parse_args()

def main():

    logging.info(args)

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
    downsamplerate = args.downsample_rate
    #output_filename = args.output_filename
    threshold = args.thresh
    channel = args.channel
    float_prec = args.float_prec
    nbins = args.nbins
    npeaks = args.npeaks
    len_series = args.len_series
    web_mode = args.web_mode
    full_width = args.x_wide
    plot_height = args.y_high
    
    if(nbins == 0):
        nbins = int(1 + (3.322 * np.log(len_series)))

    # If output_filename argument not set use the uploaded filename + .csv
    if not args.output_filename:
        output_filename = filename[:-4]+".csv"
    else:
        output_filename = args.output_filename



    # Open wav file

    frame_rate, data = wavfile.read(filename)
    num_channels = data.shape[1] if len(data.shape) > 1 else 1
    if(channel > num_channels):
        channel = num_channels-1
    if(num_channels > 1):
        amplitude_data = data[:,channel-1] # First channel has to be 1, only programmers know things start at 0
    else:
        amplitude_data = data
    amplitude_data = amplitude_data[::downsamplerate]
    timefactor = (frame_rate/downsamplerate)/1000


    normalized_amplitude = amplitude_data / np.max(np.abs(amplitude_data))
    normalized_amplitude = replace_negatives_with_neighbors(normalized_amplitude)
    envel = create_envelope(np.abs(normalized_amplitude),envelope_smoothness);
    norm_envelope = envel / np.max(np.abs(envel))
    # Find peak maxima above the threshold
    peaks_roughly, _ = find_peaks(norm_envelope, prominence=threshold,width = exclusion)
    #print(peaks_roughly)
    logging.info("Found {} rough peaks, refining...".format(len(peaks_roughly)))
    time = np.arange(0, len(normalized_amplitude))/timefactor
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
    if(len(peaks) < len_series):
        len_series = len(peaks)
    #print(peaks)
    #print(peaktimes)
    peaks = np.array(peaks)
    logging.info("Refined to {} peaks, calculating times...".format(len(peaks)))

    timearray = peaktimes
    differences = np.diff(timearray)
    accel = np.gradient(differences)    
    differences = np.append(differences, 0)
    stdev = np.zeros(len(peaks))
 
    accel= np.append(accel, 0)
    signal = normalized_amplitude
    segment_width = 1000

    diffs = differences[:-1]
    diff_std = []
    for i in range(len(diffs)-(len_series-2)):
        diff_std.append(np.std(diffs[i:i+len_series]))
    start_of_best_series=np.argmin(diff_std)
    
    best_peaks = peaks[start_of_best_series:start_of_best_series+len_series]
    best_series_times = timearray[start_of_best_series:start_of_best_series+len_series]
    best_series_times_csv = np.pad(best_series_times,(0,len(peaks)-len_series),'constant')
    best_diffs = diffs[start_of_best_series:start_of_best_series+len_series]
    stdev[0] = np.std(best_diffs)
    stdev[1] = np.std(differences)
    stdev[3] = start_of_best_series
    stdev[4] = start_of_best_series+len_series
    best_series_amps = signal[best_peaks]
    combined_array = np.column_stack((timearray,best_series_times_csv,stdev))
    #output_filename = filename[:-4]+".csv"
    logging.info("Saving output values to {}".format(output_filename))
    np_fmt = "%1.{}f".format(float_prec)
    np.savetxt(output_filename, combined_array, delimiter=",", header="Times[ms],BestTimes[ms],stdevandmore[ms]", fmt=np_fmt, comments="")



    
    fig_center = figure(title='Similarness plot - most consistent Beats', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=int(np.floor(full_width/2)), height=plot_height)
    fig_center.output_backend = 'webgl'
    center_fig = plot_centered(fig_center,signal,time, best_peaks)
    logging.info("center_plot figure created")
    
 
#    fig_peakdiff = figure(title='Tightness plot - most consistent Beats', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
#    fig_peakdiff.output_backend = 'webgl'
#    peakdiff_fig = plot_peakdiff(fig_peakdiff,signal,time,best_peaks)
#    logging.info("peakdiff figure created")
    
    fig_waveform = figure(title='Consistency/Waveform plot', x_axis_label='Time [s]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
    fig_waveform.output_backend = 'webgl'
    waveform_fig = plot_waveform(fig_waveform,signal,time,peaks,peaktimes,frame_rate,best_series_times)
    logging.info("waveform figure created")
    
    fig_stat = figure(title='Statistics plot - most consistent Beats', x_axis_label='Transient Time difference [ms]', y_axis_label='Number of Elements in Bin', width=int(np.floor(full_width/2)), height=plot_height)
    fig_stat.output_backend = 'webgl'
    stat_fig = plot_stat(fig_stat,best_series_times,best_series_amps,nbins)
    logging.info("stat figure created")
    
    layout = column(waveform_fig, row(fig_center, stat_fig))
    output_file("summary.html", title="Summary Page")
    show(layout)

def plot_centered(fig, signal, time, peaks):

    zoomval = 400
    log_val = np.exp(zoomval / 100)

    # Update centered segments plot here
    segment_width = int(log_val)
    cutoff = 0
    maxheight = 1
    for peak in peaks:
        start = max(1, peak - segment_width)
        end = min(len(time), peak + segment_width)
        segment = signal[start:end]
        centered_x = time[start:end] - time[peak]
        fig.line(centered_x[segment >cutoff], (segment[segment>cutoff])/signal[peak], alpha=0.5, legend_label='centered Waveform')
        newmax = max((segment[segment>cutoff])/signal[peak])
        if(newmax > maxheight):
            maxheight = newmax
    fig.circle(0, 1, size=10, fill_color='red', legend_label='detected peaks')
    fig.x_range.start = min(centered_x[segment >cutoff])
    fig.x_range.end = max(centered_x[segment >cutoff])
    fig.y_range.start = 0
    fig.y_range.end = maxheight + 0.05
    fig.xaxis.ticker.num_minor_ticks = 9
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
    fig.xaxis.ticker.num_minor_ticks = 9
    return fig
    
def plot_waveform(fig, signal, time, peaks,peaktimes,frame_rate,best_series_times):
    cutoff = 0.01

    signal_cut = signal[signal >cutoff]
    time_cut = time[signal >cutoff]/1000
    time_xax = time/1000
    visible_start = 0
    visible_end = max(time_xax)
    visible_indices = np.where((time_xax >= visible_start) & (time_xax <= visible_end))
    fig.y_range = Range1d(start=0, end=1)
    peak_differences = np.diff(peaktimes)
    peak_differences_best = np.diff(best_series_times)
    #peak_differences_bpm = np.where(peak_differences == 0, -1, peak_differences) # a very dirty fix
    peak_bpm = (60*1000)/peak_differences
    peak_bpm_best = (60*1000)/peak_differences_best
    peak_middles = ((time_xax[peaks[:-1]]+time_xax[peaks[1:]])/2)
    resolution = 0.01
    visible_peaks = np.where((peak_middles >= visible_start) & (peak_middles <= visible_end))
    diff_mean = np.mean(peak_bpm_best)
    diff_stdev = np.std(peak_bpm_best)
    accel = np.gradient(peak_bpm)
    norm_accel = accel/max(accel)
    accel_best = np.gradient(peak_bpm_best)
    norm_accel_best = accel/max(accel_best)
    zoom_factor = 10
    fill_color = []
    for acceleration in accel:
        if acceleration < 0:
            fill_color.append('darkred')  # Red for negative acceleration
        else:
            fill_color.append('darkorange')  # Orange for positive acceleration
    fig.extra_y_ranges = {"peak_diff_range": Range1d(max(0,diff_mean-zoom_factor*diff_stdev), diff_mean+zoom_factor*diff_stdev)}
    fig.add_layout(LinearAxis(y_range_name="peak_diff_range", axis_label="BPM [Hz]"), 'right')  # Add the right y-axis
    fig.vbar(x=peak_middles, top=peak_bpm, width=(peak_differences/1000)*0.9, y_range_name="peak_diff_range", color = 'green', fill_alpha=1, legend_label='BPM')
    #fig.line(x=peak_middles,y=(norm_accel_best*0.10)+0.5, line_color="darkgoldenrod", legend_label= 'acceleration of BPM', line_width=3)
    #fig.line(x=[0,len(time)],y=[0.5,0.5], line_color="black", line_dash="dotted")
    #fig.vbar(x=peak_middles, bottom=peak_bpm + (np.abs(accel)*-1),top=peak_bpm , width=(peak_differences/1000)*0.5, y_range_name="peak_diff_range", color = fill_color, fill_alpha=1, legend_label='BPM Acceleration')    
    fig.line(time_cut, signal_cut, legend_label='Waveform')
    fig.circle(time_xax[peaks], signal[peaks], legend_label='Detected Peaks', color = 'red')
    accel_start = max(0,diff_mean-zoom_factor*diff_stdev)
    fig.vbar(x=peak_middles, bottom=accel_start,top=accel_start + (np.abs(accel)) , width=(peak_differences/1000)*0.5, y_range_name="peak_diff_range", color = fill_color, fill_alpha=1, legend_label='BPM Acceleration')    
    x_coordinate = best_series_times[0]/1000

    fig.line(x=[x_coordinate,x_coordinate], y=[0,1], line_width=2, line_dash="dashed", line_color="black", legend_label= 'Segment of most consistent Beats')
    x_coordinate = max(best_series_times)/1000

    fig.line(x=[x_coordinate,x_coordinate], y=[0,1], line_width=2, line_dash="dashed", line_color="black")
    
    #fig.extra_y_ranges = {"peak_diff_range": Range1d(start=(1-resolution)*scaling_factor, end=(1+resolution)*scaling_factor)}



    fig.x_range.start = visible_start
    fig.x_range.end = visible_end
    y_range_start = 0  # Define the start value for the y-axis range on the left
    y_range_end = 1   # Define the end value for the y-axis range on the left
    fig.y_range = Range1d(start=y_range_start, end=y_range_end)  # Set the y-range of the left y-axis
    fig.xaxis.ticker.num_minor_ticks = 9
    return fig

    
def plot_stat(fig, peak_times, y_data,nbins):
    
    
    x_data = np.diff(peak_times)
    mean_x = np.mean(x_data)
    std_x = np.std(x_data)
    
    stddeviations = 5
    max_dist= max(abs(mean_x-min(x_data)),abs(mean_x+max(x_data)))
    x_curve = np.linspace(mean_x-stddeviations*std_x, mean_x+stddeviations*std_x, 1000)
    y_curve = np.linspace(min(y_data)-50, max(y_data)+50, 1000)
    
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
    peakamps =y_data[1:]
    
    fig.extra_y_ranges = {"gaussian_range": Range1d(start=0, end=max(x_gaussian))}
    fig.add_layout(LinearAxis(y_range_name="gaussian_range", axis_label="Probability Density [a.u.]"), 'right')  # Add the right y-axis
    fig.line(x_curve, x_gaussian, y_range_name="gaussian_range", color= 'red', line_width = 2.5, legend_label='Gaussian distribution')
    fig.circle(x_data,(peakamps / np.max(np.abs(peakamps)))*(num_elements_in_highest_bin-1), size=10, fill_color='red', legend_label='Peak Transient Time')

    fig.xaxis.ticker.num_minor_ticks = 9
    xshift = 0
    
    text_annotation1 = Label(x=(mean_x-(stddeviations-xshift)*std_x), y=(num_elements_in_highest_bin)*0.94, text="standard deviation = "+f"{std_x:.2f}"+" ms", text_font_size="20pt")
    text_annotation2 = Label(x=(mean_x-(stddeviations-xshift)*std_x), y=(num_elements_in_highest_bin)*0.88, text="mean = "+f"{mean_x:.2f}"+" ms", text_font_size="20pt")
    fig.add_layout(text_annotation1)
    fig.add_layout(text_annotation2)
    
    fig.x_range.start = mean_x-(stddeviations)*std_x
    fig.x_range.end = mean_x+(stddeviations)*std_x
    
    
    
    
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


