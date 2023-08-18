#!/usr/bin/python3

import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import LinearAxis, Range1d, Label
from bokeh.layouts import row, column
from scipy import integrate
from scipy.io import wavfile
from scipy.signal import find_peaks, correlate
from scipy.interpolate import LSQUnivariateSpline
import argparse
import logging
from skimage.measure import block_reduce

filename = ''
output_filename = ''
downsample_rate = ''
thresh = ''
channel = ''
exclusion = ''
float_prec = ''
verbose = ''
len_series = ''
work_dir = ''
web_mode = False
x_wide = ''
y_high = ''
bpm_target = ''
bpm_window = ''
klick = ''


parser = argparse.ArgumentParser(description='Map transient times')
parser.add_argument('-f', '--file', dest='filename', type=str, action='store', help='File to open')
parser.add_argument('-o', '--out', dest='output_filename', type=str, action='store', help='Filename to write output values to')
parser.add_argument('-d', '--downsample-rate', dest='downsample_rate', default='4', type=int, action='store', help='DEFAULT=4 Amount by which to reduce resolution. Higher resolution means longer compute.')
parser.add_argument('-t', '--threshold', dest='thresh', default='0.1', type=float, action='store', help='DEFAULT=0.1 Peak detection threshold. Works best 0.1 and above. Setting too high/low can cause misdetection.')
parser.add_argument('-c', '--channel', dest='channel', default='1', type=int, action='store', help='DEFAULT=1 Channel to get the waveform from.')
parser.add_argument('-ex', '--exclusion', dest='exclusion', default='3200', type=int, action='store', help='DEFAULT=3200 Minimum distance between peaks.')
parser.add_argument('-r', '--precision', dest='float_prec', default='6', type=int, action='store', help='DEFAULT=6 Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438')
parser.add_argument('-l', '--length', dest='len_series', default='100', type=int, action='store', help='DEFAULT=100 The length of the series of most consistent beats.')
parser.add_argument('-w', '--web', dest='web_mode', default=False, action='store_true', help='DEFAULT=False Get some width/height values from/ browser objects for graphing. Defaults false.')
parser.add_argument('-b', '--bpm-target', dest='bpm_target', default='0', type=float, action='store', help='DEFAULT=0 The target BPM of the song. 0 = Auto.')
parser.add_argument('-bw', '--bpm-window', dest='bpm_window', default='0', type=float, action='store', help='DEFAULT=0 Window of BPM that should be visible around the target. 0 = Auto.')
parser.add_argument('-a', '--algorithm', dest='klick', default='1', type=int, action='store', help='DEFAULT=1 Switch between peak detecting algorithm. 0 = Center, 1 = Right')
parser.add_argument('--work-dir', dest='work_dir', action='store', help='Directory structure to work under.' )
parser.add_argument('-x', '--x-width', dest='x_wide', default='2000', type=int, action='store', help='DEFAULT=2000 Fixed width for graphs.')
parser.add_argument('-y', '--plot-height', dest='y_high', default='1340', type=int, action='store', help='DEFAULT=600 Fixed height for single plot.')
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
    filename = args.filename
    #envelope_smoothness = args.envelope_smoothness
    downsamplerate = int(args.downsample_rate)
    exclusion = int(int(args.exclusion) / downsamplerate)
    #output_filename = args.output_filename
    threshold = args.thresh
    channel = args.channel
    float_prec = args.float_prec
    #nbins = args.nbins
    #npeaks = args.npeaks
    len_series = args.len_series
    full_width = args.x_wide - 15
    plot_height = args.y_high
    bpm_target = args.bpm_target    
    bpm_window = args.bpm_window
    klick = args.klick

    plot_height = int((plot_height-140)/2)

    nbins = int(1 + (3.322 * np.log(len_series)))

    # If output_filename argument not set use the uploaded filename + .csv
    if not args.output_filename:
        output_filename = filename[:-4]+".csv"
    else:
        output_filename = args.output_filename

    # If web mode add the work dir to the filenames
    if args.web_mode:
        if not args.work_dir:
            work_dir = 'static/upload/test/'
        else:
            work_dir = args.work_dir
            filename = work_dir + filename
            output_filename = work_dir + output_filename
            print("{}, {}".format(filename, output_filename))

    # Open wav file
    frame_rate, data = wavfile.read(filename)
    num_channels = data.shape[1] if len(data.shape) > 1 else 1
    if(channel > num_channels):
        channel = num_channels-1

    if(num_channels > 1):
        amplitude_data = data[:,channel-1] # First channel has to be 1, only programmers know things start at 0
    else:
        amplitude_data = data
    #amplitude_data = resample(amplitude_data, int(len(amplitude_data)/downsamplerate))
    #amplitude_data = amplitude_data[::downsamplerate]
    reduction_factor = (downsamplerate,)
    amplitude_data = block_reduce(amplitude_data,reduction_factor , np.mean)
    timefactor = (frame_rate/downsamplerate)/1000


    normalized_amplitude = np.abs(amplitude_data / np.max(np.abs(amplitude_data)))
    time = np.arange(0, len(normalized_amplitude))/timefactor
    #envel = create_envelope(time,np.abs(normalized_amplitude),envelope_smoothness)
    #norm_envelope = envel / np.max(np.abs(envel))
    # Find peak maxima above the threshold
    peaks_roughly, _ = find_peaks(normalized_amplitude, prominence=threshold,distance = exclusion)
    #peaks_roughly, _ = find_peaks(norm_envelope, prominence=threshold,distance = exclusion)
    #print(peaks_roughly)
    logging.info("Found {} rough peaks, refining...".format(len(peaks_roughly)))
    peaks = []

    peaktimes = []

    for peak in peaks_roughly:
        search_range = int(400/downsamplerate)
        max_index = peakrefiner_center_of_weight(np.abs(normalized_amplitude), peak, search_range)
        if(klick):
            max_index = peakrefiner_find_peak_to_right(np.abs(normalized_amplitude), max_index, search_range)
        #max_time, max_index, max_value = find_fwhm_center(time,np.abs(normalized_amplitude),peak,search_range)
        if(len(peaks) >0):
            if(peaks[-1] != max_index):
                peaks.append(max_index)
                peaktimes.append(time[max_index])
        else:
            peaks.append(max_index)
            peaktimes.append(time[max_index])
    #peaks = correlator(np.abs(normalized_amplitude), peakspre, 50)
    #peaktimes = time[peaks]
    if(len(peaks) < len_series):
        len_series = len(peaks)
    #print(len(peaks))
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
    best_diffs_csv = np.pad(best_diffs,(0,len(peaks)-len_series),'constant')
    #print(len(best_peaks))
    stdev[0] = np.std(best_diffs)
    stdev[1] = np.std(differences)
    stdev[3] = start_of_best_series
    stdev[4] = start_of_best_series+len_series
    stdev[5] = len(peaks)
    stdev[7] = threshold
    best_series_amps = signal[best_peaks]
    combined_array = pad_and_stack_arrays([timearray,differences,best_series_times_csv,best_diffs_csv,stdev])
    #output_filename = filename[:-4]+".csv"
    logging.info("Saving output values to {}".format(output_filename))
    np_fmt = "%1.{}f".format(float_prec)
    np.savetxt(output_filename, combined_array, delimiter=",", header="PeakTimes[ms],PeakDifferences[ms],BestTimes[ms],BestDifferrences[ms],data", fmt=np_fmt, comments="")
    norm_envelope = []
    fig_center = figure(title='Similarness plot - most consistent Beats', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=int(np.floor(full_width/2)), height=plot_height)
    fig_center.output_backend = 'webgl'
    center_fig = plot_centered(fig_center,signal,time, best_peaks, norm_envelope,downsamplerate)
    logging.info("center_plot figure created")
    
    fig_waveform = figure(title='Consistency/Waveform plot', x_axis_label='Time [s]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
    fig_waveform.output_backend = 'webgl'
    waveform_fig = plot_waveform(fig_waveform,signal,time,peaks,peaktimes,frame_rate,best_series_times,threshold,bpm_target,bpm_window, norm_envelope)
    logging.info("waveform figure created")

    fig_stat = figure(title='Statistics plot - most consistent Beats', x_axis_label='Transient Time difference [ms]', y_axis_label='Probability density[1/ms]', width=int(np.floor(full_width/2)), height=plot_height)
    fig_stat.output_backend = 'webgl'
    stat_fig = plot_stat(fig_stat,best_series_times,best_series_amps,nbins)
    logging.info("stat figure created")

    layout = column(waveform_fig, row(fig_center, stat_fig))
    if args.web_mode:
        print("Writing graphs to {}summary.html".format(work_dir))
        output_file("{}summary.html".format(work_dir), title="Summary Page")
        save(layout)
    else:
        output_file("summary.html", title="Summary Page")
        show(layout)

def pad_and_stack_arrays(arrays):
    # Find the length of the longest array
    max_length = max(len(arr) for arr in arrays)
    
    # Pad shorter arrays and stack them
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in arrays]
    stacked_array = np.column_stack(padded_arrays)
    
    return stacked_array

def plot_centered(fig, signal, time, peaks, norm_envelope,downsamplerate):
    # Update centered segments plot here
    segment_width = int(400/downsamplerate)
    cutoff = -1
    maxheight = 1
    startvals = []
    endvals = []
    xs, ys, ys2, peakheights = [], [], [], []
    for peak in peaks:
        start = max(1, peak - segment_width)
        end = min(len(time), peak + segment_width)
        segment = signal[start:end]
        #segment2= norm_envelope[start:end]
        centered_x = time[start:end] - time[peak]
        xs.append( centered_x[segment >cutoff])
        ys.append((segment[segment>cutoff])/signal[peak])
        #ys2.append((segment2[segment>cutoff])/max(segment2))
        peakheights.append(signal[peak]/max(segment))
        newmax = max((segment[segment>cutoff])/signal[peak])
        if(newmax > maxheight):
            maxheight = newmax
        startvals.append(min(centered_x))
        endvals.append(max(centered_x))
    #colors = Category20[len(peaks)]
    
    fig.multi_line(xs, ys, alpha=0.5, legend_label='centered Waveform')
    #fig.multi_line(xs, ys2, alpha=0.5, legend_label='centered Waveform', color='orange')
    fig.circle(0, 1, size=10, fill_color='red', legend_label='detected peaks')
    fig.x_range.start = min(startvals)
    fig.x_range.end = max(endvals)
    fig.y_range.start = 0
    fig.y_range.end = min(maxheight + 0.05,2)
    fig.xaxis.ticker.num_minor_ticks = 9
    return fig

def plot_peakdiff(fig, signal, time, peaks):
    cutoff = 0.01
    diff_times = []
    xs, ys = [],[]
    for i in range(len(peaks) - 2):
        start = peaks[i]  # Start index at the current peak
        end = peaks[i + 2]  # End index at the next peak
        segment = signal[start:end]
        peakdiff_x = time[start:end] - time[start]  # Adjust x-axis values relative to the start
        diff_times.append(time[peaks[i+1] - peaks[i]])
        xs.append(peakdiff_x[segment >cutoff])
        ys.append(segment[segment >cutoff])
        fig.circle(diff_times[i], signal[peaks[i+1]], size=10, fill_color='red', legend_label='detected Peak') 
    
    fig.multi_line(xs, ys, alpha=0.5, legend_label='Peak Waveform')
    zoomfact = 5
    data_center = round(np.mean(diff_times))
    segment_width = zoomfact*round(np.std(diff_times))
    fig.x_range.start = data_center - segment_width
    fig.x_range.end = data_center + segment_width
    fig.y_range.start = 0
    fig.y_range.end = 1
    fig.xaxis.ticker.num_minor_ticks = 9
    return fig

    
def plot_waveform(fig, signal, time, peaks,peaktimes,frame_rate,best_series_times,sensitivity,bpm_target,bpm_window,envelope):
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
    if(not bpm_target):
        if(not bpm_window):
            secyrange_start = max(0,diff_mean-zoom_factor*diff_stdev)
            secyrange_end = diff_mean+zoom_factor*diff_stdev
        else:
            secyrange_start = max(0,diff_mean-bpm_window)
            secyrange_end = diff_mean+bpm_window
        accel_bottom = secyrange_start
        accel_top = accel_bottom + np.abs(accel)
    else:
        if(not bpm_window):
            secyrange_start = 0
            secyrange_end = bpm_target/0.75
        else:
            secyrange_start = bpm_target - bpm_window
            secyrange_end = bpm_target + bpm_window
        accel_bottom = 0
        accel_top = np.abs(accel)
    
    fig.extra_y_ranges = {"peak_diff_range": Range1d(secyrange_start, secyrange_end)}
    fig.add_layout(LinearAxis(y_range_name="peak_diff_range", axis_label="BPM [Hz]"), 'right')  # Add the right y-axis
    fig.vbar(x=peak_middles, top=peak_bpm, width=(peak_differences/1000)*0.9, y_range_name="peak_diff_range", color = 'green', fill_alpha=1, legend_label='BPM')
    #fig.line(x=peak_middles,y=(norm_accel_best*0.10)+0.5, line_color="darkgoldenrod", legend_label= 'acceleration of BPM', line_width=3)
    #fig.line(x=[0,len(time)],y=[0.5,0.5], line_color="black", line_dash="dotted")
    #fig.vbar(x=peak_middles, bottom=peak_bpm + (np.abs(accel)*-1),top=peak_bpm , width=(peak_differences/1000)*0.5, y_range_name="peak_diff_range", color = fill_color, fill_alpha=1, legend_label='BPM Acceleration')
    fig.circle(time_xax[peaks], signal[peaks], legend_label='Detected Peaks', color = 'red')
    #accel_start = max(0,diff_mean-zoom_factor*diff_stdev)
    fig.vbar(x=peak_middles, bottom=accel_bottom,top=accel_top , width=(peak_differences/1000)*0.5, y_range_name="peak_diff_range", color = fill_color, fill_alpha=1, legend_label='BPM Acceleration')
    fig.line(time_cut, signal_cut, legend_label='Waveform')
    #fig.line(time/1000, envelope, line_color="orange", legend_label='Envelope') 
    
    x_coordinate = best_series_times[0]/1000

    fig.line(x=[x_coordinate,x_coordinate], y=[0,1], line_width=2, line_dash="dashed", line_color="black", legend_label= 'Segment of most consistent Beats')
    x_coordinate = max(best_series_times)/1000

    fig.line(x=[x_coordinate,x_coordinate], y=[0,1], line_width=2, line_dash="dashed", line_color="black")

    fig.x_range.start = visible_start
    fig.x_range.end = visible_end
    y_range_start = 0  # Define the start value for the y-axis range on the left
    y_range_end = 1   # Define the end value for the y-axis range on the left
    fig.y_range = Range1d(start=y_range_start, end=y_range_end)  # Set the y-range of the left y-axis
    fig.xaxis.ticker.num_minor_ticks = 9
    text_annotation1 = Label(x=0, y=0, text="sensitivity = "+f"{sensitivity:.2f}", text_font_size="12pt", background_fill_color = "white")
    fig.add_layout(text_annotation1)
    
    return fig

def plot_stat(fig, peak_times, y_data,nbins):
    if(nbins == 0):
        nbins = int(1 + (3.322 * np.log(len_series)))
    x_data = np.diff(peak_times)
    mean_x = np.mean(x_data)
    std_x = np.std(x_data)

    stddeviations = 5
    max_dist= max(abs(mean_x-min(x_data)),abs(mean_x+max(x_data)))
    x_curve = np.linspace(mean_x-stddeviations*std_x, mean_x+stddeviations*std_x, 1000)
    y_curve = np.linspace(min(y_data)-50, max(y_data)+50, 1000)

    # Calculate the unnormalized Gaussian values
    x_gaussian = np.exp(-0.5 * ((x_curve - mean_x) / std_x)**2) / (std_x * np.sqrt(2 * np.pi))
    area_under_curve= np.trapz(x_gaussian, x_curve)
    x_gaussian = x_gaussian / area_under_curve
    # Plot the curves
    num_bins = nbins
    hist, bin_edges = np.histogram(x_data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_bins = np.mean(bin_centers)
    std_bins = np.std(bin_centers)
    binsize=bin_centers[1]-bin_centers[0]
    max_count_index = np.argmax(hist)
    num_elements_in_highest_bin = hist[max_count_index]
    fig.quad(top=hist, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], fill_color="blue", line_color="white", alpha=0.7, legend_label='Probability density')
    row_spacing = max(x_gaussian)/(num_elements_in_highest_bin)
    fig.y_range = Range1d(start=0, end=num_elements_in_highest_bin)
    peakamps =y_data[1:]
    fig.extra_y_ranges = {"amplitude_range": Range1d(start=0, end=1)}
    fig.add_layout(LinearAxis(y_range_name="amplitude_range", axis_label="Amplitude [a.u.]"), 'right')  # Add the right y-axis
    fig.line(x_curve, x_gaussian, color= 'red', line_width = 2.5, legend_label='Gaussian distribution')
    fig.circle(x_data,peakamps, size=7, fill_color='red', legend_label='Peak Transient Time',y_range_name="amplitude_range")

    fig.xaxis.ticker.num_minor_ticks = 9
    xshift = 0
    
    x_coordinate = mean_x - std_x

    fig.line(x=[x_coordinate,x_coordinate], y=[0,num_elements_in_highest_bin/4], line_width=2, line_dash="dashed", line_color="black", legend_label= 'standard deviation')
    x_coordinate = mean_x + std_x

    fig.line(x=[x_coordinate,x_coordinate], y=[0,num_elements_in_highest_bin/4], line_width=2, line_dash="dashed", line_color="black")
    x_coordinate = mean_x

    fig.line(x=[x_coordinate,x_coordinate], y=[0,num_elements_in_highest_bin/4], line_width=2, line_color="black", legend_label= 'mean')

    text_annotation1 = Label(x=0, y=fig.height-100, x_units="screen", y_units='screen', text="standard deviation = "+f"{std_x:.2f}"+" ms", text_font_size="16pt")
    text_annotation2 = Label(x=0, y=fig.height-125, x_units="screen", y_units='screen', text="mean = "+f"{mean_x:.2f}"+" ms", text_font_size="16pt")
    fig.add_layout(text_annotation1)
    fig.add_layout(text_annotation2)

    fig.x_range.start = mean_x-(stddeviations)*std_x
    fig.x_range.end = mean_x+(stddeviations)*std_x
    fig.legend.location = 'top_right'

    return fig

def peakrefiner_maximum(data, peak_location, length):
    center_index = peak_location
    array = data
    search_range = length

    
    max_gradient = float('-inf')
    max_gradient_index = -1
    for i in range(center_index - search_range, center_index + search_range + 1):     
        gradient = array[i]
        
        if gradient > max_gradient:
            max_gradient = gradient
            max_gradient_index = i
    
    return max_gradient_index
    
def peakrefiner_center_of_weight(data, peak_location, length):
    center_index = peak_location
    search_range = length
    start_index = max(center_index - search_range, 0)
    end_index = min(center_index + search_range + 1, len(data))
    array = data[start_index:end_index]

    indices = np.arange(len(array))
    weighted_indices = indices * np.power(array, 1.5)
    total_weighted_index = np.sum(weighted_indices)
    total_weight = np.sum(np.power(array, 1.5))

    if total_weight == 0:
        raise ValueError("Array has zero total weight, cannot calculate center of gravity.")

    cog_index = int(total_weighted_index / total_weight)
    return start_index + cog_index    
    
    
    
'''   
def peakrefiner_center_of_weight(data, peak_location, length):
    center_index = peak_location
    search_range = length
    search_range_indices = range(min(center_index - search_range,0), max(center_index + search_range + 1,len(data)))
    array = data[search_range_indices]

    total_weighted_index = 0
    total_weight = 0
    order = 1.5
    for index, value in enumerate(array):
        total_weighted_index += index * (value)**order
        total_weight += (value)**order

    if total_weight == 0:
        raise ValueError("Array has zero total weight, cannot calculate center of gravity.")

    cog_index = int(total_weighted_index / total_weight)
    return search_range_indices[cog_index]
'''   
def peakrefiner_leftmost(data, peak_location, length):
    center_index = peak_location
    search_range = length
    search_range_indices = range(center_index - search_range, center_index + search_range + 1)
    array = data[search_range_indices]
   
    # Find the indices of the three highest points in descending order
    sorted_indices = sorted(range(len(array)), key=lambda i: array[i], reverse=True)
    highest_indices = sorted_indices[:2]

    # Return the index of the leftmost highest point
    leftmost_index = min(highest_indices)
    return search_range_indices[leftmost_index]
    
    
def peakrefiner_find_peak_to_right(data, peak_location, search_range):
    end_index = min(peak_location + search_range + 1, len(data))
    array = data[peak_location:end_index]

    if len(array) == 0:
        raise ValueError("No data in the search range.")

    peak_index = np.argmax(array) + peak_location
    return peak_index
    
def peakrefiner_rightmost(data, peak_location, length):
    center_index = peak_location
    search_range = length
    search_range_indices = range(center_index - search_range, center_index + search_range + 1)
    array = data[search_range_indices]
   
    # Find the indices of the three highest points in descending order
    sorted_indices = sorted(range(len(array)), key=lambda i: array[i], reverse=True)
    highest_indices = sorted_indices[:2]

    # Return the index of the leftmost highest point
    leftmost_index = max(highest_indices)
    return search_range_indices[leftmost_index]


def correlator(data, peak_locations, length):
    segments = []
    indexes = peak_locations
    range_value = length
    for idx in indexes:
        if idx < range_value or idx >= len(data) - range_value:
            continue
        
        segment = data[idx - range_value : idx + range_value + 1]
        segments.append(segment)
    alignment_shifts = []
    target_segment = segments[10]
    for segment in segments:
        segment_n = segment / np.linalg.norm(segment)
        target_segment_n = target_segment / np.linalg.norm(target_segment)
        cross_corr = correlate(target_segment_n, segment_n, mode='full')
        max_corr_index = np.argmax(cross_corr)
        shift = max_corr_index - len(target_segment) + 1
        alignment_shifts.append(shift)
    return np.sum([alignment_shifts,peak_locations], axis=0)


def create_envelope(x_data, signal, window_size):
    """
    Create an envelope for the input signal using a moving average.

    Parameters:
        signal (numpy.ndarray): Input signal.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: Envelope of the signal.
    """
    absolute_signal = np.abs(signal)
    envelope = moving_average(absolute_signal, 30)
    return envelope
'''  
def create_envelope1(x_data,signal, window_size):
    """
    Create an envelope for the input signal using a moving average.

    Parameters:
        signal (numpy.ndarray): Input signal.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: Envelope of the signal.
    """
    smoothing_factor = 0.5
    print(len(x_data))
    print(len(signal))
    spline = LSQUnivariateSpline(x_data, signal, t=None, k=3)
    envelope = spline(x_data)
    return envelope
'''  
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

if __name__ == '__main__':
    main()
