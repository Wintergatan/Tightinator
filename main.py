#!/usr/bin/python3

import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, CustomJS, LinearAxis, Range1d, Label, Line, TextInput, Button#, LinearColorMapper
from bokeh.layouts import row, column
from scipy.io import wavfile
from scipy.signal import find_peaks, correlate
from datetime import datetime
import json
import argparse
import logging
from skimage.measure import block_reduce #stays if downsampling is reintroduced

filename = ''
output_filename = ''
threshold = ''
channel = ''
chunk_size = ''
exclusion = ''
float_prec = ''
len_series = ''
work_dir = ''
l_bestseries = ''
web_mode = False
x_wide = ''
y_high = ''
bpm_target = ''
bpm_window = ''
correlation = False

parser = argparse.ArgumentParser(description='Map transient times')
parser.add_argument('-f', '--file', dest='filename', type=str, action='store', help='File to open.')
parser.add_argument('-o', '--out', dest='output_filename', type=str, action='store', help='Filename to write output values to.')
parser.add_argument('-t', '--threshold', dest='threshold', default='0.1', type=float, action='store', help='Peak detection threshold. Works best 0.1 and above. Setting too high/low can cause misdetection. Defaults 0.1.')
parser.add_argument('-cf', '--cutoff', dest='cutoff', default='0.01', type=float, action='store', help='The threshold below which the waveform should be cutoff for drawing. Does not affect anything outside the way the waveform is drawn, lowering below 0.01 will heavily decrease performance. Defaults 0.01.')
parser.add_argument('-c', '--channel', dest='channel', default='1', type=int, action='store', help='Channel to get the waveform from. Defaults 1.')
parser.add_argument('-d', '--downsampling', dest='downsample_rate', default='8', type=int, action='store', help='The downsampling used for drawing the waveform. Does not affect anything outside the way the waveform is drawn, lowering below 8 will heavily decrease performance. Defaults 8.')
parser.add_argument('-cz', '--chunk-size', dest='chunk_size', default='8.4', type=float, action='store', help='Multiplied by sample rate, smaller chunks will increase run times. Defaults 8.4.')
parser.add_argument('-ex', '--exclusion', dest='exclusion', default='150', type=int, action='store', help='Minimum distance between peaks in ms. Defaults 150.')
parser.add_argument('-r', '--precision', dest='float_prec', default='6', type=int, action='store', help='Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438. Defaults 6.')
parser.add_argument('-l', '--length', dest='l_bestseries', default='100', type=int, action='store', help='The length of the series of most consistent beats. Defaults 100.')
parser.add_argument('-cp', '--correlation', dest='correlation', default=False, action='store_true', help='Decide whether correlation is used as a peakfinder. Must enable.')
parser.add_argument('-b', '--bpm-target', dest='bpm_target', default='0', type=float, action='store', help='The target BPM of the song. Use 0 for auto. Defaults 0.')
parser.add_argument('-bw', '--bpm-window', dest='bpm_window', default='0', type=float, action='store', help='Window of BPM that should be visible around the target. Will be scaled to 75%% target height if 0. Defaults 0.')
parser.add_argument('--work-dir', dest='work_dir', action='store', help='Directory structure to work under.' )
parser.add_argument('-w', '--web', dest='web_mode', default=False, action='store_true', help='Get some width/height values from/ browser objects for graphing. Defaults false.')
parser.add_argument('-x', '--x-width', dest='x_wide', default='2000', type=int, action='store', help='Fixed width for graphs. Defaults 2000.')
parser.add_argument('-y', '--plot-height', dest='y_high', default='1340', type=int, action='store', help='Fixed height for single plot. Defaults 1340.')
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
    work_dir = ''
    filename = args.filename
    channel = args.channel
    threshold = args.threshold
    exclusion = args.exclusion
    downsample_rate = args.downsample_rate
    float_prec = args.float_prec
    l_bestseries = args.l_bestseries
    chunk_size = args.chunk_size
    correlation = args.correlation
    full_width = args.x_wide# - 15
    plot_height = args.y_high
    bpm_target = args.bpm_target    
    bpm_window = args.bpm_window
    cutoff = args.cutoff
    plot_height = int((plot_height - 140) / 2)

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
    
    
    signal, time, sample_rate = load_wav(filename, channel) #load the wav file, outputs a normalized signal of the selected channel, the corresponding time-xaxis and the sample_rate
    exclusion_samples = int(exclusion*sample_rate/1000) #calculate exclusion in samples from exclusion in ms
    chunk_size = int(chunk_size*sample_rate/1000) #calculate exclusion in samples from exclusion in ms
    peaks = rough_peaks(signal, time, threshold, exclusion_samples) #searches for the highest peaks in the file, they need to have a min height of threshold and a min distance of exclusion
    peaks = peakrefiner_center_of_weight(signal, time, peaks, chunk_size) #refines the rough peaks found before by centering them on their center of weight 
    if(correlation):
        peaks = peakrefiner_correlation(signal, time, peaks, chunk_size//2) #further refines the peaks by applying a correlation method to find the point of best overlap with current average
    peaks = peakrefiner_maximum_right(signal, time, peaks, chunk_size//8) #looks for a maximum in a very small window around the refined peak

    ### find the best series
    begin_best, l_bestseries = find_chunk_with_lowest_std(peaks, l_bestseries)
    best_peak_numbers = np.arange(l_bestseries)+begin_best

    ###calculate data for the best peaks
    best_peaks = create_peaks(signal, time, peaks["Samples"], best_peak_numbers)

    ### export data
    export_csv(output_filename, peaks, best_peaks, threshold, float_prec)
    export_json(work_dir, args, peaks, best_peaks)
    ### make similarness plot
    fig_center = figure(title='Similarness plot - most consistent Beats', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=int(np.floor(full_width/2)), height=plot_height)
    fig_center.output_backend = 'webgl'
    line_renderers = plot_centered(fig_center, signal, time, peaks, best_peak_numbers, chunk_size)

    ### make waveform plot
    fig_wave = figure(title='Waveform plot', x_axis_label='Time [s]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
    fig_wave.output_backend = 'webgl'
    button, input_bpm_target, input_bpm_window, circle_source = plot_waveform(fig_wave, signal, time, peaks, best_peaks, bpm_window, bpm_target, threshold, downsample_rate, cutoff)

    ### make stat plot
    fig_stat = figure(title='Statistics plot - most consistent Beats', x_axis_label='Transient Time difference [ms]', y_axis_label='Probability density[1/ms]', width=int(np.floor(full_width/2)), height=plot_height, tools="lasso_select,reset,pan,wheel_zoom,box_zoom,save")
    fig_stat.output_backend = 'webgl'
    plot_stat(fig_stat, signal, time, peaks, best_peak_numbers, line_renderers, circle_source)



    ### plot it
    layout = column(row(input_bpm_target, input_bpm_window, button), fig_wave, row(fig_center, fig_stat))
    if args.web_mode:
        print("Writing graphs to {}summary.html".format(work_dir))
        output_file("{}summary.html".format(work_dir), title="Summary Page")

        save(layout)
    else:
        output_file("summary.html", title="Summary Page")
        show(layout)

def create_peaks(signal, time, samples, peak_numbers=[]):
    """Creates a dict containing the following Keys:
        "Numbers": The number of the peaks.
        "Samples" : The sample position of the peaks.
        "Heights": The height of the peaks.
        "Times": The time of the peaks.
        "Diffs": The time difference between peak.
        "MeanDiff": The mean time difference between peaks.
        "StdDiff": The standard deviation of the time difference between peaks.
        "BPM": The BPM between peaks.
        "MeanBPM": The mean BPM between peaks.
        "StdBPM": The standard deviation of the BPM between peaks.
        "TimeMiddles": The middles between peaks.
        "Accel": The acceleration of difference between peaks.
        "AccelBPM": The acceleration of the BPM.
        "AccelDiff": The difference in acceleration of the BPM.
        
    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform.     
    time : array
        np.array that contains the corresponding time of each sample in ms.
    samples : array
        np.array that contains the position of each peak in samples.
    peak_numbers: array
        np.array that contains the Number of each peak that should be selected.
        
    Returns
    -------
    peaks : dict
        dict that contains all relevant data of the peaks.   
    """
    
    if(len(peak_numbers) == 0):
        peak_numbers = np.arange(len(samples))
    else:
        samples = samples[peak_numbers]
    times = time[samples]
    diffs = np.diff(times)
    BPM = (60 * 1000) / diffs
    accel = np.gradient(diffs)

    peaks ={
        "Numbers": peak_numbers,
        "Samples" : samples,
        "Heights": signal[samples],
        "Times": times,
        "Diffs": diffs,
        "MeanDiff": np.mean(diffs),
        "StdDiff": np.std(diffs),
        "BPM": BPM,
        "MeanBPM": np.mean(BPM),
        "StdBPM": np.std(BPM),
        "TimeMiddles": (times[:-1] + times[1:]) / 2,
        "Accel": accel,
        "AccelBPM": np.gradient(BPM),
        "AccelDiff": np.diff(accel)
    }
    return peaks

def export_csv(output_filename, peaks, best_peaks, threshold, float_prec): 
    """Exports a csv file of the following form:
        PeakTimes[ms]|PeakDifferences[ms]|BestTimes[ms]|BestDifferrences[ms]|data
        data contains row by row:
            standard deviation of differences in best series [ms].
            standard deviation of differences [ms].
            0
            start of best series by peaknumber.
            end of best series by peaknumber.
            0
            threshold in a range of 0 to 1 [a.u.].
    
    Parameters
    ----------
    output_filename : path
        The filename as which the csv should be saved.       
    peaks : dict
        A dict containing all the required peakdata.    
    best_peaks : dict
        A dict containing all the required peakdata of the best series of peaks.
    threshold: float
        The threshold used for detection.
    float_prec: int
        The precision used for exporting floats.
    """
    
    stdev = np.zeros(len(peaks["Times"]))
    stdev[0] = best_peaks["StdDiff"]              #the standard deviation of the differences in the best series
    stdev[1] = peaks["StdDiff"]                  #the standard deviation of all peak differences
    stdev[3] = np.min(best_peaks["Numbers"])  #the start of the best series
    stdev[4] = np.max(best_peaks["Numbers"])  #the end of the best series
    stdev[5] = len(peaks["Times"])        #the number of peaks
    stdev[7] = threshold                #the threshold used in recording the data
    combined_array = pad_and_stack_arrays([peaks["Times"], peaks["Diffs"], best_peaks["Times"], best_peaks["Diffs"], stdev])
    logging.info("Saving output values to {}".format(output_filename))
    np_fmt = "%1.{}f".format(float_prec)
    np.savetxt(output_filename, combined_array, delimiter=",", header="PeakTimes[ms],PeakDifferences[ms],BestTimes[ms],BestDifferrences[ms],data", fmt=np_fmt, comments="")


def export_json(work_dir, args, peaks, best_peaks): 
    """Exports a json file containing all passed args and some data about the peaks.
    
    Parameters
    ----------
    work_dir : path
        The path where the json should be saved.
    args : args
        Commandline arguments passed to the script.
    peaks : dict
        A dict containing all the required peakdata.    
    best_peaks : dict
        A dict containing all the required peakdata of the best series of peaks.
    """
    
    data = {
    "file" : args.filename,
    "out" : args.output_filename,
    "threshold" : args.threshold,
    "channel" : args.channel,
    "chunksize" : args.chunk_size,
    "exclusion" : args.exclusion,
    "precision" : args.float_prec,
    "length" : args.l_bestseries,
    "web" : args.web_mode,
    "correlation" : args.correlation,
    "bpm-target" : args.bpm_target,
    "bpm-window" : args.bpm_window,
    "work-dir" : args.work_dir,
    "x-width" : args.x_wide,
    "plot-height" : args.y_high,
    "verbose" : args.verbose,
    "best-peak-stdev" : best_peaks["StdDiff"],
    "peak-stdev" : peaks["StdDiff"],
    "start-best-series" : int(np.min(best_peaks["Numbers"])),
    "end-best-series" : int(np.max(best_peaks["Numbers"])),
    "number_of_peaks" : len(peaks["Times"])
    }
    # Write the dictionary to a JSON file
    with open('{}results_{}.json'.format(work_dir,datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), 'w') as json_file:
        json.dump(data, json_file)
    
def load_wav(Path, channel):
    """Loads a wav file, selects a channel and returns the waveform.

    Parameters
    ----------
    Path : path
        The path to the location of the wav file .       
    channel : int
        The selected channel, is offset by 1, so the first channel can be indexed with 1 .   

    Returns
    -------
    signal : array
        The Waveform of the selected channel as a normalized np.array of floats.
    time : array
        The time of each point in the signal array in ms.
    sample_rate : int
        The sample_rate of the wav file .  
    """
    
    sample_rate, data = wavfile.read(Path)
    num_channels = data.shape[1] if len(data.shape) > 1 else 1
    if(channel > num_channels):
        channel = num_channels - 1
    
    if(num_channels > 1):
        amplitude_data = data[:,channel-1] # First channel has to be 1, only programmers know things start at 0
    else:
        amplitude_data = data
    #reduced_signal = block_reduce(amplitude_data,(downsample_rate,), np.mean)
    reduced_signal = amplitude_data
    signal = np.abs(reduced_signal / np.max(np.abs(reduced_signal)))
    sample_rate = sample_rate
    time_factor = sample_rate/1000
    time = np.arange(0, len(signal))/time_factor
    return signal, time, sample_rate

def rough_peaks(signal, time, threshold, exclusion):
    """A simple peakfinder that finds all peaks that are higher than the set threshold. 
    Will only detect one peak within exclusion number of samples.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform.  
    time : array
        np.array that contains the corresponding time of each sample in ms.        
    threshold : float
        Lower bound of when a peak is concidered a peak.   
    exclusion : int
        Size of the window that excludes other peaks in samples.

    Returns
    -------
    peaks : dict
        dict that contains all relevant data of the peaks.    
    """
    
    samples, _ = find_peaks(signal, prominence=threshold, distance = exclusion)
    peaks = create_peaks(signal, time, samples)
    return peaks

    
def peak_chunks(signal, peak_Samples, chunk_size):
    """Cuts chunks around the given peaks and returns them as a 2d array, with each row showing a window of size chunk_size around the given peak_Samples.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform.     
    peak_Samples : array
        Points around which the chunks of the signal should be centered.  
    chunk_size : int
        Length of the chunks.

    Returns
    -------
    padded_chunks : 2darray
        An np.ndarray containing samples of size chunk_size centered around given peak_Samples.
        Will pad itself so that hitting a border is not a problem.
    """
    
    waveform = signal
    center_points = peak_Samples
    num_center_points = len(center_points)
    half_chunk_size = chunk_size // 2
    start_indices = np.maximum(0, np.array(center_points) - half_chunk_size)
    end_indices = np.minimum(len(waveform), np.array(center_points) + half_chunk_size + 1)
    padding = chunk_size - (end_indices - start_indices)
    padding[padding < 0] = 0
    
    indices = np.arange(chunk_size)
    indices = indices[np.newaxis, :] + start_indices[:, np.newaxis]
    chunks = np.take(waveform, indices, mode='clip')
    
    padded_chunks = np.where(padding[:, np.newaxis] > 0, 0, chunks)
#    tonormalize = True
#    if tonormalize:
    peak_heights = signal[peak_Samples]
    padded_chunks = padded_chunks / np.maximum(0.1, peak_heights[:, np.newaxis])
        
    return padded_chunks


def peakrefiner_center_of_weight(signal, time, old_peaks, chunk_size):
    """A peak refiner that takes a rough estimate of a peak location and shifts it to the center of weight of a given peak.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform .   
    time : array
        np.array that contains the corresponding time of each sample in ms.                
    old_peaks : dict
        Old peaks to refine.
    chunk_size : int
        Length of the window of calculation.

    Returns
    -------
    new_peaks : dict
        dict that contains all relevant data of the refocused peaks.    
    """
    
    chunks = peak_chunks(signal, old_peaks["Samples"], chunk_size)

    window_size = 51
    sigma = 50  # Adjust the sigma value as needed for your Gaussian kernel
    x = np.arange(window_size)
    weights = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - window_size) / sigma)**2)
    weights /= weights.sum()  # Normalize the weights
    chunks = np.apply_along_axis(lambda row: np.convolve(row, weights, mode='same'), axis=1, arr=chunks)
    center_index = chunk_size // 2

    start_indexes = np.maximum(0, old_peaks["Samples"]-center_index)
    # Calculate the weighted average (center of weight) for each time sample
    power = 1.5
    centers_of_weight = np.round(np.sum((np.arange(chunk_size)*np.power(chunks, power)), axis=1) / np.sum(np.power(chunks, power), axis=1)).astype(int)
    new_peak_samples = start_indexes + centers_of_weight 
    new_peaks = create_peaks(signal, time, new_peak_samples)
    return new_peaks
    
def peakrefiner_maximum_right(signal, time, old_peaks, chunk_size):
    """A peak refiner that takes the center of weight of a peak and finds the maximum right of that.
    It works surprisingly well.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform.    
    time : array
        np.array that contains the corresponding time of each sample in ms.               
    old_peaks : dict
        Old peaks to refine.
    chunk_size : int
        Length of the window of calculation.

    Returns
    -------
    new_peaks : dict
        dict that contains all relevant data of the refocused peaks.    
    """
    
    new_chunk_size = chunk_size //2
    shift = 0#new_chunk_size//2
    chunks = peak_chunks(signal, old_peaks["Samples"]+shift, new_chunk_size)
    center_index = new_chunk_size  // 2
    window_size = 31
    sigma = 10  # Adjust the sigma value as needed for your Gaussian kernel
    x = np.arange(window_size)
    weights = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - window_size) / sigma)**2)
    weights /= weights.sum()  # Normalize the weights
    chunks = np.apply_along_axis(lambda row: np.convolve(row, weights, mode='same'), axis=1, arr=chunks)
    start_indexes = np.maximum(0, old_peaks["Samples"] - center_index)  -chunk_size // 2
    
    # Find the index of the maximum value within each chunk
    max_indices = np.argmax(chunks, axis=1)
    
    # Calculate the new peak_Samples using the maximum indices
    new_peak_samples = old_peaks["Samples"] + max_indices
    new_peaks = create_peaks(signal, time, new_peak_samples)
    return new_peaks

 
def peakrefiner_correlation(signal, time, old_peaks, chunk_size):
    """A peak refiner that uses the correlation method to align all peaks.
    Best described as a "put stuff where it best fits in" method.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform.    
    time : array
        np.array that contains the corresponding time of each sample in ms.                
    old_peaks : dict
        Old peaks to refine.
    chunk_size : int
        Length of the window of calculation.

    Returns
    -------
    new_peaks : dict
        dict that contains all relevant data of the refocused peaks.   
    """
    
    new_chunk_size = chunk_size * 2
    chunks = peak_chunks(signal, old_peaks["Samples"], new_chunk_size)
    center_index = new_chunk_size  // 2
    window_size = 31
    sigma = 10  # Adjust the sigma value as needed for your Gaussian kernel
    x = np.arange(window_size)
    weights = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - window_size) / sigma)**2)
    weights /= weights.sum()  # Normalize the weights
    chunks = np.apply_along_axis(lambda row: np.convolve(row, weights, mode='same'), axis=1, arr=chunks)
    row_norms = np.max(np.abs(chunks), axis=1)
    chunks = np.abs(chunks / row_norms[:, np.newaxis])
    start_indexes = np.maximum(0, old_peaks["Samples"] - center_index) - new_chunk_size // 2
    diff_chunks = np.gradient(chunks, axis = 1)
    row_norms = np.max(np.abs(diff_chunks), axis=1)
    diff_chunks = np.abs(diff_chunks / row_norms[:, np.newaxis])
    acc_chunks = np.gradient(diff_chunks, axis = 1)
    row_norms = np.max(np.abs(acc_chunks), axis=1)
    acc_chunks = np.abs(acc_chunks / row_norms[:, np.newaxis])
    maxi_sum = diff_chunks + acc_chunks

    mean_trace = np.mean(maxi_sum, axis = 0)
    mean_trace = np.abs(mean_trace / np.max(mean_trace))
    best_shift_index = np.array([]).astype(np.int32)
    # Find the index of the maximum correlation value
    for trace in maxi_sum:
        window_size = 15
        trace_smooth = np.convolve(trace, weights, mode='same')
        tracenorm = np.abs(trace_smooth / np.max(trace_smooth))
        correlation = np.correlate(tracenorm, mean_trace, mode='full')

        index = np.argmax(correlation) - new_chunk_size
        index = index.astype(int)
        best_shift_index = np.append(best_shift_index, index)
    new_indices = old_peaks["Samples"] + best_shift_index
    new_peak_samples = new_indices
    new_peaks = create_peaks(signal, time, new_peak_samples)
    return new_peaks


def find_chunk_with_lowest_std(peaks, l_bestseries):
    """Finds the location of the best series of transients.

    Parameters
    ----------
    peaks : dict
        A dict containing all the required peakdata.    
    l_bestseries : int
        Length of best series.

    Returns
    -------
    start_of_best_series : int
        Start of the best series.
    l_bestseries : int
        Possibly modified length.
    """
    
    samples = peaks["Samples"]
    if(l_bestseries >= len(samples)):
        l_bestseries = len(samples)
    diffs = np.diff(samples)
    diff_std = []
    for i in range(len(diffs) - (l_bestseries-2)):
        diff_std.append(np.std(diffs[i:i+l_bestseries]))
    start_of_best_series = np.argmin(diff_std)
    return start_of_best_series, l_bestseries

def draw_line(fig, legend_str, pos_sample, chunk_height, vertical=True, color="black", dash="dashed"):
    """Draws a straight line on a figure.

    Parameters
    ----------
    fig : figure
        The figure to draw on.   
    legend_str : str
        String that describes what the line represents.
    pos_sample : float
        Where that line should be drawn.
    chunk_height : float
        Length of the line (set to height of plotwindow for max height).
    vertical : bool
        Whether to draw vertical or horizontal.
    color : str
        The color that the line should be.
    dash : str
        Change between dashed or other linemodes.
    """
    
    # Draw horizontal line
    if(not vertical):
        fig.line(x=[0, chunk_height], y=[pos_sample, pos_sample],
             line_width=2, line_dash="dashed", line_color=color, legend_label=legend_str)
    else:
    # Draw vertical line
        fig.line(x=[pos_sample, pos_sample], y=[0, chunk_height],
             line_width=2, line_dash="dashed", line_color=color, legend_label=legend_str)
'''
def plot_chunks(chunks, time, chunk_size, full_width, plot_height, best_peak_numbers=[]):
    """Draws chunks as a false color plot, might not work right now.

    Parameters
    ----------
    chunks : 2darray
        An np.ndarray containing samples of size chunk_size.
    time : array
        np.array that contains the corresponding time of each sample in ms.  
    chunk_size : int
        Length of the window of calculation.
    best_peak_numbers : array
        np.array that contains the Number of each peak of the best peak series.
    full_width: int
        How wide the plot should be drawn in pixels.
    plot_height: int
        How high the plot should be drawn in pixels.
    """
    
    fig_chunks = figure(title='chunks', width=full_width, height=plot_height)
    fig_chunks.output_backend = 'webgl'
    palette = 'Viridis256'  # Or any other palette you want
    color_mapper = LinearColorMapper(palette=palette, low=0, high=1)
    fig_chunks.image(image=[chunks], x=-time[chunk_size//2], y=0, dw=time[chunk_size], dh=chunks.shape[0], color_mapper=color_mapper)
    fig_chunks.line(x=[0,0], y=[0,chunks.shape[0]], color="red")
    if(len(best_peak_numbers) != 0):
        x_coordinate = np.min(best_peak_numbers)
        fig_chunks.line(x=[-time[chunk_size//2],time[chunk_size//2]], y=[x_coordinate,x_coordinate], line_width=2, line_dash="dashed", line_color="white")
        x_coordinate = np.max(best_peak_numbers)
        fig_chunks.line(x=[-time[chunk_size//2],time[chunk_size//2]], y=[x_coordinate,x_coordinate], line_width=2, line_dash="dashed", line_color="white")
    show(fig_chunks)

def plot_chunk_sim(chunks, time, chunk_size, full_width, plot_height):
    """Draws chunks in style of the similarness plot, might not work right now.

    Parameters
    ----------
    chunks : 2darray
        An np.ndarray containing samples of size chunk_size.
    time : array
        np.array that contains the corresponding time of each sample in ms. 
    chunk_size : int
        Length of the window of calculation.
    full_width: int
        How wide the plot should be drawn in pixels.
    plot_height: int
        How high the plot should be drawn in pixels.
    """
    
    fig_wave = figure(title='Wave plot', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
    fig_wave.output_backend = 'webgl'
    ys, xs = [],[]
    xrange = time[np.arange(chunk_size)] - time[chunk_size // 2]
    for chunk in chunks:
        ys.append(chunk)
        xs.append(xrange)
    fig_wave.multi_line(xs, ys, alpha = 0.5)
    fig_wave.circle(x=0, y=1, color="red")
    y_range_start = 0
    y_range_end = min(2,np.max(ys))
    fig_wave.y_range = Range1d(start=y_range_start, end=y_range_end)  # Set the y-range of the left y-axis
'''
def plot_waveform(fig, signal, time, peaks, best_peaks, bpm_window, bpm_target, threshold, downsample_rate, cutoff):
    """draws the waveform plot of the given signal and peaks.
    Will contain the Signal as Waveform, The BPM as green bars the BPM acceleration as orange and red bars, aswell as all detected peaks as red circles.

    Parameters
    ----------
    fig : figure
        The figure to draw into.
    signal : array
        np.array that contains the normalized waveform.    
    time : array
        np.array that contains the corresponding time of each sample in ms.  
    peaks : dict
        A dict containing all the required peakdata.    
    best_peaks : dict
        A dict containing all the required peakdata of the best series.   
    bpm_window : int
        How big the upper and lower bounds of the y-axis centered on the bpm_target or mean BPM will be.
    bpm_target: int
        On which bpm the y-axis should be centered.
    threshold: float
        The threshold used for peakdetection.
    downsample_rate: float
        The downsample rate used for drawing the waveform.
    cutoff: float
        The cutoff below which the waveform is not drawn.
    """

    reduced_signal = block_reduce(signal,(downsample_rate,), np.max)
    reduced_time = time[::downsample_rate]
    signal_cut = reduced_signal[reduced_signal > cutoff]
    time_cut = reduced_time[reduced_signal > cutoff] / 1000
    zoom_factor = 10
    fill_color = np.where(peaks["AccelBPM"] < 0, 'darkred', 'darkorange')
    if(not bpm_target):
        if(not bpm_window):
            secondary_yrange_start = max(0, best_peaks["MeanBPM"] - zoom_factor * best_peaks["StdBPM"])
            secondary_yrange_end = best_peaks["MeanBPM"] + zoom_factor * best_peaks["StdBPM"]
        else:
            secondary_yrange_start = max(0, best_peaks["MeanBPM"] - bpm_window)
            secondary_yrange_end = best_peaks["MeanBPM"] + bpm_window
    else:
        if(not bpm_window):
            secondary_yrange_start = 0
            secondary_yrange_end = bpm_target / 0.75
        else:
            secondary_yrange_start = bpm_target - bpm_window
            secondary_yrange_end = bpm_target + bpm_window
    accel_bottom = np.zeros(len(peaks["AccelBPM"])) + secondary_yrange_start
    accel_top = accel_bottom + np.abs(peaks["AccelBPM"])
    peak_second_middles = peaks["TimeMiddles"] / 1000
    peak_second_diffs= peaks["Diffs"] / 1000
    accel_source = ColumnDataSource(data=dict(x = peak_second_middles, bottom = accel_bottom, top = accel_top, width=(peak_second_diffs) * 0.5, color = fill_color))
    fig.y_range = Range1d(start=0, end=1)
    sec_y_range = Range1d(secondary_yrange_start, secondary_yrange_end)
    fig.extra_y_ranges = {"peak_diff_range": sec_y_range}
    fig.add_layout(LinearAxis(y_range_name="peak_diff_range", axis_label="BPM [Hz]"), 'right')  # Add the right y-axis
    fig.vbar(x=peak_second_middles, top=peaks["BPM"], width=(peak_second_diffs) * 0.9, y_range_name="peak_diff_range", color = 'green', fill_alpha=1, legend_label='BPM')
    circle_source = ColumnDataSource(data=dict(x = peaks["Times"] / 1000, y = peaks["Heights"], colors = ['red'] * len( peaks["Heights"]), alpha = [1]* len( peaks["Heights"])))
    circle_renderers = fig.circle(x = 'x', y = 'y' , legend_label='Detected Peaks', color = 'colors', alpha = 'alpha', source = circle_source)
    accel_renderer = fig.vbar(x='x', bottom='bottom',top='top', width='width', y_range_name="peak_diff_range", color = 'color', fill_alpha=1, legend_label='BPM Acceleration', source = accel_source)

    fig.line(time_cut, signal_cut, legend_label='Waveform')
    
    x_coordinate = np.min(best_peaks["Times"]) / 1000
    draw_line(fig, 'Segment of most consistent Beats', x_coordinate, 1, vertical=True, color="black", dash="dashed")

    x_coordinate = np.max(best_peaks["Times"]) / 1000
    draw_line(fig, 'Segment of most consistent Beats', x_coordinate, 1, vertical=True, color="black", dash="dashed")


    fig.x_range.start = 0
    fig.x_range.end = time_cut[-1]
    y_range_start = 0  # Define the start value for the y-axis range on the left
    y_range_end = 1   # Define the end value for the y-axis range on the left
    fig.y_range = Range1d(start=y_range_start, end=y_range_end)  # Set the y-range of the left y-axis
    fig.xaxis.ticker.num_minor_ticks = 9
    text_annotation1 = Label(x=0, y=0, text="sensitivity = "+f"{threshold:.2f}", text_font_size="12pt", background_fill_color = "white")
    fig.add_layout(text_annotation1)
    window_mean = (secondary_yrange_start + secondary_yrange_end)/2
    window_size = secondary_yrange_end - secondary_yrange_start
    input_bpm_target = TextInput(title="BPM target:", value=f'{window_mean:.2f}')
    input_bpm_window = TextInput(title="BPM window:", value=f'{window_size:.2f}')
    # Create a CustomJS callback for the button's onclick event
    callback = CustomJS(args=dict(y_range = sec_y_range, input_bpm_target = input_bpm_target, input_bpm_window = input_bpm_window, accel_source = accel_source, accelerations = np.abs(peaks["AccelBPM"])), code=
        """
            const bpm_target = parseFloat(input_bpm_target.value);
            const bpm_window = parseFloat(input_bpm_window.value);
            const y_start = Math.max(0,bpm_target-(bpm_window/2));
            const tops = accelerations.map(element => element + y_start);
            y_range.start = y_start;
            y_range.end = bpm_target+(bpm_window/2);
            const y_starts = new Float64Array(accelerations.length).fill(y_start);
            console.log(y_starts.length)
            console.log(tops.length)
            accel_source.data['bottom'] = y_starts;
            accel_source.data['top'] = tops;
            accel_source.change.emit();
        """)
    button = Button(label="Rescale BPM")
    button.js_on_click(callback)
    return button, input_bpm_target, input_bpm_window, circle_source

def plot_centered(fig, signal, time, peaks, best_peak_numbers, chunk_size):
    """Draws the similarness plot of the given signal and peaks.
    Will contain both peakshapes of the best series in blue, aswell as the peakshapes outside the best series in gray.
    The point where the peaks were detected is marked by a red circle.

    Parameters
    ----------
    fig : figure
        The figure to draw into.
    signal : array
        np.array that contains the normalized waveform.   
    time : array
        np.array that contains the corresponding time of each sample in ms.  
    peaks : dict
        A dict containing all the required peakdata.    
    best_peaks : dict
        A dict containing all the required peakdata of the best series.
    chunk_size : int
        Width of the window of samples that is shown.
    """
    
    cutoff = 0.01
    best_peaks = create_peaks(signal, time, peaks["Samples"], best_peak_numbers)
    not_chunks = peak_chunks(signal, peaks["Samples"], chunk_size)
    xs, ys = [], []
    
    x_axis = time[0:chunk_size] - time[chunk_size // 2]
    for i in np.arange(not_chunks.shape[0]):
        chunk = not_chunks[i]
        peakSample = peaks["Samples"][i]
        xs.append( x_axis[chunk > cutoff])
        ys.append((chunk[chunk > cutoff]))

    fig.multi_line(xs, ys, alpha=0.3, color = 'gray', legend_label='Peakshape outside best series')
    max_height = 1
    
    best_peak_samples = best_peaks["Samples"]
    chunks = peak_chunks(signal, best_peak_samples, chunk_size)

    xs, ys = [], []
    line_sources, line_renderers = [], []
    x_axis = time[0:chunk_size] - time[chunk_size//2]
    for i in np.arange(chunks.shape[0]):
        chunk = chunks[i]
        line_sources.append(ColumnDataSource(data={'x': x_axis[chunk > cutoff], 'y': chunk[chunk > cutoff]}))
        line_renderers.append(Line(x='x', y='y', line_color="blue", line_alpha=0.5))
        new_max = max(chunk[chunk > cutoff])
        if(new_max > max_height):
            max_height = new_max
    for line, source in zip(line_renderers, line_sources):
        fig.add_glyph(source,line)
        
    fig.circle(0, 1, size=10, fill_color='red', legend_label='detected peaks')
    fig.x_range.start = min(x_axis)
    max_xrange = max(x_axis)
    fig.x_range.end = max_xrange
    fig.y_range.start = 0
    max_yrange = min(max_height + 0.05, 2)
    fig.y_range.end = max_yrange
    fig.xaxis.ticker.num_minor_ticks = 9
    return line_renderers

def plot_stat(fig, signal, time, peaks, best_peak_numbers, line_renderers, circle_source_wav):
    """Draws the stat plot of the given peaks.
    Will contain difference between two peaks at the height of the left peak as red circles, aswell as a histogram showing the distribution of peak differences.
    On the top left the standard devation of the peaks aswell as the mean will be annotated.

    Parameters
    ----------
    fig : figure
        The figure to draw into.
    best_peaks : dict
        A dict containing all the required peakdata of the best series.
    """
    best_peaks = create_peaks(signal, time, peaks["Samples"], best_peak_numbers)
    x_data = best_peaks["Diffs"]  
    num_bins = int(1 + (3.322 * np.log(len(x_data))))

    mean_x = best_peaks["MeanDiff"] 
    std_x = best_peaks["StdDiff"] 

    number_of_standard_devations = 5
    max_dist= max(abs(mean_x - min(x_data)),abs(mean_x + max(x_data)))
    hist, bin_edges = np.histogram(x_data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_bins = np.mean(bin_centers)
    std_bins = np.std(bin_centers)
    binsize=bin_centers[1] - bin_centers[0]
    max_count_index = np.argmax(hist)
    bin_height = hist[max_count_index]
    fig.quad(top=hist, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], fill_color="blue", line_color="white", alpha=0.7, legend_label='Probability density')
    fig.y_range = Range1d(start=0, end=bin_height * 1.10)
    fig.extra_y_ranges = {"amplitude_range": Range1d(start=0, end=1)}
    fig.add_layout(LinearAxis(y_range_name="amplitude_range", axis_label="Amplitude [a.u.]"), 'right')  # Add the right y-axis
    plot_heights = best_peaks["Heights"]
    circle_source = ColumnDataSource(data=dict(x=x_data, y=best_peaks["Heights"][1:]))
    fig.circle(x='x',y='y', size=7, fill_color='red', legend_label='Peak Transient Time', y_range_name="amplitude_range", source=circle_source)
    fig.xaxis.ticker.num_minor_ticks = 9
    line_source_stdmin = ColumnDataSource(data=dict(x=[mean_x - std_x, mean_x - std_x], y=[0,bin_height/4]))
    fig.line(x='x', y='y', line_width=2, line_dash="dashed", line_color="black", legend_label= 'standard deviation', source = line_source_stdmin)
    line_source_stdmax = ColumnDataSource(data=dict(x=[mean_x + std_x, mean_x + std_x], y=[0,bin_height/4]))
    fig.line(x='x', y='y', line_width=2, line_dash="dashed", line_color="black", legend_label= 'standard deviation', source = line_source_stdmax)
    line_source_mean = ColumnDataSource(data=dict(x=[mean_x, mean_x], y=[0,bin_height/4]))
    fig.line(x='x', y='y', line_width=2, line_dash="dashed", line_color="red", legend_label= 'mean', source = line_source_mean)

    text_annotation1 = Label(x=0, y=fig.height-100, x_units="screen", y_units='screen', text="standard deviation = "+f"{std_x:.2f}"+" ms", text_font_size="16pt")
    text_annotation2 = Label(x=0, y=fig.height-125, x_units="screen", y_units='screen', text="mean = "+f"{mean_x:.2f}"+" ms", text_font_size="16pt")
    fig.add_layout(text_annotation1)
    fig.add_layout(text_annotation2)
    
    fig.x_range.start = mean_x - (number_of_standard_devations) * std_x
    fig.x_range.end = mean_x + (number_of_standard_devations) * std_x
    fig.legend.location = 'top_right'
    circle_source.selected.js_on_change('indices', CustomJS(args=dict(circle_source=circle_source, text_annotation1=text_annotation1, text_annotation2=text_annotation2, line_source_stdmin=line_source_stdmin, line_source_stdmax=line_source_stdmax, line_source_mean=line_source_mean, all_mean=mean_x, all_std=std_x, line_renderers=line_renderers, circle_source_wav = circle_source_wav, best_peak_numbers = best_peak_numbers), code="""
     if (!window.isCallbackQueued) {
         console.log(circle_source_wav)
        // Set a timeout to execute the callback after a delay (e.g., 200 milliseconds)
        window.isCallbackQueued = true;
        setTimeout(function() {
            window.isCallbackQueued = false;
            const selected_indices = circle_source.selected.indices;
            const lineColorUpdates = {}; // Collect line_color updates
            const circleUpdatesWav = []; // Collect circle updates
            const num_circles = circle_source_wav.data['alpha'];
            line_renderers.forEach((_, i) => {
                lineColorUpdates[i] = 0;
            });
            circle_source_wav.data['alpha'].forEach((_, i) => {
                circleUpdatesWav[i] = 0.1;
            });
            if (selected_indices.length > 0) {
                // Calculate mean and stdev
                const mean = selected_indices.reduce((a, b) => a + circle_source.data.x[b], 0) / selected_indices.length;
                const stdev = Math.sqrt(selected_indices.reduce((a, b) => a + Math.pow(circle_source.data.x[b] - mean, 2), 0) / selected_indices.length);
                
                // Update text annotations
                text_annotation1.text = "standard deviation = " + stdev.toFixed(2) + " ms";
                text_annotation2.text = "mean = " + mean.toFixed(2) + " ms";
                
                // Update lines
                line_source_stdmin.data = { x: [mean - stdev, mean - stdev], y: line_source_stdmin.data.y };
                line_source_stdmax.data = { x: [mean + stdev, mean + stdev], y: line_source_stdmin.data.y };
                line_source_mean.data = { x: [mean, mean], y: line_source_stdmin.data.y };
                const best_peaks_start = best_peak_numbers[0]
                // Collect line_color updates for selected indices
                for (let i = 0; i < selected_indices.length; i++) {
                    const index = selected_indices[i];
                    lineColorUpdates[index] = 0.5;
                    lineColorUpdates[index+1] = 0.5;
                    // Change color of Circles in waveform plot
                    circleUpdatesWav[index+best_peaks_start] = 1;
                    circleUpdatesWav[index+best_peaks_start+1] = 1;

                
                }
            } else {
                // On reset --> no circles selected
                text_annotation1.text = "standard deviation = " + all_std.toFixed(2) + " ms";
                text_annotation2.text = "mean = " + all_mean.toFixed(2) + " ms";
                line_source_stdmin.data = { x: [all_mean - all_std, all_mean - all_std], y: line_source_stdmin.data.y };
                line_source_stdmax.data = { x: [all_mean + all_std, all_mean + all_std], y: line_source_stdmin.data.y };
                line_source_mean.data = { x: [all_mean, all_mean], y: line_source_stdmin.data.y };

                for (let i = 0; i < line_renderers.length; i++) {
                    lineColorUpdates[i] = 0.5;
                }
                for (let i = 0; i < num_circles.length; i++) {
                    circleUpdatesWav[i] = 1;
                }
            }
            circle_source_wav.data['alpha'] = circleUpdatesWav;
            circle_source_wav.change.emit();
            // Apply batched line_color updates
            Object.entries(lineColorUpdates).forEach(([index, color]) => {
                const renderer = line_renderers[index];
                if (renderer) {
                    renderer.line_alpha = { value: color };
                }
            });
        }, 50); // milliseconds delay, adjust as needed
    }
    """))

def pad_and_stack_arrays(arrays):
    """Takes arrays of different lengths and pads them with zeroes to have the same length, then stacks them into a 2d array.

    Parameters
    ----------
    arrays : array of arrays
        Arrays of differing lengths.
        
    Returns
    -------
    stacked_array : 2darray
        A np.ndarray containing all arrays padded with zeroes.
    """
    
    # Find the length of the longest array
    max_length = max(len(arr) for arr in arrays)
    
    # Pad shorter arrays and stack them
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in arrays]
    stacked_array = np.column_stack(padded_arrays)
    
    return stacked_array


if __name__ == '__main__':
    main()
