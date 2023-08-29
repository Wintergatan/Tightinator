import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import LinearAxis, Range1d, Label, Legend, LinearColorMapper
from bokeh.layouts import row, column
from bokeh.transform import linear_cmap
from scipy.io import wavfile
from scipy.signal import find_peaks, correlate
import argparse
import logging
#from skimage.measure import block_reduce
import os
import random

filename = ''
output_filename = ''
threshold = ''
channel = ''
chunksize = ''
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


parser = argparse.ArgumentParser(description='Map transient times')
parser.add_argument('-f', '--file', dest='filename', type=str, action='store', help='File to open')
parser.add_argument('-o', '--out', dest='output_filename', type=str, action='store', help='Filename to write output values to')
parser.add_argument('-t', '--threshold', dest='threshold', default='0.1', type=float, action='store', help='DEFAULT=0.1 Peak detection threshold. Works best 0.1 and above. Setting too high/low can cause misdetection.')
parser.add_argument('-c', '--channel', dest='channel', default='1', type=int, action='store', help='DEFAULT=1 Channel to get the waveform from.')
parser.add_argument('-cz', '--chunksize', dest='chunksize', default='400', type=int, action='store', help='DEFAULT=400 Basissize of the chunks used for peakfinding.')
parser.add_argument('-ex', '--exclusion', dest='exclusion', default='6400', type=int, action='store', help='DEFAULT=6400 Minimum distance between peaks.')
parser.add_argument('-r', '--precision', dest='float_prec', default='6', type=int, action='store', help='DEFAULT=6 Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438')
parser.add_argument('-l', '--length', dest='l_bestseries', default='100', type=int, action='store', help='DEFAULT=100 The length of the series of most consistent beats.')
parser.add_argument('-w', '--web', dest='web_mode', default=False, action='store_true', help='DEFAULT=False Get some width/height values from/ browser objects for graphing. Defaults false.')
parser.add_argument('-b', '--bpm-target', dest='bpm_target', default='0', type=float, action='store', help='DEFAULT=0 The target BPM of the song. Use 0 for auto.')
parser.add_argument('-bw', '--bpm-window', dest='bpm_window', default='0', type=float, action='store', help='DEFAULT=0 Window of BPM that should be visible around the target. Will be scaled to 75%% target height if 0. Default 0.')
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
    channel = args.channel
    threshold = args.threshold
    exclusion = args.exclusion

    float_prec = args.float_prec
    l_bestseries = args.l_bestseries
    chunksize = args.chunksize
    full_width = args.x_wide# - 15
    plot_height = args.y_high
    bpm_target = args.bpm_target    
    bpm_window = args.bpm_window

    plot_height = int((plot_height-140)/2)

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

    signal, time, samplerate = loadwav(filename, channel) #load the wav file, outputs a normalized signal of the selected channel, the corresponding time-xaxis and the samplerate
    roughpeakSamples, roughpeakHeights = roughpeaks(signal, threshold, exclusion) #searches for the highest peaks in the file, they need to have a min height of threshold and a min distance of exclusion
    peakSamples, peakHeights =peakrefiner_center_of_weight(signal, roughpeakSamples, chunksize) #refines the rough peaks found before by centering them on their center of weight
    correlation = True
    if(correlation):
        peakSamples, peakHeights =peakrefiner_correlation(signal, peakSamples, chunksize//2) #further refines the peaks by applying a correlation method to find the point of best overlap with current average
    peakSamples, peakHeights =peakrefiner_maximum_right(signal, peakSamples, chunksize//8) #looks for a maximum in a very small window around the refined peak





    ### calculate all sorts of data

    peakTimes = time[peakSamples]
    peakDiffs = np.diff(peakTimes)
    meanDiff = np.mean(peakDiffs)
    stdDiff = np.std(peakDiffs)
    peakBPM = (60*1000)/peakDiffs
    meanBPM = np.mean(peakBPM)
    stdBPM = np.std(peakBPM)
    peakTimeMiddles = (peakTimes[:-1] + peakTimes[1:]) / 2
    peakAccel = np.gradient(peakDiffs)
    peakAccelBPM = np.gradient(peakBPM)
    peakAccelDiff = np.diff(peakAccel)


    ### find the best series
    begin_best, l_bestseries = find_chunk_with_lowest_std(peakSamples, l_bestseries)
    bestpeakNumbers = np.arange(l_bestseries)+begin_best

    ### calculate all sorts of data for the best series
    bestpeakSamples = peakSamples[bestpeakNumbers]
    bestpeakHeights = signal[bestpeakSamples]
    bestpeakTimes = time[bestpeakSamples]
    bestpeakDiffs = np.diff(bestpeakTimes)
    bestmeanDiff = np.mean(bestpeakDiffs)
    beststdDiff = np.std(bestpeakDiffs)
    bestpeakBPM = (60*1000)/bestpeakDiffs
    bestmeanBPM = np.mean(bestpeakBPM)
    beststdBPM = np.std(bestpeakBPM)
    bestpeakTimeMiddles = (bestpeakTimes[:-1] + bestpeakTimes[1:]) / 2
    bestpeakAccel = np.gradient(bestpeakDiffs)
    bestpeakAccelBPM = np.gradient(bestpeakBPM)
    bestpeakAccelDiff = np.diff(bestpeakAccel)

 
    
    ### export csv
    stdev = np.zeros(len(peakSamples))
    stdev[0] = beststdDiff              #the standard deviation of the differences in the best series
    stdev[1] = stdDiff                  #the standard deviation of all peak differences
    stdev[3] = np.min(bestpeakNumbers)  #the start of the best series
    stdev[4] = np.max(bestpeakNumbers)  #the end of the best series
    stdev[5] = len(peakSamples)        #the number of peaks
    stdev[7] = threshold                #the threshold used in recording the data
    combined_array = pad_and_stack_arrays([peakTimes,peakDiffs,bestpeakTimes,bestpeakDiffs,stdev])
    logging.info("Saving output values to {}".format(output_filename))
    np_fmt = "%1.{}f".format(float_prec)
    np.savetxt(output_filename, combined_array, delimiter=",", header="PeakTimes[ms],PeakDifferences[ms],BestTimes[ms],BestDifferrences[ms],data", fmt=np_fmt, comments="")


    ### make similarness plot
    fig_center = figure(title='Similarness plot - most consistent Beats', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=int(np.floor(full_width/2)), height=plot_height)
    fig_center.output_backend = 'webgl'
    center_fig = plot_centered(fig_center,signal,time,peakSamples, bestpeakSamples,chunksize,meanDiff,stdDiff,bestmeanDiff,beststdDiff)

    ### make stat plot
    fig_stat = figure(title='Statistics plot - most consistent Beats', x_axis_label='Transient Time difference [ms]', y_axis_label='Probability density[1/ms]', width=int(np.floor(full_width/2)), height=plot_height)
    fig_stat.output_backend = 'webgl'
    stat_fig = plot_stat(fig_stat, peakDiffs,peakHeights,meanDiff,stdDiff,bestpeakDiffs,bestpeakHeights,bestmeanDiff,beststdDiff)

    ### make waveform plot
    fig_wave = figure(title='Waveform plot', x_axis_label='Time [s]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
    fig_wave.output_backend = 'webgl'
    plot_waveform(fig_wave,signal,time, time[peakSamples],signal[peakSamples], bestpeakTimes, peakTimeMiddles, peakDiffs, peakBPM, peakAccelBPM, bestmeanBPM, beststdBPM, bpm_window, bpm_target, threshold)

    ### plot it
    layout = column(fig_wave, row(fig_center, stat_fig))
    output_file("summary.html", title="Summary Page")
    show(layout)


def normalize(array):
    """Normalizes the input array, so that max(array) = 1

    Parameters
    ----------
    file_loc : array
        an array of numbers

    Returns
    -------
    normalized array : array
        an array of numbers that dont exceed 1
    """
    return np.abs(array / np.max(np.abs(array)))

def loadwav(Path, channel):
    """loads a wav file, selects a channel and downsamples it

    Parameters
    ----------
    Path : path
        the path to the location of the wav file        
    channel : int
        the selected channel, is offset by 1, so the first channel can be indexed with 1    
    downsamplerate : int
        factor by which the data should be reduced

    Returns
    -------
    signal : array
        the Waveform of the selected channel as a normalized np.array of floats
    time : array
        the time of each point in the signal array in milliseconds
    samplerate : int
       the samplerate of the wav file   
    """
    sample_rate, data = wavfile.read(Path)
    num_channels = data.shape[1] if len(data.shape) > 1 else 1
    if(channel > num_channels):
        channel = num_channels-1
    
    if(num_channels > 1):
        amplitude_data = data[:,channel-1] # First channel has to be 1, only programmers know things start at 0
    else:
        amplitude_data = data
    #reduced_signal = block_reduce(amplitude_data,(downsamplerate,) , np.mean)
    reduced_signal = amplitude_data
    signal = normalize(reduced_signal)
    samplerate = sample_rate
    timefactor = sample_rate/1000
    time = np.arange(0, len(signal))/timefactor
    return signal, time, samplerate

def roughpeaks(signal, threshold, exclusion):
    """a simple peakfinder that finds all peaks that are higher than the set threshold. 
    Will only detect one peak within exclusion number of samples.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform     
    threshold : float
        lower bound of when a peak is concidered a peak    
    exclusion : int
        size of the window that excludes other peaks

    Returns
    -------
    peakSamples : array
        position of the peaks in the signal in samples
    peakHeight : array
        height of the peaks in peakSamples
    """
    peakSamples, _ = find_peaks(signal, prominence=threshold,distance = exclusion)
    return peakSamples, signal[peakSamples]
    
def peakchunks(signal, peakSamples, chunksize):
    """cuts chunks around the given peaks and returns them as a 2d array, with each row showing a window of size chunksize around the given peakSamples.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform     
    peakSamples : array
        Points around which the chunks of the signal should be centered   
    chunksize : int
        length of the chunks

    Returns
    -------
    paddedchunks : 2darray
        an np.ndarray containing samples of size chunksize centered around given peakSamples.
        Will pad itself so that hitting a border is not a problem
    """
    waveform = signal
    centerpoints = peakSamples
    num_centerpoints = len(centerpoints)
    half_chunk_size = chunksize // 2
    start_indices = np.maximum(0, np.array(centerpoints) - half_chunk_size)
    end_indices = np.minimum(len(waveform), np.array(centerpoints) + half_chunk_size + 1)
    padding = chunksize - (end_indices - start_indices)
    padding[padding < 0] = 0
    
    indices = np.arange(chunksize)
    indices = indices[np.newaxis, :] + start_indices[:, np.newaxis]
    chunks = np.take(waveform, indices, mode='clip')
    
    padded_chunks = np.where(padding[:, np.newaxis] > 0, 0, chunks)
#    tonormalize = True
#    if tonormalize:
    peakHeights = signal[peakSamples]
    padded_chunks = padded_chunks / np.maximum(0.1,peakHeights[:, np.newaxis])
        
    return padded_chunks

def gaussian_weights(mu, sigma):
    """Provides weights for gaussian smoothing

    Parameters
    ----------
    mu : float
        width of the distribution (?)
    sigma : float
        steepness of the distribution(?)

    Returns
    -------
    weights : array
        gaussian weights for smoothing
    """
    x = np.arange(mu)
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma)**2)


def peakrefiner_center_of_weight(signal, oldpeakSamples, chunksize):
    """a peak refiner that takes a rough estimate of a peak location and shifts it to the center of weight of a given peak.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform     
    oldpeakSamples : array
        peak locations to refine
    chunksize : int
        length of the window of calculation

    Returns
    -------
    newpeakSamples : array
        an array of refocused peaks in Samples
    newpeakHeights : array
        an array of Heights of those refocused peaks
    """
    chunks = peakchunks(signal, oldpeakSamples, chunksize)

    window_size = 51
    sigma = 50  # Adjust the sigma value as needed for your Gaussian kernel
    weights = gaussian_weights(window_size, sigma)
    weights /= weights.sum()  # Normalize the weights
    chunks = np.apply_along_axis(lambda row: np.convolve(row, weights, mode='same'), axis=1, arr=chunks)
    center_index = chunksize // 2

    start_indexes = np.maximum(0,oldpeakSamples-center_index)
    # Calculate the weighted average (center of weight) for each time sample
    power =1.5
    centers_of_weight = np.round(np.sum((np.arange(chunksize)*np.power(chunks, power)), axis=1) / np.sum(np.power(chunks,power), axis=1)).astype(int)
    newpeakSamples = start_indexes + centers_of_weight 
    return newpeakSamples, signal[newpeakSamples]
    
def peakrefiner_maximum_right(signal, oldpeakSamples, chunksize):
    """a peak refiner that takes the center of weight of a peak and finds the maximum right of that.
    It works surprisingly well.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform     
    oldpeakSamples : array
        peak locations to refine
    chunksize : int
        length of the window of calculation

    Returns
    -------
    newpeakSamples : array
        an array of refocused peaks in Samples
    newpeakHeights : array
        an array of Heights of those refocused peaks
    """
    newchunksize = chunksize //2
    shift = 0#newchunksize//2
    chunks = peakchunks(signal, oldpeakSamples+shift, newchunksize)
    center_index = newchunksize  // 2
    window_size = 31
    sigma = 10  # Adjust the sigma value as needed for your Gaussian kernel
    weights = gaussian_weights(window_size, sigma)
    weights /= weights.sum()  # Normalize the weights
    chunks = np.apply_along_axis(lambda row: np.convolve(row, weights, mode='same'), axis=1, arr=chunks)
    start_indexes = np.maximum(0, oldpeakSamples - center_index)-chunksize//2
    
    # Find the index of the maximum value within each chunk
    max_indices = np.argmax(chunks, axis=1)
    
    # Calculate the new peakSamples using the maximum indices
    new_peakSamples = oldpeakSamples + max_indices
    
    return new_peakSamples, signal[new_peakSamples]

 
def peakrefiner_correlation(signal, oldpeakSamples, chunksize):
    """a peak refiner that takes the center of weight of a peak and finds the maximum right of that.
    It works surprisingly well.

    Parameters
    ----------
    signal : array
        np.array that contains the normalized waveform     
    oldpeakSamples : array
        peak locations to refine
    chunksize : int
        length of the window of calculation

    Returns
    -------
    newpeakSamples : array
        an array of refocused peaks in Samples
    newpeakHeights : array
        an array of Heights of those refocused peaks
    """

    

    newchunksize = chunksize * 2
    chunks = peakchunks(signal, oldpeakSamples, newchunksize)
    center_index = newchunksize  // 2
    window_size = 31
    sigma = 10  # Adjust the sigma value as needed for your Gaussian kernel
    weights = gaussian_weights(window_size, sigma)
    weights /= weights.sum()  # Normalize the weights
    chunks = np.apply_along_axis(lambda row: np.convolve(row, weights, mode='same'), axis=1, arr=chunks)
    row_norms = np.max(np.abs(chunks), axis=1)
    chunks = np.abs(chunks / row_norms[:, np.newaxis])
    start_indexes = np.maximum(0, oldpeakSamples - center_index)-newchunksize//2
    diffchunks= np.gradient(chunks,axis = 1)
    row_norms = np.max(np.abs(diffchunks), axis=1)
    diffchunks = np.abs(diffchunks / row_norms[:, np.newaxis])
    accchunks= np.gradient(diffchunks,axis = 1)
    row_norms = np.max(np.abs(accchunks), axis=1)
    accchunks = np.abs(accchunks / row_norms[:, np.newaxis])
    maxisum = diffchunks + accchunks

    meantrace = np.mean(maxisum,axis = 0)
    meantrace = np.abs(meantrace/np.max(meantrace))
    best_shift_index = np.array([]).astype(np.int32)
    # Find the index of the maximum correlation value
    for trace in maxisum:
        window_size = 15
        trace_smooth = np.convolve(trace, weights, mode='same')
        tracenorm = np.abs(trace_smooth/np.max(trace_smooth))
        correlation = np.correlate(tracenorm, meantrace, mode='full')

        index = np.argmax(correlation)-newchunksize
        index = index.astype(int)
        best_shift_index= np.append(best_shift_index, index)
    new_indices = oldpeakSamples + best_shift_index
    new_peakSamples = new_indices
    
    return new_peakSamples, signal[new_peakSamples]


def find_chunk_with_lowest_std(peakSamples, l_bestseries):
    """finds the location of the best series of transients.

    Parameters
    ----------
    peakSamples : array
        peak locations     
    l_bestseries : int
        length of best series

    Returns
    -------
    start_of_best_series : int
        start of the best series
    l_bestseries : int
        possibly modified length
    """
    if(l_bestseries >= len(peakSamples)):
        l_bestseries = len(peakSamples)
    diffs = np.diff(peakSamples)
    diff_std = []
    for i in range(len(diffs)-(l_bestseries-2)):
        diff_std.append(np.std(diffs[i:i+l_bestseries]))
    start_of_best_series=np.argmin(diff_std)
    return start_of_best_series, l_bestseries

def drawline(fig,legendstr,posSample,chunkheight,vertical=True,color="black",dash="dashed"):
    """draws a straight line on a figure

    Parameters
    ----------
    fig : figure
        the figure to draw on    
    legendstr : str
        string that describes what the line represents
    posSample : float
        where that line should be drawn
    chunkheight : float
        length of the line (set to height of plotwindow for max height)
    vertical : bool
        wheter to draw vertical or horizontal
    """
    # Draw horizontal line
    if(not vertical):
        fig.line(x=[0, chunkheight], y=[posSample, posSample],
             line_width=2, line_dash="dashed", line_color=color, legend_label=legendstr)
    else:
    # Draw vertical line
        fig.line(x=[posSample, posSample], y=[0, chunkheight],
             line_width=2, line_dash="dashed", line_color=color, legend_label=legendstr)

def plotchunks(chunks,time,chunksize,bestpeakNumbers=[]):
    fig_chunks = figure(title='chunks', width=full_width, height=plot_height)
    fig_chunks.output_backend = 'webgl'
    palette = 'Viridis256'  # Or any other palette you want
    color_mapper = LinearColorMapper(palette=palette, low=0, high=1)
    fig_chunks.image(image=[chunks], x=-time[chunksize//2], y=0, dw=time[chunksize], dh=chunks.shape[0], color_mapper=color_mapper)
    fig_chunks.line(x=[0,0], y=[0,chunks.shape[0]], color="red")
    if(len(bestpeakSamples) != 0):
        x_coordinate = np.min(bestpeakNumbers)
        fig_chunks.line(x=[-time[chunksize//2],time[chunksize//2]], y=[x_coordinate,x_coordinate], line_width=2, line_dash="dashed", line_color="white")
        x_coordinate = np.max(bestpeakNumbers)
        fig_chunks.line(x=[-time[chunksize//2],time[chunksize//2]], y=[x_coordinate,x_coordinate], line_width=2, line_dash="dashed", line_color="white")
    show(fig_chunks)

def plotchunksim(chunks,time,chunksize, full_width, plot_height):
    fig_wave = figure(title='Wave plot', x_axis_label='Time [ms]', y_axis_label='Amplitude [a.u.]', width=full_width, height=plot_height)
    fig_wave.output_backend = 'webgl'
    ys, xs = [],[]
    xrange = time[np.arange(chunksize)]-time[chunksize //2]
    for chunk in chunks:
        ys.append(chunk)
        xs.append(xrange)
    fig_wave.multi_line(xs, ys, alpha = 0.5)
    fig_wave.circle(x=0, y=1, color="red")
    y_range_start =0
    y_range_end = min(2,np.max(ys))
    fig_wave.y_range = Range1d(start=y_range_start, end=y_range_end)  # Set the y-range of the left y-axis

def plot_waveform(fig,signal,time, peakTimes, peakHeights, bestpeakTimes, peakTimeMiddles, peakDiffs, peakBPM, peakAccelBPM, bestmeanBPM, beststdBPM, bpm_window, bpm_target, threshold):
    cutoff = 0.01
    signalc = signal[signal >cutoff]
    timec = time[signal >cutoff]/1000
    zoom_factor = 10
    MStoBPM = (60*1000)
    fill_color = np.where(peakAccelBPM < 0, 'darkred', 'darkorange')
    if(not bpm_target):
        if(not bpm_window):
            secyrange_start = max(0,bestmeanBPM-zoom_factor*beststdBPM)
            secyrange_end = bestmeanBPM+zoom_factor*beststdBPM
        else:
            secyrange_start = max(0,bestmeanBPM-bpm_window)
            secyrange_end = bestmeanBPM+bpm_window
        accel_bottom = secyrange_start
        accel_top = accel_bottom + np.abs(peakAccelBPM)
    else:
        if(not bpm_window):
            secyrange_start = 0
            secyrange_end = bpm_target/0.75
        else:
            secyrange_start = bpm_target - bpm_window
            secyrange_end = bpm_target + bpm_window
        accel_bottom = secyrange_start
        accel_top = secyrange_start+np.abs(peakAccelBPM)
    peakSecondsMiddles = peakTimeMiddles/1000
    peakSecondDiffs= peakDiffs/1000
    
    fig.y_range = Range1d(start=0, end=1)
    
    fig.extra_y_ranges = {"peak_diff_range": Range1d(secyrange_start, secyrange_end)}
    fig.add_layout(LinearAxis(y_range_name="peak_diff_range", axis_label="BPM [Hz]"), 'right')  # Add the right y-axis
    fig.vbar(x=peakSecondsMiddles, top=peakBPM, width=(peakSecondDiffs)*0.9, y_range_name="peak_diff_range", color = 'green', fill_alpha=1, legend_label='BPM')
    fig.circle(peakTimes/1000, peakHeights, legend_label='Detected Peaks', color = 'red')
    fig.vbar(x=peakSecondsMiddles, bottom=accel_bottom,top=accel_top , width=(peakSecondDiffs)*0.5, y_range_name="peak_diff_range", color = fill_color, fill_alpha=1, legend_label='BPM Acceleration')
    fig.line(timec, signalc, legend_label='Waveform')
    
    x_coordinate = np.min(bestpeakTimes)/1000

    fig.line(x=[x_coordinate,x_coordinate], y=[0,1], line_width=2, line_dash="dashed", line_color="black", legend_label= 'Segment of most consistent Beats')
    x_coordinate = np.max(bestpeakTimes)/1000
    
    fig.line(x=[x_coordinate,x_coordinate], y=[0,1], line_width=2, line_dash="dashed", line_color="black")

    fig.x_range.start = 0
    fig.x_range.end = timec[-1]
    y_range_start = 0  # Define the start value for the y-axis range on the left
    y_range_end = 1   # Define the end value for the y-axis range on the left
    fig.y_range = Range1d(start=y_range_start, end=y_range_end)  # Set the y-range of the left y-axis
    fig.xaxis.ticker.num_minor_ticks = 9
    text_annotation1 = Label(x=0, y=0, text="sensitivity = "+f"{threshold:.2f}", text_font_size="12pt", background_fill_color = "white")
    fig.add_layout(text_annotation1)

def plot_centered(fig, signal, time,peakSamples, bestpeakSamples,chunksize,meanDiff,stdDiff,bestmeanDiff,beststdDiff):
    cutoff = 0.01
    not_bestpeakSamples = np.setdiff1d(peakSamples, bestpeakSamples)
    not_chunks = peakchunks(signal, not_bestpeakSamples , chunksize)
    xs, ys, ys2, peakheights = [], [], [], []
    x_axis = time[0:chunksize] - time[chunksize//2]
    for i in np.arange(not_chunks.shape[0]):
        chunk = not_chunks[i]
        peakSample = not_bestpeakSamples[i]
        xs.append( x_axis[chunk >cutoff])
        ys.append((chunk[chunk>cutoff]))

    fig.multi_line(xs, ys, alpha=0.3, color = 'gray', legend_label='Peakshape outside best series')
    maxheight = 1
    chunks = peakchunks(signal, bestpeakSamples , chunksize)

    xs, ys, ys2, peakheights = [], [], [], []
    x_axis = time[0:chunksize] - time[chunksize//2]
    for i in np.arange(chunks.shape[0]):
        chunk = chunks[i]
        peakSample = bestpeakSamples[i]
        xs.append( x_axis[chunk >cutoff])
        ys.append((chunk[chunk>cutoff]))
        newmax = max(chunk[chunk>cutoff])
        if(newmax > maxheight):
            maxheight = newmax
        
    fig.multi_line(xs, ys, alpha=0.5, color = 'blue', legend_label='Peakshape best series')
    fig.circle(0, 1, size=10, fill_color='red', legend_label='detected peaks')
    fig.x_range.start = min(x_axis)
    max_xrange = max(x_axis)
    fig.x_range.end = max_xrange
    fig.y_range.start = 0
    max_yrange = min(maxheight + 0.05,2)
    fig.y_range.end = max_yrange
    fig.xaxis.ticker.num_minor_ticks = 9

def plot_stat(fig, peakDiffs,peakHeights,meanDiff,stdDiff,bestpeakDiffs,bestpeakHeights,bestmeanDiff,beststdDiff):

    num_bins = int(1 + (3.322 * np.log(len(bestpeakDiffs))))
    x_data = bestpeakDiffs
    mean_x = bestmeanDiff
    std_x = beststdDiff

    stddeviations = 5
    max_dist= max(abs(mean_x-min(x_data)),abs(mean_x+max(x_data)))
    hist, bin_edges = np.histogram(x_data, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_bins = np.mean(bin_centers)
    std_bins = np.std(bin_centers)
    binsize=bin_centers[1]-bin_centers[0]
    max_count_index = np.argmax(hist)
    binheight = hist[max_count_index]
    fig.quad(top=hist, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], fill_color="blue", line_color="white", alpha=0.7, legend_label='Probability density')
    fig.y_range = Range1d(start=0, end=binheight*1.10)
    fig.extra_y_ranges = {"amplitude_range": Range1d(start=0, end=1)}
    fig.add_layout(LinearAxis(y_range_name="amplitude_range", axis_label="Amplitude [a.u.]"), 'right')  # Add the right y-axis
    fig.circle(bestpeakDiffs,bestpeakHeights[1:], size=7, fill_color='red', legend_label='Peak Transient Time',y_range_name="amplitude_range")
    fig.xaxis.ticker.num_minor_ticks = 9
    x_coordinate = mean_x - std_x

    drawline(fig,'standard deviation',x_coordinate,binheight/4,vertical=True,color="black",dash="dashed")
    x_coordinate = mean_x + std_x
    drawline(fig,'standard deviation',x_coordinate,binheight/4,vertical=True,color="black",dash="dashed")
    x_coordinate = mean_x
    drawline(fig,'mean',x_coordinate,binheight/4,vertical=True,color="red",dash="dashed")

    text_annotation1 = Label(x=0, y=fig.height-100, x_units="screen", y_units='screen', text="standard deviation = "+f"{std_x:.2f}"+" ms", text_font_size="16pt")
    text_annotation2 = Label(x=0, y=fig.height-125, x_units="screen", y_units='screen', text="mean = "+f"{mean_x:.2f}"+" ms", text_font_size="16pt")
    fig.add_layout(text_annotation1)
    fig.add_layout(text_annotation2)
    
    fig.x_range.start = mean_x-(stddeviations)*std_x
    fig.x_range.end = mean_x+(stddeviations)*std_x
    fig.legend.location = 'top_right'
    return fig

def pad_and_stack_arrays(arrays):
    # Find the length of the longest array
    max_length = max(len(arr) for arr in arrays)
    
    # Pad shorter arrays and stack them
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in arrays]
    stacked_array = np.column_stack(padded_arrays)
    
    return stacked_array


if __name__ == '__main__':
    main()
