# Wintergatan Timing Data Analysis
## aka THE TIGHTINATOR

The best peak detector in the tri-state area!

The community has long since clamoured for automatic peak detection and generation of timing data for the Wintergatan Marble Machine project. An offhand comment during the 2023 Community Meetup colalesced into this project.

## View the live version

Run analysis at [https://tightinator.fun](https://tightinator.fun)

There is a 150M file size limit for upload, and working data will be removed after 7 days on disk.


## Getting Started

The main.py file in the top-level of this repository is the best file for use locally and testing. The `web/` directory contains the versions currently deployed onto the mcnulty.in endpoint.

The docker configuration is used for hosting and deploy, hence the internet host IP hard-coded in the gunicorn settings. This is to prevent header forging. You will need to edit a few sticking points if you want to use it for local development.

main.py will output help:
```
usage: main.py [-h] [-f FILENAME] [-o OUTPUT_FILENAME] [-t THRESHOLD] [-cf CUTOFF] [-c CHANNEL] [-d DOWNSAMPLE_RATE] [-cz CHUNK_SIZE] [-ex EXCLUSION] [-r FLOAT_PREC]
               [-l L_BESTSERIES] [-cp] [-b BPM_TARGET] [-bw BPM_WINDOW] [--work-dir WORK_DIR] [-w] [-x X_WIDE] [-y Y_HIGH] [-v]

Map transient times

options:
  -h, --help            show this help message and exit
  -f FILENAME, --file FILENAME
                        File to open.
  -o OUTPUT_FILENAME, --out OUTPUT_FILENAME
                        Filename to write output values to.
  -t THRESHOLD, --threshold THRESHOLD
                        Peak detection threshold. Works best 0.1 and above. Setting too high/low can cause misdetection. Defaults 0.1.
  -cf CUTOFF, --cutoff CUTOFF
                        The threshold below which the waveform should be cutoff for drawing. Does not affect anything outside the way the waveform is drawn, lowering below
                        0.01 will heavily decrease performance. Defaults 0.01.
  -c CHANNEL, --channel CHANNEL
                        Channel to get the waveform from. Defaults 1.
  -d DOWNSAMPLE_RATE, --downsampling DOWNSAMPLE_RATE
                        The downsampling used for drawing the waveform. Does not affect anything outside the way the waveform is drawn, lowering below 8 will heavily
                        decrease performance. Defaults 8.
  -cz CHUNK_SIZE, --chunk-size CHUNK_SIZE
                        Multiplied by sample rate, smaller chunks will increase run times. Defaults 8.4.
  -ex EXCLUSION, --exclusion EXCLUSION
                        Minimum distance between peaks in ms. Defaults 150.
  -r FLOAT_PREC, --precision FLOAT_PREC
                        Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438. Defaults 6.
  -l L_BESTSERIES, --length L_BESTSERIES
                        The length of the series of most consistent beats. Defaults 100.
  -cp, --correlation    Decide whether correlation is used as a peakfinder. Defaults True.
  -b BPM_TARGET, --bpm-target BPM_TARGET
                        The target BPM of the song. Use 0 for auto. Defaults 0.
  -bw BPM_WINDOW, --bpm-window BPM_WINDOW
                        Window of BPM that should be visible around the target. Will be scaled to 75% target height if 0. Defaults 0.
  --work-dir WORK_DIR   Directory structure to work under.
  -w, --web             Get some width/height values from/ browser objects for graphing. Defaults false.
  -x X_WIDE, --x-width X_WIDE
                        Fixed width for graphs. Defaults 2000.
  -y Y_HIGH, --plot-height Y_HIGH
                        Fixed height for single plot. Defaults 1340.
  -v, --verbose         Set debug logging
```

### Prerequisites

This project is written in Python 3, and uses pip to manage dependencies.

To install these libraries, run the following command:
```bash
pip install -r requirements.txt
```

### Running with Docker
This project comes with a [Dockerfile](./Dockerfile) for the webapp implementation. There is also a [compose file](./docker-compose.yml) and a [shell script](./docker.sh) for building and running the container in one go with `docker-compose`. If the container successfuly launches, you should be able to access it at http://127.0.0.1:5000 .

#### Standalone:
```sh
## Build and tag.
docker build . -t yanfett/wintergatan-data-analysis:latest

## Run the program.
# --network host      Use the host machine's loopback network. (127.0.0.1)
# -d                  Detach, runs the container in the background.
# -e DEBUG_MODE=True  Debug mode for the webapp. Remove or set to "False" to turn off.
docker run --network host -d -e DEBUG_MODE=True yanfett/wintergatan-data-analysis:latest

## Stop the program
docker stop <container name>
```

#### Compose:
```sh
## Build using the composefile.
docker compose build

## Run the program.
# -d                  Detach, runs the container in the background.
# -e DEBUG_MODE=True  Debug mode for the webapp. Remove or set to "False" to turn off.
docker compose run -d -e DEBUG_MODE=True wintergatan-data-analysis

## Stop the program
docker compose stop wintergatan-data-analysis
```

## Contributing

Any contribution is welcome! Please branch your feature and create a Pull Request when ready for review.


## Authors

So far written by Tom and Yan, with contributions from others! Thanks all!
https://github.com/YanFett/Wintergatan_data_analysis/graphs/contributors

