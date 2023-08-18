# Wintergatan Timing Data Analysis
## aka THE TIGHTINATOR

The best peak detector in the tri-state area!

The community has long since clamoured for automatic peak detection and generation of timing data for the Wintergatan Marble Machine project. An offhand comment during the 2023 Community Meetup colalesced into this project.

## View the live version

A live workflow for detection and generation of timing data is live at [https://wtg.mcnulty.in](https://wtg.mcnulty.in)

There is a 150M file size limit for upload, and working data will be removed after 7 days on disk.


## Getting Started

The main.py file in the top-level of this repository is the best file for use locally and testing. The `web/` directory contains the versions currently deployed onto the mcnulty.in endpoint.

main.py will output help:
```
usage: main.py [-h] [-f FILENAME] [-o OUTPUT_FILENAME] [-d DOWNSAMPLE_RATE] [-t THRESH] [-c CHANNEL] [-ex EXCLUSION] [-r FLOAT_PREC] [-l LEN_SERIES] [-w] [-b BPM_TARGET]
               [-bw BPM_WINDOW] [-a KLICK] [--work-dir WORK_DIR] [-x X_WIDE] [-y Y_HIGH] [-v]

Map transient times

options:
  -h, --help            show this help message and exit
  -f FILENAME, --file FILENAME
                        File to open
  -o OUTPUT_FILENAME, --out OUTPUT_FILENAME
                        Filename to write output values to
  -d DOWNSAMPLE_RATE, --downsample-rate DOWNSAMPLE_RATE
                        DEFAULT=4 Amount by which to reduce resolution. Higher resolution means longer compute.
  -t THRESH, --threshold THRESH
                        DEFAULT=0.1 Peak detection threshold. Works best 0.1 and above. Setting too high/low can cause misdetection.
  -c CHANNEL, --channel CHANNEL
                        DEFAULT=1 Channel to get the waveform from.
  -ex EXCLUSION, --exclusion EXCLUSION
                        DEFAULT=3200 Minimum distance between peaks.
  -r FLOAT_PREC, --precision FLOAT_PREC
                        DEFAULT=6 Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438
  -l LEN_SERIES, --length LEN_SERIES
                        DEFAULT=100 The length of the series of most consistent beats.
  -w, --web             DEFAULT=False Get some width/height values from/ browser objects for graphing. Defaults false.
  -b BPM_TARGET, --bpm-target BPM_TARGET
                        DEFAULT=0 The target BPM of the song. 0 = Auto.
  -bw BPM_WINDOW, --bpm-window BPM_WINDOW
                        DEFAULT=0 Window of BPM that should be visible around the target. 0 = Auto.
  -a KLICK, --algorithm KLICK
                        DEFAULT=1 Switch between peak detecting algorithm. 0 = Center, 1 = Right
  --work-dir WORK_DIR   Directory structure to work under.
  -x X_WIDE, --x-width X_WIDE
                        DEFAULT=2000 Fixed width for graphs.
  -y Y_HIGH, --plot-height Y_HIGH
                        DEFAULT=600 Fixed height for single plot.
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

