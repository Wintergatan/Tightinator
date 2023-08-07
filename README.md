# Wintergatan Timing Data Analysis

The community has long since clamoured for automatic peak detection and generation of timing data for the Wintergata Marble Machine project. An offhand comment during the 2023 Community Meetup colalesced into this project.

## View the live version

A live workflow for detection and generation of timing data is live at [https://wtg.mcnulty.in](https://wtg.mcnulty.in)

There is a 100M file size limit for upload, and working data will be removed after 7 days on disk.


## Getting Started

The main.py file in the top-level of this repository is the best file for use locally and testing. The `web/` directory contains the versions currently deployed onto the mcnulty.in endpoint.

main.py will output help:
```
./main.py -h
usage: main.py [-h] [-f FILENAME] [-o OUTPUT_FILENAME] [-t THRESH] [-c NUM_CHANNELS] [-s OFF_CHANNEL] [-e ENVELOPE_SMOOTHNESS] [-x EXCLUSION] [-p FLOAT_PREC] [-v]

Map transient times

options:
  -h, --help            show this help message and exit
  -f FILENAME, --file FILENAME
                        File to open
  -o OUTPUT_FILENAME, --out OUTPUT_FILENAME
                        Filename to write output values to
  -t THRESH, --threshold THRESH
                        DEFAULT=0.25 Peak detection threshold, lower is rougher
  -c NUM_CHANNELS, --number-channels NUM_CHANNELS
                        DEFAULT=3 Number of channels, 2=MONO, 3=STEREO, etc
  -s OFF_CHANNEL, --channel-offset OFF_CHANNEL
                        DEFAULT=2 Channel offset, channel to analyze.
  -e ENVELOPE_SMOOTHNESS, --envelope-smoothness ENVELOPE_SMOOTHNESS
                        DEFAULT=100 Amount of rounding around the envelope
  -x EXCLUSION, --exclusion EXCLUSION
                        DEFAULT=30 Exclusion threshold
  -p FLOAT_PREC, --precision FLOAT_PREC
                        DEFAULT=6 Number of decimal places to round measurements to. Ex: -p 6 = 261.51927438
  -v, --verbose         Set debug logging
```

### Prerequisites

Currently no automated way of installing requirements is available. Check the `import` statements in the head of app/main.py


## Contributing

Any contribution is welcome! Please branch your feature and create a Pull Request when ready for review.


## Authors

So far written by Tom and Yan!


