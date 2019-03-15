# General Tracker
by Balazs Vagvolgyi

Simple utility for tracking a single object.

User needs to manually select the ROI of the patch to be tracked.

Frames may be skipped when the object is lost by the tracker or is not visible.

Output is generated in CSV format:

    <frame-num>,<status>,<x-coordinate>,<y-coordinate>

    # <status> values:
    #     0 - not tracking
    #     1 - tracking automatically
    #     2 - template manually selected

Follow instructions in the terminal.

Keyboard inputs are active when the image window is in focus.

Requires OpenCV 3.4.1+.


Command line arguments:
-----------------------

    track <video-file> <output-file> <match-threshold>

    <match-threshold> range: (-1.0, 1.0)

Example:
-------------------------------------------------

    track video.mp4 results.csv 0.7

Compilation on Linux and Mac:
-----------------------

    cd <general_tracker_src>
    mkdir build
    cd build
    cmake ../ -DCMAKE_BUILD_TYPE=Release
    make

Compilation on Windows (with OpenCV installation):
-----------------------

Run-time dependencies on Windows:
-----------------------
