# General Tracker
by Balazs Vagvolgyi

Simple utility for tracking a single object.

User needs to manually select the ROI of the patch to be tracked.

Frames may be skipped when the object is lost by the tracker or is not visible.

Output is generated in CSV format:

    <frame-num>,<valid-flag>,<x-coordinate>,<y-coordinate>

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
