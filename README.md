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
Download and install OpenCV (4.0.1) for Windows (please see the following link for further instructions: https://docs.opencv.org/3.4.3/d3/d52/tutorial_windows_install.html)

Download opencv_contrib (4.0.1) form the following link: https://github.com/opencv/opencv_contrib/releases 

Load and configure OpenCV project in CMake, as follows:

- In the fields "Where is the source code" and "where to build the binaries", specify the path to the source folder and build folders for the OpenCV package, repectively;

- Click on "Add Entry"; 

- Enter "OpenCV_DIR" for the Name;

- From the Type dropdown, set Type to be PATH;

- Leave the Value unassigned for now and press OK. Then "Configure" and "Generate";

- After successful configuration, set the PATH value. For this purpose, in the red section, click on OpenCV_DIR to modify its value by entering the the directory containing the CMake configuration file for OpenCV (the path to *\opencv_contrib-4.0.1\modules, where * indicates the path where opencv_contrib-4.0.1 is located);

- "Configure" and "Generate" another time.


After generating the CMAke file, open the project in Visual Studio and Build it.

Before running the code, you will need to either copy all the DLL files to the folder containing the .exe file, or add the location of the DLL files to environment PATHs.




Run-time dependencies on Windows:
-----------------------
