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
1. Download and install OpenCV (4.0.1) for Windows by following these instructions: https://docs.opencv.org/3.4.3/d3/d52/tutorial_windows_install.html

2. Download opencv_contrib (4.0.1): https://github.com/opencv/opencv_contrib/releases 

3. Generate a Visual Studio solution for OpenCV using CMake, as follows:

   1. In CMake, in the fields "Where is the source code" and "Where to build the binaries", specify the path to the source folder and build folder for the OpenCV package, repectively. These will look like *\opencv-4.0.1\sources and *\opencv-4.0.1\build, where * indicates the folder path containing opencv-4.0.1.

   2. Click "Configure". If asked, select the 64-bit version of your Visual Studio.

   3. Click "Add Entry".

   4. Enter "OPENCV_EXTRA_MODULES_PATH" for the Name.

   5. Select PATH for Type.

   6. Leave the Value unassigned for now and press "OK" then "Configure".

   7. After successful configuration, set the PATH value. For this purpose, in the red section, click on OPENCV_EXTRA_MODULES_PATH to modify its value then select the directory containing the OpenCV Contrib modules (*\opencv_contrib-4.0.1\modules, where * indicates the folder path containing opencv_contrib-4.0.1).

   8. "Configure" again, then "Generate".

4. Open the OpenCV Visual Studio Solution and compile it (Build All).

5. Create a build directory for general_tracker.

6. Generate a Visual Studio solution for general_tracker using CMake, as follows:

   1. Set source and build directories.
   
   2. "Configure".
   
   3. Set "OpenCV_DIR" to the build directory of your OpenCV installation.
   
   4. "Configure", then "Generate".

7. (Optional) Select Release mode in the Debug/Release drop down list.

8. Open the general_tracker Visual Studio Solution and build it.

Run-time dependencies on Windows:
-----------------------

Before running the code, you will need to either copy all the OpenCV DLL files to the folder containing the track.exe file, or add the location of the DLL files to "PATH" environment variable.
