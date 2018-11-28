#include <fstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>


int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cout << ". Usage:    track <video-file> <output-file>" << std::endl
                  << ". Example:  track video.mp4 results.csv" << std::endl;
        return 1;
    }

    std::string video_file = argv[1];
    std::string output_file = argv[2];

//    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
//    cv::Ptr<cv::Tracker> tracker = cv::TrackerMIL::create();
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

    cv::VideoCapture video(video_file);

    if (!video.isOpened()) {
        std::cerr << ". Error - could not open video file" << std::endl;
        return -1;
    }

    cv::Mat frame;
    if (video.read(frame) == false) {
        std::cerr << ". Error - could not read video file" << std::endl;
        return -1;
    }

    std::ofstream output(output_file, std::ofstream::out);
    if (output.is_open() == false) {
        std::cerr << ". Error - could not create output file" << std::endl;
        return -2;
    }

    std::cout << ". Press ESC to exit" << std::endl;

    cv::Rect2d roi = cv::selectROI("Tracking", frame, true, true);

    cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2, 1);
    output << "1,"
           << static_cast<double>(roi.x) + static_cast<double>(roi.width) * 0.5
           << ","
           << static_cast<double>(roi.y) + static_cast<double>(roi.height) * 0.5 << std::endl;

    cv::imshow("Tracking", frame);
    tracker->init(frame, roi);

    while (video.read(frame))
    {
        if (tracker->update(frame, roi)) {
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0 ), 2, 1);
            output << "1,"
                   << static_cast<double>(roi.x) + static_cast<double>(roi.width) * 0.5
                   << ","
                   << static_cast<double>(roi.y) + static_cast<double>(roi.height) * 0.5 << std::endl;
        } else {
            cv::rectangle(frame, roi, cv::Scalar(0, 0, 255), 2, 1);
            output << "0,0,0" << std::endl;
        }

        cv::imshow("Tracking", frame);

        // Exit on ESC
        int k = cv::waitKey(1);
        if(k == 27) {
            break;
        }
    }

    std::cout << ". Finished" << std::endl;
    return 0;
}
