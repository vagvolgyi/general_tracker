#include <fstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>


float CompareToTemplate(const cv::Mat& patch, const cv::Mat& tmplt)
{
    cv::Mat result(1, 1, CV_32FC1);
    cv::matchTemplate(patch, tmplt, result, cv::TM_CCOEFF_NORMED);

    return result.at<float>(0, 0);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cout << ". Usage:    track <video-file> <output-file> <match-threshold>" << std::endl
                  << ". Example:  track video.mp4 results.csv 0.7" << std::endl;
        return 1;
    }

    std::string video_file = argv[1];
    std::string output_file = argv[2];
    float match_threshold = std::stof(argv[3]);

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

    cv::Mat frame_small;
    cv::resize(frame, frame_small, cv::Size(frame.cols / 2, frame.rows / 2));

    cv::Rect2d roi = cv::selectROI("Tracking", frame_small, true, true);

    cv::resize(frame, frame_small, cv::Size(frame.cols / 2, frame.rows / 2));
    cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2, 1);
    cv::imshow("Tracking", frame_small);

    roi.x *= 2;
    roi.y *= 2;
    roi.width *= 2;
    roi.height *= 2;

    cv::Mat tmplt;
    cv::Mat(frame, roi).copyTo(tmplt);

    cv::Mat patch_scaled;

    output << "1,"
           << static_cast<double>(roi.x) + static_cast<double>(roi.width) * 0.5
           << ","
           << static_cast<double>(roi.y) + static_cast<double>(roi.height) * 0.5 << std::endl;

    tracker->init(frame, roi);

    bool tracker_ok = true, prev_action_skip = false;

    std::cout << ". Press ESC to exit" << std::endl;

    while (video.read(frame))
    {
        cv::resize(frame, frame_small, cv::Size(frame.cols / 2, frame.rows / 2));

        if (tracker_ok) {
            if (tracker->update(frame, roi)) {
                // Check quality of the match
                cv::resize(cv::Mat(frame, roi), patch_scaled, tmplt.size(), cv::INTER_LINEAR);
                float match = CompareToTemplate(patch_scaled, tmplt);

                if (match >= match_threshold) {
                    // Good match
                    output << "1,"
                           << static_cast<double>(roi.x) + static_cast<double>(roi.width) * 0.5
                           << ","
                           << static_cast<double>(roi.y) + static_cast<double>(roi.height) * 0.5 << std::endl;

                    roi.x = (roi.x + 1) / 2;
                    roi.y = (roi.y + 1) / 2;
                    roi.width = (roi.width + 1) / 2;
                    roi.height = (roi.height + 1) / 2;
                    cv::rectangle(frame_small, roi, cv::Scalar(0, 255, 0 ), 2, 1);

                    prev_action_skip = false;
                } else {
                    // Bad match
                    tracker_ok = false;
                }
            } else {
                // Tracker failed
                tracker_ok = false;
                output << "0,0,0" << std::endl;
            }
        }

        cv::imshow("Tracking", frame_small);

        if (!tracker_ok) {
            if (!prev_action_skip) {
                std::cerr << ". Tracking failed, press SPACE to re-select, ESC to exit, or any other key to fast-forward." << std::endl;
            }
            int k = cv::waitKey(0);
            if (k == 27) {
                // Exit on ESC
                break;
            } else if (k == ' ') {
                // Re-select
                roi = cv::selectROI("Tracking", frame_small, true, true);

                roi.x *= 2;
                roi.y *= 2;
                roi.width *= 2;
                roi.height *= 2;

                cv::Mat(frame, roi).copyTo(tmplt);

                output << "1,"
                       << static_cast<double>(roi.x) + static_cast<double>(roi.width) * 0.5
                       << ","
                       << static_cast<double>(roi.y) + static_cast<double>(roi.height) * 0.5 << std::endl;

                tracker->init(frame, roi);

                tracker_ok = true;
                prev_action_skip = false;

                std::cout << ". Press ESC to exit" << std::endl;
            } else {
                // Skip frame
                output << "0,0,0" << std::endl;
                prev_action_skip = true;
            }
            continue;
        }

        int k = cv::waitKey(1);
        if (k == 27) {
            // Exit on ESC
            break;
        }
    }

    std::cout << ". Finished" << std::endl;
    return 0;
}
