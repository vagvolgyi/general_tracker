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
    if (argc < 5) {
        std::cout << ". Usage:    track <video-file> <output-file> <match-threshold> <dislpay-scale>" << std::endl
                  << ". Example:  track video.mp4 results.csv 0.7 0.5" << std::endl;
        return 1;
    }

    std::string video_file = argv[1];
    std::string output_file = argv[2];

    std::stringstream match_threshold_ss(argv[3]);
    float match_threshold;
    match_threshold_ss >> match_threshold;

    std::stringstream display_scale_ss(argv[4]);
    float display_scale;
    display_scale_ss >> display_scale;

    std::cout << ". video-file      = \"" << video_file << "\"" << std::endl
              << ". output-file     = \"" << output_file << "\"" << std::endl
              << ". match-threshold = " << std::fixed << match_threshold << std::endl
              << ". display-scale = " << std::fixed << display_scale << std::endl;

//    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
//    cv::Ptr<cv::Tracker> tracker = cv::TrackerMIL::create();
    cv::Ptr<cv::Tracker> tracker;// = cv::TrackerCSRT::create();

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

    std::ofstream output(output_file.c_str(), std::ofstream::out);
    if (output.is_open() == false) {
        std::cerr << ". Error - could not create output file" << std::endl;
        return -2;
    }

    cv::Rect2d roi;
    cv::Mat frame_scaled, tmplt, patch_scaled;
    bool tracker_ok = false, prev_action_skip = true;

    std::cout << ". Press SPACE to select, ESC to exit, or any other key to fast-forward." << std::endl;

    int frame_cnt = 0;

    while (video.read(frame))
    {
        const int disp_width = static_cast<int>(round(frame.cols * display_scale));
        const int disp_height = static_cast<int>(round(frame.rows * display_scale));
        const double real_display_scale = static_cast<double>(disp_width) / static_cast<double>(frame.cols);

        cv::resize(frame, frame_scaled, cv::Size(disp_width, disp_height), 0, 0, cv::INTER_LINEAR);

        if (tracker_ok) {
            if (tracker->update(frame, roi) &&
                roi.x >= 0.0 &&
                roi.y >= 0.0 &&
                roi.x + roi.width <= frame.cols &&
                roi.y + roi.height <= frame.rows) {

                // Check quality of the match
                cv::resize(cv::Mat(frame, roi), patch_scaled, tmplt.size(), 0, 0, cv::INTER_LINEAR);
                float match = CompareToTemplate(patch_scaled, tmplt);

                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << match;

                if (match >= match_threshold) {
                    // Good match
                    cv::putText(frame_scaled, ss.str(), cv::Point(3, 23), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);

                    output << frame_cnt << ",1,"
                           << static_cast<double>(roi.x) + static_cast<double>(roi.width) * 0.5
                           << ","
                           << static_cast<double>(roi.y) + static_cast<double>(roi.height) * 0.5 << std::endl;

                    roi.x = roi.x * real_display_scale;
                    roi.y = roi.y * real_display_scale;
                    roi.width = roi.width * real_display_scale;
                    roi.height = roi.height * real_display_scale;
                    cv::rectangle(frame_scaled, roi, cv::Scalar(0, 255, 0 ), 2, 1);

                    prev_action_skip = false;
                } else {
                    // Bad match
                    cv::putText(frame_scaled, ss.str(), cv::Point(3, 23), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

                    tracker_ok = false;
                }
            } else {
                // Tracker failed
                tracker_ok = false;
                output << frame_cnt << ",0,0,0" << std::endl;
            }
        }

        std::stringstream ss;
        ss << frame_cnt;
        cv::putText(frame_scaled, ss.str(), cv::Point(3, 11), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        cv::imshow("Tracking", frame_scaled);

        if (!tracker_ok) {
            if (!prev_action_skip) {
                std::cout << ". Tracking failed, press SPACE to re-select, ESC to exit, or any other key to fast-forward." << std::endl;
            }
            int k = cv::waitKey(0);
            if (k == 27) {
                // Exit on ESC
                break;
            } else if (k == ' ') {
                // Re-select
                roi = cv::selectROI("Tracking", frame_scaled, true, true);

                roi.x /= real_display_scale;
                roi.y /= real_display_scale;
                roi.width /= real_display_scale;
                roi.height /= real_display_scale;

                cv::Mat(frame, roi).copyTo(tmplt);

                output << frame_cnt << ",2," << roi.x + roi.width * 0.5 << "," << roi.y + roi.height * 0.5 << std::endl;

                tracker.release();
                tracker = cv::TrackerCSRT::create();
                tracker->init(frame, roi);

                tracker_ok = true;
                prev_action_skip = false;

                std::cout << ". Press ESC to exit" << std::endl;
            } else {
                // Skip frame
                output << frame_cnt << ",0,0,0" << std::endl;
                prev_action_skip = true;
            }

            frame_cnt ++;
            continue;
        }

        int k = cv::waitKey(1);
        if (k == 27) {
            // Exit on ESC
            break;
        }

        frame_cnt ++;
    }

    std::cout << ". Finished" << std::endl;
    return 0;
}
