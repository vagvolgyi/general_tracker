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

class TemplateTracker
{
public:
    TemplateTracker() = delete;

    TemplateTracker(int rad_x, int rad_y) :
        RadiusX(rad_x),
        RadiusY(rad_y),
        MatchThreshold(0.5f)
    {
    }

    void SetPosition(int x, int y)
    {
        PosX = x;
        PosY = y;
    }

    void GetPosition(int& x, int& y)
    {
        x = PosX;
        y = PosY;
    }

    void SetMatchThreshold(float thrsh)
    {
        MatchThreshold = thrsh;
    }

    bool SetTemplate(const cv::Mat& tmplt, int cx, int cy)
    {
        if (tmplt.empty()) {
            std::cerr << ". TemplateTracker::SetTemplate: error - empty template image" << std::endl;
            return false;
        }
        if (tmplt.cols % 2 != 1 || tmplt.rows % 2 != 1) {
            std::cerr << ". TemplateTracker::SetTemplate: error - template image dimensions must be odd numbers" << std::endl;
            return false;
        }
        if (tmplt.elemSize() == 3) {
            cv::cvtColor(tmplt, Template, cv::COLOR_RGB2GRAY);
        } else if (tmplt.elemSize() == 1) {
            tmplt.copyTo(Template);
        } else {
            std::cerr << ". TemplateTracker::SetTemplate: error - invalid template image" << std::endl;
            return false;
        }
        TemplateCenterX = cx;
        TemplateCenterY = cy;

        return true;
    }

    double Track(const cv::Mat& image)
    {
        if (image.empty()) {
            std::cerr << ". TemplateTracker::Track: error - empty input image" << std::endl;
            return -2.0;
        }
        if (Template.empty()) {
            std::cerr << ". TemplateTracker::Track: error - empty template image" << std::endl;
            return -2.0;
        }
        if (image.elemSize() == 3) {
            cv::cvtColor(image, GrayImage, cv::COLOR_RGB2GRAY);
        } else if (image.elemSize() == 1) {
            GrayImage = image;
        } else {
            std::cerr << ". TemplateTracker::Track: error - invalid input image" << std::endl;
            return -2.0;
        }

        cv::Rect crop(PosX - RadiusX - TemplateCenterX,
                      PosY - RadiusY - TemplateCenterY,
                      RadiusX * 2 + Template.cols,
                      RadiusY * 2 + Template.rows);
        crop = cv::Rect(0, 0, image.cols, image.rows) & crop;

        int result_w = crop.width - Template.cols + 1;
        int result_h = crop.height - Template.rows + 1;
        if (result_w < 1 || result_h < 1) {
            return -2.0;
        }

        cv::Mat image_roi(GrayImage, crop);

        cv::Mat result(result_h, result_w, CV_32FC1);
        cv::matchTemplate(image_roi, Template, result, cv::TM_CCOEFF_NORMED);

        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);

        if (max_val >= static_cast<double>(MatchThreshold)) {
            PosX = max_loc.x + crop.x + TemplateCenterX;
            PosY = max_loc.y + crop.y + TemplateCenterY;
        }

        return max_val;
    }

protected:
    int PosX;
    int PosY;
    int RadiusX;
    int RadiusY;
    cv::Mat Template;
    int TemplateCenterX;
    int TemplateCenterY;
    cv::Mat GrayImage;
    float MatchThreshold;
};

int main(int argc, char **argv)
{
    if (argc < 5) {
        std::cout << ". Usage:    track <video-file> <output-file> <match-threshold> <display-scale>" << std::endl
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

    int tracker_type = 0;
    if (argc >= 6) {
        std::stringstream tracker_type_ss(argv[5]);
        tracker_type_ss >> tracker_type;
        if (tracker_type <= 0) tracker_type = 0;
        else tracker_type = 1;
    }

    std::cout << ". video-file      = \"" << video_file << "\"" << std::endl
              << ". output-file     = \"" << output_file << "\"" << std::endl
              << ". match-threshold = " << std::fixed << match_threshold << std::endl
              << ". display-scale = " << std::fixed << display_scale << std::endl;

//    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
//    cv::Ptr<cv::Tracker> tracker = cv::TrackerMIL::create();
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
//    cv::Ptr<cv::Tracker> tracker = cv::Tracker::create("MIL");

    TemplateTracker simple_tracker(20, 20);
    simple_tracker.SetMatchThreshold(match_threshold);

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
            if (tracker_type == 0) {
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
            } else {
                double score = simple_tracker.Track(frame);
                int x, y;
                simple_tracker.GetPosition(x, y);
                if (score >= -1.0 &&
                    x >= 0 && y >= 0 &&
                    x < frame.cols && y < frame.rows) {

                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(3) << score;

                    if (score >= match_threshold) {
                        // Good match
                        cv::putText(frame_scaled, ss.str(), cv::Point(3, 23), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);

                        output << frame_cnt << ",1," << x << "," << y << std::endl;

                        roi.x = (x - tmplt.cols / 2) * real_display_scale;
                        roi.y = (y - tmplt.rows / 2) * real_display_scale;
                        roi.width = tmplt.cols * real_display_scale;
                        roi.height = tmplt.rows * real_display_scale;
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
                // Make sure template dimensions are even
                if (static_cast<int>(roi.width) % 2 == 0) roi.width -= 1;
                if (static_cast<int>(roi.height) % 2 == 0) roi.height -= 1;

                cv::Mat(frame, roi).copyTo(tmplt);

                output << frame_cnt << ",2," << roi.x + roi.width * 0.5 << "," << roi.y + roi.height * 0.5 << std::endl;

                if (tracker_type == 0) {
                    tracker.release();
                    tracker = cv::Tracker::create("MIL");
                    tracker->init(frame, roi);
                } else {
                    simple_tracker.SetTemplate(tmplt, tmplt.cols / 2, tmplt.rows / 2);
                    simple_tracker.SetPosition(static_cast<int>(roi.x + roi.width * 0.5), static_cast<int>(roi.y + roi.height * 0.5));
                }

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
