#include <fstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include "lm/lmmin.h"


void CalculateCost_RotPos2D(const double *params, int num_inputs, const void *inputs, double *fvec, int *info);

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

    TemplateTracker(int rad_x, int rad_y, unsigned int dof, int upscaling = 3, int metric = 1) :
        RotAngle(0.0),
        Scale(1.0),
        RadiusX(rad_x),
        RadiusY(rad_y),
        MatchThreshold(0.5f),
        MatchMetric(metric == 1 ? cv::TM_CCOEFF_NORMED : (metric == 2 ? cv::TM_SQDIFF_NORMED : (metric == 3 ? cv::TM_CCOEFF : (metric == 4 ? cv::TM_SQDIFF : cv::TM_CCOEFF_NORMED)))),
        DoF((dof >= 2 && dof <= 3) ? dof : 2),
        Upscaling(upscaling >= 0 ? upscaling : 0)
    {
    }

    void SetPosition(double x, double y)
    {
        PosX = x;
        PosY = y;
    }

    void GetPosition(double& x, double& y) const
    {
        x = PosX;
        y = PosY;
    }

    void GetRotation(double& angle) const
    {
        angle = RotAngle;
    }

    void GetScale(double& scale) const
    {
        scale = Scale;
    }

    int GetUpscaling() const
    {
        return Upscaling;
    }

    int GetMatchMetric() const
    {
        return MatchMetric;
    }

    const cv::Mat& GetGrayImage() const
    {
        return GrayImage;
    }

    const cv::Mat& GetUpscaledTemplate() const
    {
        return UpscaledTemplate;
    }

    cv::Mat& GetLMWarpedImage()
    {
        return LM_WarpedImage;
    }

    void SetMatchThreshold(float thrsh)
    {
        MatchThreshold = thrsh;
    }

    bool SetTemplate(const cv::Mat& tmplt)
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

        if (Upscaling > 0) {
            cv::resize(Template, UpscaledTemplate, cv::Size(), Upscaling, Upscaling, cv::INTER_LINEAR);
        }

        RotAngle = 0.0;
        Scale = 1.0;

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

        cv::Rect roi;
        cv::Mat image_roi;

        if (Upscaling > 0 && DoF > 2) {
            roi = cv::Rect(0, 0,
                           RadiusX * 2 + Template.cols,
                           RadiusY * 2 + Template.rows);
            cv::Point2f src[3];
            double pt0x = -0.5 * (roi.width - 1), pt0y = -0.5 * (roi.height - 1);
            double pt1x = -pt0x, pt1y = pt0y;
            double pt2x = pt0x, pt2y = -pt0y;
            src[0] = cv::Point2d(PosX + pt0x * std::cos(RotAngle) - pt0y * std::sin(RotAngle), PosY + pt0x * std::sin(RotAngle) + pt0y * std::cos(RotAngle));
            src[1] = cv::Point2d(PosX + pt1x * std::cos(RotAngle) - pt1y * std::sin(RotAngle), PosY + pt1x * std::sin(RotAngle) + pt1y * std::cos(RotAngle));
            src[2] = cv::Point2d(PosX + pt2x * std::cos(RotAngle) - pt2y * std::sin(RotAngle), PosY + pt2x * std::sin(RotAngle) + pt2y * std::cos(RotAngle));
            cv::Point2f dst[3];
            dst[0] = cv::Point2d(0.0, 0.0);
            dst[1] = cv::Point2d(roi.width - 1.0, 0.0);
            dst[2] = cv::Point2d(0.0, roi.height - 1.0);
            cv::Mat warp_mat = cv::getAffineTransform(src, dst);
            cv::warpAffine(GrayImage, WarpedGrayImage, warp_mat, roi.size(), cv::INTER_LINEAR);

            image_roi = WarpedGrayImage;
        } else {
            roi = cv::Rect(static_cast<int>(PosX) - RadiusX - Template.cols / 2,
                           static_cast<int>(PosY) - RadiusY - Template.rows / 2,
                           RadiusX * 2 + Template.cols,
                           RadiusY * 2 + Template.rows);
            roi = cv::Rect(0, 0, image.cols, image.rows) & roi;

            image_roi = cv::Mat(GrayImage, roi);
        }

//        cv::namedWindow("image_roi");
//        cv::imshow("image_roi", image_roi);

        int result_w = roi.width - Template.cols + 1;
        int result_h = roi.height - Template.rows + 1;
        if (result_w < 1 || result_h < 1) {
            return -2.0;
        }

        cv::Mat result(result_h, result_w, CV_32FC1);
        cv::matchTemplate(image_roi, Template, result, MatchMetric);

        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);

        if (Upscaling > 0 && DoF > 2) {
            double px = static_cast<double>(max_loc.x - RadiusX);
            double py = static_cast<double>(max_loc.y - RadiusY);
            PosX = PosX + px * std::cos(RotAngle) - py * std::sin(RotAngle);
            PosY = PosY + px * std::sin(RotAngle) + py * std::cos(RotAngle);

            // Save current values
            double tx = PosX, ty = PosY, tan = RotAngle, tsc = Scale;

            if (DoF == 3) Refine3DoF();

            cv::Point2f src[3];
            double pt0x = -0.5 * (Template.cols - 1), pt0y = -0.5 * (Template.rows - 1);
            double pt1x = -pt0x, pt1y = pt0y;
            double pt2x = pt0x, pt2y = -pt0y;
            src[0] = cv::Point2d(PosX + pt0x * std::cos(RotAngle) - pt0y * std::sin(RotAngle), PosY + pt0x * std::sin(RotAngle) + pt0y * std::cos(RotAngle));
            src[1] = cv::Point2d(PosX + pt1x * std::cos(RotAngle) - pt1y * std::sin(RotAngle), PosY + pt1x * std::sin(RotAngle) + pt1y * std::cos(RotAngle));
            src[2] = cv::Point2d(PosX + pt2x * std::cos(RotAngle) - pt2y * std::sin(RotAngle), PosY + pt2x * std::sin(RotAngle) + pt2y * std::cos(RotAngle));
            cv::Point2f dst[3];
            dst[0] = cv::Point2d(0.0, 0.0);
            dst[1] = cv::Point2d(Template.cols - 1.0, 0.0);
            dst[2] = cv::Point2d(0.0, Template.rows - 1.0);
            cv::Mat warp_mat = cv::getAffineTransform(src, dst);
            cv::warpAffine(GrayImage, WarpedGrayImage, warp_mat, Template.size(), cv::INTER_LINEAR);

            cv::Mat t_result(1, 1, CV_32FC1);
            cv::matchTemplate(Template, WarpedGrayImage, t_result, MatchMetric);

            double max_val2 = static_cast<double>(t_result.at<float>(0));
            if (max_val2 >= 0.999999 || std::abs(PosX - tx) > 5.0 || std::abs(PosX - tx) > 5.0) {
                // Error is too big, restore saved values
                PosX = tx; PosY = ty; RotAngle = tan; Scale = tsc;
            } else {
                max_val = max_val2;
            }
        } else {
            if (max_val >= static_cast<double>(MatchThreshold)) {
                PosX = static_cast<double>(max_loc.x + roi.x + Template.cols / 2);
                PosY = static_cast<double>(max_loc.y + roi.y + Template.rows / 2);
            }
        }

        return max_val;
    }

protected:
    bool Refine3DoF()
    {
        const int dof = 3;
        const int m = 4;

        double* params = new double[dof];

        params[0] = PosX;
        params[1] = PosY;
        params[2] = RotAngle * 0.1;

        LM_fvec = new double[m];
        LM_diag = new double[dof];
        LM_qtf  = new double[dof];
        LM_fjac = new double[dof * m];
        LM_wa1  = new double[dof];
        LM_wa2  = new double[dof];
        LM_wa3  = new double[dof];
        LM_wa4  = new double[m];
        LM_ipvt = new int [dof];

        double epsilon = 1e-6;
        int maxcall = 200;

        lm_control_struct control;
        control = lm_control_double;
        control.maxcall = maxcall;
        control.ftol = 1e-10;
        control.xtol = 1e-10;
        control.gtol = 1e-10;
        control.epsilon = epsilon;
//        control.stepbound = 100.0;
        control.printflags = 0;

        lm_status_struct status;
        status.info = 1;

        if (!control.scale_diag) {
            for (int j = 0; j < dof; j ++) LM_diag[j] = 1;
        }

        lm_lmdif(m,
                 dof,
                 params,
                 LM_fvec,
                 control.ftol,
                 control.xtol,
                 control.gtol,
                 control.maxcall * (dof + 1),
                 control.epsilon,
                 LM_diag,
                 (control.scale_diag ? 1 : 2),
                 control.stepbound,
                 &(status.info),
                 &(status.nfev),
                 LM_fjac,
                 LM_ipvt,
                 LM_qtf,
                 LM_wa1,
                 LM_wa2,
                 LM_wa3,
                 LM_wa4,
                 CalculateCost_RotPos2D,
                 lm_printout_std,
                 control.printflags,
                 this);

        PosX = params[0];
        PosY = params[1];
        RotAngle = params[2] * 10.0;

        delete [] LM_fvec;
        delete [] LM_diag;
        delete [] LM_qtf;
        delete [] LM_fjac;
        delete [] LM_wa1;
        delete [] LM_wa2;
        delete [] LM_wa3;
        delete [] LM_wa4;
        delete [] LM_ipvt;
        delete [] params;

        return true;
    }

    // Degrees of freedom
    double PosX;
    double PosY;
    double RotAngle;
    double Scale;

    int RadiusX;
    int RadiusY;
    cv::Mat Template;
    cv::Mat GrayImage;
    cv::Mat WarpedGrayImage;
    float MatchThreshold;
    const int MatchMetric;

    const unsigned int DoF;
    const int Upscaling;
    cv::Mat UpscaledTemplate;

    // LM solver buffers
    cv::Mat LM_WarpedImage;
    double *LM_fvec;
    double *LM_diag;
    double *LM_fjac;
    double *LM_qtf;
    double *LM_wa1;
    double *LM_wa2;
    double *LM_wa3;
    double *LM_wa4;
    int    *LM_ipvt;
};

void CalculateCost_RotPos2D(const double *params, int num_inputs, const void *inputs, double *fvec, int *info)
{
    TemplateTracker* parent = const_cast<TemplateTracker*>(reinterpret_cast<const TemplateTracker*>(inputs));
    if (!parent) return;

    const double upscaling = static_cast<double>(parent->GetUpscaling());
    const cv::Mat& image = parent->GetGrayImage();
    const cv::Mat& tmplt = parent->GetUpscaledTemplate();
    cv::Mat& warped_image = parent->GetLMWarpedImage();

    double pos_x = params[0];
    double pos_y = params[1];
    double rot_angle = params[2] * 10.0;

    cv::Point2f src[3];
    double pt0x = -0.5 * (tmplt.cols / upscaling - 1), pt0y = -0.5 * (tmplt.rows / upscaling - 1);
    double pt1x = -pt0x, pt1y = pt0y;
    double pt2x = pt0x, pt2y = -pt0y;
    src[0] = cv::Point2d(pos_x + pt0x * std::cos(rot_angle) - pt0y * std::sin(rot_angle), pos_y + pt0x * std::sin(rot_angle) + pt0y * std::cos(rot_angle));
    src[1] = cv::Point2d(pos_x + pt1x * std::cos(rot_angle) - pt1y * std::sin(rot_angle), pos_y + pt1x * std::sin(rot_angle) + pt1y * std::cos(rot_angle));
    src[2] = cv::Point2d(pos_x + pt2x * std::cos(rot_angle) - pt2y * std::sin(rot_angle), pos_y + pt2x * std::sin(rot_angle) + pt2y * std::cos(rot_angle));
    cv::Point2f dst[3];
    dst[0] = cv::Point2d(0.0, 0.0);
    dst[1] = cv::Point2d(tmplt.cols - 1.0, 0.0);
    dst[2] = cv::Point2d(0.0, tmplt.rows - 1.0);
    cv::Mat warp_mat = cv::getAffineTransform(src, dst);
    cv::warpAffine(image, warped_image, warp_mat, tmplt.size(), cv::INTER_LINEAR);

    cv::Mat result(1, 1, CV_32FC1);
    cv::matchTemplate(tmplt, warped_image, result, parent->GetMatchMetric());

//    cv::namedWindow("tmplt");
//    cv::imshow("tmplt", tmplt);
//    cv::namedWindow("warped_image");
//    cv::imshow("warped_image", warped_image);
//    cv::Mat diff;
//    cv::absdiff(tmplt, warped_image, diff);
//    cv::namedWindow("diff");
//    cv::imshow("diff", diff);
//    cv::waitKey(0);

    fvec[0] = 1.0 - static_cast<double>(result.at<float>(0));
    for (int i = 1; i < num_inputs; i ++) fvec[i] = fvec[0];
}

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

    TemplateTracker simple_tracker(20, 20, 3, 3, 1);
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
                               << static_cast<double>(roi.y) + static_cast<double>(roi.height) * 0.5
                               << ",0" << std::endl;

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
                    output << frame_cnt << ",0,0,0,0" << std::endl;
                }
            } else {
                double score = simple_tracker.Track(frame);
                double x, y, angle;
                simple_tracker.GetPosition(x, y);
                simple_tracker.GetRotation(angle);
                if (score >= -1.0 &&
                    x >= 0.0 && y >= 0.0 &&
                    x <= static_cast<double>(frame.cols - 1) && y <= static_cast<double>(frame.rows - 1)) {

                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(3) << score;

                    if (score >= match_threshold) {
                        // Good match
                        cv::putText(frame_scaled, ss.str(), cv::Point(3, 23), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);

                        output << frame_cnt << ",1," << std::fixed << std::setprecision(2) << x << "," << y << "," << angle * 180.0 / M_PI << std::endl;

                        cv::Point2f src[4];
                        double pt0x = -0.5 * (tmplt.cols - 1), pt0y = -0.5 * (tmplt.rows - 1);
                        double pt1x = -pt0x, pt1y = pt0y;
                        double pt2x = pt1x, pt2y = -pt0y;
                        double pt3x = pt0x, pt3y = pt2y;
                        src[0] = cv::Point2d(x + pt0x * std::cos(angle) - pt0y * std::sin(angle), y + pt0x * std::sin(angle) + pt0y * std::cos(angle)) * real_display_scale;
                        src[1] = cv::Point2d(x + pt1x * std::cos(angle) - pt1y * std::sin(angle), y + pt1x * std::sin(angle) + pt1y * std::cos(angle)) * real_display_scale;
                        src[2] = cv::Point2d(x + pt2x * std::cos(angle) - pt2y * std::sin(angle), y + pt2x * std::sin(angle) + pt2y * std::cos(angle)) * real_display_scale;
                        src[3] = cv::Point2d(x + pt3x * std::cos(angle) - pt3y * std::sin(angle), y + pt3x * std::sin(angle) + pt3y * std::cos(angle)) * real_display_scale;

                        cv::line(frame_scaled, src[0], src[1], cv::Scalar(0, 255, 0 ), 2);
                        cv::line(frame_scaled, src[1], src[2], cv::Scalar(0, 255, 0 ), 2);
                        cv::line(frame_scaled, src[2], src[3], cv::Scalar(0, 255, 0 ), 2);
                        cv::line(frame_scaled, src[3], src[0], cv::Scalar(0, 255, 0 ), 2);

                        prev_action_skip = false;
                    } else {
//                        std::cout << ". failure: score=" << score << "; pos=(" << x << ", " << y << "); angle=" << angle * 180.0 / M_PI << std::endl;

                        // Bad match
                        cv::putText(frame_scaled, ss.str(), cv::Point(3, 23), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

                        tracker_ok = false;
                    }
                } else {
//                    std::cout << ". failure: score=" << score << "; pos=(" << x << ", " << y << "); angle=" << angle * 180.0 / M_PI << std::endl;

                    // Tracker failed
                    tracker_ok = false;
                    output << frame_cnt << ",0,0,0,0" << std::endl;
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

                output << frame_cnt << ",2," << roi.x + roi.width * 0.5 << "," << roi.y + roi.height * 0.5 << ",0" << std::endl;

                if (tracker_type == 0) {
                    tracker.release();
                    tracker = cv::TrackerCSRT::create();
//                    tracker = cv::Tracker::create("MIL");
                    tracker->init(frame, roi);
                } else {
                    simple_tracker.SetTemplate(tmplt);
                    simple_tracker.SetPosition(static_cast<int>(roi.x + roi.width * 0.5), static_cast<int>(roi.y + roi.height * 0.5));
                }

                tracker_ok = true;
                prev_action_skip = false;

                std::cout << ". Press ESC to exit" << std::endl;
            } else {
                // Skip frame
                output << frame_cnt << ",0,0,0,0" << std::endl;
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
