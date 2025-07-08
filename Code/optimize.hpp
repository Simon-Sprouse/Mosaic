#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

#include <opencv2/opencv.hpp>


namespace Optimize { 

    double reward(cv::Mat segment_image, cv::Point center, double size, double theta, double decay_rate);
    double rewardFromCanny(const cv::Mat& edge_image, cv::Point2f center, double size, double theta_deg, double decay_rate);
    void findBestTheta();

}

#endif