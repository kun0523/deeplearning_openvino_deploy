#include <iostream>
#include <opencv2/opencv.hpp>

/// @brief 将图片切成多个块
/// @return 
int splitImage(const cv::Mat& img, size_t patch_size, std::vector<cv::Mat> patches);