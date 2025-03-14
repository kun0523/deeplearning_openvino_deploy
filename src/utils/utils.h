#include <iostream>
#include <opencv2/opencv.hpp>


class TickTock{
    private void start;
    private void stop;

    public void Start();
    public void Stop();
    public size_t Spend();
}

class Log{
    
}

/// @brief 将图片切成多个块
/// @return 
int splitImage(const cv::Mat& img, size_t patch_size, std::vector<cv::Mat> patches);