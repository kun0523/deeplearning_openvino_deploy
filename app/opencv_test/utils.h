#include <iostream>
#include <opencv2/opencv.hpp>

struct MatData{
    size_t rows;
    size_t cols;
    int type;
    uchar* data=nullptr;
};

void getMat(MatData* mat);

void getMatArray(MatData* mats, size_t& num);

void destroyMat(MatData* mat, const size_t& num=0);