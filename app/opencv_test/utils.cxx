#include "utils.h"


void getMat(MatData* mat){
    cv::Mat m{300, 300, CV_8UC3, cv::Scalar(255, 0, 0)};
    mat->rows = m.rows;
    mat->cols = m.cols;
    mat->type = m.type();
    mat->data = new uchar[m.total()*m.elemSize()];
    memcpy(mat->data, m.data, m.total()*m.elemSize());
}

void getMatArray(MatData* mats, size_t& num){

    cv::RNG rng(cv::getTickCount());    
    int mat_size = 1024;
    for(int i{}; i<num; ++i){
        cv::Mat t(mat_size, mat_size, CV_8UC3, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        // cv::randu(t, cv::Scalar(0,0,0), cv::Scalar(255,255,255)); // 噪声图片
        mats[i].rows = t.rows;
        mats[i].cols = t.cols;
        mats[i].type = t.type();
        mats[i].data = new uchar[t.total()*t.elemSize()];
        memcpy(mats[i].data, t.data, t.total()*t.elemSize());
    }

}

void destroyMat(MatData* mat, const size_t& num){
    if(num == 0){
        delete[] mat->data;
        mat->data=nullptr;
    }else{
        for(int i{}; i<num; ++i){
            delete[] mat[i].data;
            mat[i].data = nullptr;
        }
    }

}