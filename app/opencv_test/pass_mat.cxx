#include "utils.h"


// 在MatData下定义一个 uchar* 指针 用于操作 Mat 数据
int main(){
    std::cout << "Test OpenCV Demo" << std::endl;

    // 从函数接收一个 Mat
    MatData d1;
    getMat(&d1);
    cv::Mat m(d1.rows, d1.cols, d1.type, d1.data);
    // cv::imshow("Mat", m);
    // cv::waitKey(0);
    if(d1.data!=nullptr) std::cout << "no nullptr" << std::endl;
    destroyMat(&d1);
    if(d1.data==nullptr) std::cout << "is nullptr" << std::endl;


    // 从函数接收一组 Mat
    MatData mats[5000];
    size_t num{sizeof(mats)/sizeof(MatData)};
    getMatArray(mats, num);
    std::cout << "get mat num: " << num << std::endl;
    for(int i{}; i<num; ++i){
        cv::Mat t(mats[i].rows, mats[i].cols, mats[i].type, mats[i].data);
        // cv::imshow("MultiMat_"+std::to_string(i), t);
        // cv::waitKey(0);
    }

    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "-----------------" << std::endl;
    destroyMat(mats, num);
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "Job Done" << std::endl;


    return 0;
}