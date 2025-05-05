#include <inference.h>

void testInferenceSpeed(){
    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11\yolo11n.onnx)"};
    std::vector<std::string> image_files{
        // R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",  // 1560*1600  168ms
        // R"(E:\DataSets\imageNet\n02086240_Shih-Tzu.JPEG)",  // 1280*1024  78ms
        // R"(E:\le_trt\models\dog.jpg)",  // 500*374  65ms

        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)"
    };
    std::vector<double> costs{};
    double total_costs{};
    char msg[1024];
    initModel(onnx_file.c_str(), msg);
    for(const auto& file:image_files){
        int det_num;
        auto start = std::chrono::high_resolution_clock::now();
        auto r = doInferenceByImgPth(file.c_str(), nullptr, 0.3f, det_num, msg);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> spend = end - start;
        total_costs += spend.count();
        costs.push_back(spend.count());
        std::cout << "Result: " << r->get_info() << std::endl;
        std::cout << "cost: " << spend.count() << "ms" << std::endl;
        freeResult(r, det_num);
    }
    std::cout << "Average speed: " << total_costs/image_files.size() << "ms" << std::endl;    
}


int main(){
    std::cout << "OpenVINO Detection Demo" << std::endl;

    testInferenceSpeed();  // openvino 比 onnxruntime 快差不多一倍
    return 0;

    std::string onnx_pth = R"(E:\Pretrained_models\YOLOv11\yolo11n.onnx)";
    char msg[1024];
    initModel(onnx_pth.c_str(), msg);
    std::cout << msg << std::endl;

    std::cout << "-----------------------------" << std::endl;
    std::string img_file = R"(E:\le_pp3\Paddle\test\dataset\cat.jpg)";
    int det_num{};
    auto r1 = doInferenceByImgPth(img_file.c_str(), nullptr, 0.3f, det_num, msg);
    std::cout << msg << std::endl;
    std::cout << r1->get_info() << std::endl;
    cv::Mat img = cv::imread(img_file);
    for(int i{}; i<det_num; ++i){
        cv::rectangle(img, cv::Rect(cv::Point(r1[i].tl_x, r1[i].tl_y), cv::Point(r1[i].br_x, r1[i].br_y)), cv::Scalar(0,0,255));
    }

    cv::imshow("test", img);
    cv::waitKey(0);

    // std::cout << "-----------------------------" << std::endl;
    // cv::Mat img = cv::imread(img_file);
    // auto r2 = doInferenceByImgMat(img, 0.3f, det_num, msg);
    // std::cout << msg << std::endl;

    // std::cout << "-----------------------------" << std::endl;
    // auto r3 = doInferenceBy3chImg(img.ptr(), img.rows, img.cols, 0.3f, det_num, msg);
    // std::cout << msg << std::endl;




    return 0;
}