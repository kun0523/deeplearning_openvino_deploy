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
    destroyModel();
}

int main(){
    std::cout << "OnnxRuntime Detection Demo" << std::endl;

    for(int i{}; i<100; i++)
        testInferenceSpeed();  // 有概率推理错误，待排查是什么原因！！
    return 0;

    char msg[1024];
    std::string onnx_file = R"(E:\Pretrained_models\YOLOv11\yolo11n.onnx)";
    initModel(onnx_file.c_str(), msg);

    int num{};
    std::string img_file = R"(E:\le_pp3\Paddle\test\dataset\cat.jpg)";
    DET_RES* r1 = doInferenceByImgPth(img_file.c_str(), nullptr, 0.5f, num, msg);
    for(int i{}; i<num; ++i){
        std::cout << r1[i].get_info() << std::endl;
    }

    std::cout << "-----------------------------------" << std::endl;
    auto img = cv::imread(img_file);
    DET_RES* r2 = doInferenceByImgMat(img, 0.5f, num, msg);
    for(int i{}; i<num; ++i){
        std::cout << r2[i].get_info() << std::endl;
    }

    std::cout << "-----------------------------------" << std::endl;
    DET_RES* r3 = doInferenceBy3chImg(img.ptr(), img.rows, img.cols, 0.5f, num, msg);
    for(int i{}; i<num; ++i){
        std::cout << r3[i].get_info() << std::endl;
    }

    // cv::imshow("test", img);
    // cv::waitKey(0);


    // run(img_file.c_str(), onnx_file.c_str());

    return 0;
}