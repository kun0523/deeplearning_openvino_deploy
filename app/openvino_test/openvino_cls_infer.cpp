#include "inference.h"
#include <execution>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>

struct Item{
    std::filesystem::path file_path{};
    CLS_RES result;

    Item(std::string p, CLS_RES res):file_path(p), result(res){};
};

void test_parallel(){
    
    std::string onnx_pth = R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls.onnx)";
    char msg[1024];
    initModel(onnx_pth.c_str(), msg);

    // auto res = doInferenceByImgPth(R"(E:\DataSets\imageNet\n01496331_electric_ray.JPEG)", model_ptr, nullptr, msg);
    // std::cout << res.cls << " " << res.confidence << std::endl;

    std::vector<Item> images {
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01440764_tench.JPEG)", CLS_RES(0,0)},
        {R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)", CLS_RES(0,0)}, 
        {R"(E:\DataSets\imageNet\n01484850_great_white_shark.JPEG)", CLS_RES(0,0)},        
    };

    std::cout << "Inference One By One: " << std::endl;
    auto start_s = std::chrono::high_resolution_clock::now();
    for(const auto& i:images){
        auto ret = doInferenceByImgPth(i.file_path.string().c_str(), nullptr, msg);
        // std::cout << "Result: " << ret.cls << " " << ret.confidence << std::endl;
    }
    auto end_s = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> spend_s = end_s-start_s;
    std::cout << "Average Cost: " << spend_s.count()/images.size() << "ms" << std::endl;

    std::cout << "Inference Parallel: " << std::endl;
    auto start_p = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::par, images.begin(), images.end(), [&](Item& item){ 
        item.result = doInferenceByImgPth(item.file_path.string().c_str(), nullptr, msg);
    });
    // for(const auto& i:images){
    //     std::cout << "Result: " << i.result.cls << " " << i.result.confidence << std::endl;
    // }
    auto end_p = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> spend_p = end_p - start_p;
    std::cout << "Parallel Average Cost: " << spend_p.count()/images.size() << "ms" << std::endl;
    std::cout << "Job Done" << std::endl;
}

std::string classes[2]{"NG", "OK"};
std::mutex log_mutex;
std::ofstream log_file{"log.csv", std::ios::beg};
void process_data(Item& item, char* msg){
    item.result = doInferenceByImgPth(item.file_path.string().c_str(), nullptr, msg);
    // std::cout << item.file_path.string() << " Result: " << item.result.cls << " " << item.result.confidence << std::endl;
    std::lock_guard<std::mutex> lock(log_mutex);
    log_file << item.file_path.string() << "," << classes[item.result.cls] << "," << item.result.confidence << "\n";
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 如果没有停顿会把CPU跑满
}

void test_iqc_proj(const std::string image_dir, const std::string onnx_pth, const std::string file_type){
    std::stringstream buffer;
    std::vector<Item> images;
    std::regex date_pattern{R"(\d{4}\.\d\.\d{1,2})"};
    std::regex plate_pattern{R"([DU]\d(_\d)*)"};
    for(const auto& file:std::filesystem::recursive_directory_iterator(image_dir)){
        if (file.path().has_extension() && file.path().string().find("NG") != std::string::npos && file.path().extension() == file_type) {
            images.emplace_back(file.path().string(), CLS_RES(-1, -1));

            std::string tmp = file.path().string();
            std::smatch date_match, plate_match;
            if(std::regex_search(tmp, date_match, date_pattern) && std::regex_search(tmp, plate_match, plate_pattern)){
                buffer << file.path().string() << "," << date_match.str() <<","<< plate_match.str() << "\n";
            }else{
                buffer << file.path().string() << "," << "date not match" << "," << "plate not match" << "\n";
            }

            if(buffer.str().size()>(1024*1024)){
                log_file << buffer.str();
                buffer.clear();buffer.str("");
            }
        } 
    }
    log_file << buffer.str();
    std::cout << "Image Scan Finished. Image num: " << images.size() << std::endl;

    char msg[1024];
    initModel(onnx_pth.c_str(), msg);

    auto start_p = std::chrono::high_resolution_clock::now();

    // for(int i{}; i<images.size(); ++i){
    //     process_data(images[i], model_ptr, msg);
    // }

    std::for_each(std::execution::par, images.begin(), images.end(), [&](Item& item){ 
        process_data(item, msg);
    });

    auto end_p = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> spend_p = end_p - start_p;
    std::cout << "Parallel Average Cost: " << spend_p.count()/images.size() << "ms" << std::endl;
}

void testInferenceSpeed(){
    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls.onnx)"};
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
        auto start = std::chrono::high_resolution_clock::now();
        auto r = doInferenceByImgPth(file.c_str(), nullptr, msg);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> spend = end - start;
        total_costs += spend.count();
        costs.push_back(spend.count());
        std::cout << "Result: " << r.cls << " " << r.confidence << std::endl;
        std::cout << "cost: " << spend.count() << "ms" << std::endl;
        // destroyModel(ptr);
    }
    std::cout << "Average speed: " << total_costs/image_files.size() << "ms" << std::endl;    
}

int main(){
    std::cout << "OpenVINO Classification Demo" << std::endl;

    testInferenceSpeed();
    return 0;

    // printInfo();

    // run();
    // test_parallel();

    std::string onnx_pth = R"(D:\share_dir\iqc_crack\ultr_workdir\crack_cls0308\yolo11s_sgd_lr00052\weights\crack_cls_20250308.onnx)";
    // std::cout << "Input Your Model File(onnx): ";
    // std::cin >> onnx_pth;
    std::string image_dir{R"(E:\DataSets\iqc_cracks\new_250307)"};
    // std::cout<<"Input Target Image dir: ";
    // std::cin >> image_dir;
    std::string file_type{".jpg"};
	// std::cout << "Then Tell Me Which Type of File You Want to Inference [.jpg .bmp .png .pt]: ";
	// std::cin >> file_type;

    // test_iqc_proj(image_dir, onnx_pth, file_type);

    std::cout << "----------------------------------------------------------------------" << std::endl;
    char msg[1024];
    initModel(onnx_pth.c_str(), msg);
    std::cout << msg << std::endl;
    std::string file{R"(E:\DataSets\iqc_cracks\20250320163821925.jpg)"};
    CLS_RES res = doInferenceByImgPth(file.c_str(), nullptr, msg);
    std::cout << res.cls << " " << res.confidence << std::endl;

    cv::Mat img = cv::imread(file);
    CLS_RES r2 = doInferenceBy3chImg(img.ptr(), img.rows, img.cols, msg);
    std::cout << r2.cls << " " << r2.confidence << std::endl;

    // CLS_RES r3 = doInferenceByImgMat(img, model_ptr, msg);
    // std::cout << r3.cls << " " << r3.confidence << std::endl;
    std::cout << msg << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

   if(log_file.is_open())
    log_file.close();
    
}
