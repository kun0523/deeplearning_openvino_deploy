#include "inference.h"


MY_DLL void* initModel(const char* model_pth){
    auto logger = spdlog::daily_logger_format_mt("daily_logger", "logs/inference_log.txt", 0, 0);
    logger->set_level(spdlog::level::info);
    logger->info("Call initModel...");
    logger->info("Got Model Path: {}", model_pth);

    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 使用优化的模型图
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    size_t str_len = std::mbstowcs(nullptr, model_pth, 0)+1;
    std::wstring wstr(str_len, '\0');
    std::mbstowcs(&wstr[0], model_pth, str_len);
    
    Ort::Session* session_ptr = nullptr;
    try{
        session_ptr = new Ort::Session(env, wstr.c_str(), session_options);
    }catch(const std::exception& ex){
        logger->error("Got Error: {}", ex.what());
        logger->flush();
        return session_ptr;
    }
    logger->info("Load Model Success.");
    logger->flush();
    return session_ptr;
}


MY_DLL CLS_RES doInference(void* ort_session, Point base_point, Point* points, size_t p_num){

    auto logger = spdlog::get("daily_logger");
    if (logger == nullptr){
        logger = spdlog::daily_logger_format_mt("daily_logger", "logs/inference_log.txt", 0, 0);
        logger->set_level(spdlog::level::info);
    }    
    logger->info("Call Inference...");

    // 创建会话选项
    Ort::Session* session_ptr = nullptr;
    try{
        session_ptr = static_cast<Ort::Session*>(ort_session);
    }catch(const std::exception& ex){
        logger->error("Convert Model Pointer Failed! message: {}", ex.what());
        logger->flush();
        return CLS_RES{1, -1};
    }
    logger->info("Convert Model Pointer Success.");

    std::vector<const char*> input_names = {"float_input"};
    std::vector<const char*> output_names = {"output_label", "output_probability"};
    
    auto input_tensor_info = session_ptr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = input_tensor_info.GetShape();
    size_t features_num = input_shape[1];  // 模型需要的特征个数
    if(p_num != features_num){
        logger->error("Expect Features Number: {} But Got Features Number: {}", features_num, p_num);
        throw std::runtime_error("Features Num Error!!");
    }

    std::vector<float> features = calDistances(base_point, points, p_num);
    std::stringstream ss;
    ss << "Points: ";
    for(int i{}; i<p_num; ++i){
        ss << (*(points+i)).show();
    }
    logger->info(ss.str());
    ss.str("");
    ss.clear();
    ss << "Features: ";
    for(int i{}; i<p_num; ++i){
        ss << *(features.data()+i) << ", ";
    }
    logger->info(ss.str());
    logger->flush();

    // 创建 ONNX 运行时张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, features.data(), features.size(), input_shape.data(), input_shape.size());

    CLS_RES ret{};
    try{
        // 运行模型
        auto output_tensors = session_ptr->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 2);
        logger->info("Inference Over");
        
        // 获取输出
        auto output1 = output_tensors[0].GetTensorMutableData<int64_t>();
        int64_t cls_res = *output1;

        // 获取 Sequence 的长度        
        Ort::Value map_tensor = output_tensors[1].GetValue(0, Ort::AllocatorWithDefaultOptions());                
        auto key_tensor = map_tensor.GetValue(0, Ort::AllocatorWithDefaultOptions());
        auto value_tensor = map_tensor.GetValue(1, Ort::AllocatorWithDefaultOptions());
        int64_t* keys = key_tensor.GetTensorMutableData<int64_t>();
        float* values = value_tensor.GetTensorMutableData<float>();        

        if (cls_res == 0){
            ret.isNg = false;
            ret.confidence = *values;
        }else{
            ret.isNg = true;
            ret.confidence = *(values+1);
        }

    }catch(const std::exception &ex){        
        logger->error("Message:{}", ex.what());
    }catch(...){
        logger->error("Unexpect Error");
    }

    logger->info("Inference Result (isNG: {})  (confidence: {})", ret.isNg, ret.confidence);
    logger->flush();
    return ret;
}

//TODO: 注意做传入参数的校验
std::vector<float> calDistances(Point base_point, Point* points, size_t p_num){
    std::vector<float> distances{};
    for(int i=0; i<p_num; ++i){
        Point tmp_point = *(points+i);
        auto diff = base_point - tmp_point;
        float dist = std::sqrt(std::pow(diff.x, 2)+std::pow(diff.y, 2));
        distances.push_back(dist);
    }
    return distances;
}