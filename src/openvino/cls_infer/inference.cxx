#include "inference.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;

void* gModelPtr = nullptr;

void printInfo(){
#ifdef DEBUG_STAT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << ov::get_openvino_version() << "\n";
#endif
    std::cout << "OpenVINO CLassification Lib" << std::endl;
    std::cout << ov::get_openvino_version() << endl;

    ov::Core core;
    std::stringstream msg_ss;
    try{   
        // 验证openvino环境是否正确     
        auto devices = core.get_available_devices();
        auto version = core.get_versions(core.get_available_devices()[0]);
        for(auto& item:version){
            std::cout << item.first << " : " << item.second << "\n";
            #ifdef DEBUG_STAT
                fs << item.first << " : " << item.second << "\n";
            #endif
        }
    }catch(std::exception &ex){
        std::cout << "OpenVINO Error!\n";
        std::cout << ex.what() << "\n";
        #ifdef DEBUG_STAT
            fs << ex.what() << "\n";
            fs.close();
        #endif
    }   
}

void run(){

    ov::Core core;
    std::string onnx_pth{R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls.onnx)"};
    auto model = core.read_model(onnx_pth);
    auto compiled_model = ov::CompiledModel(core.compile_model(model, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::hint::num_requests(4), ov::auto_batch_timeout(100)));

    std::string image_pth = R"(E:\DataSets\imageNet\n01514859_hen.JPEG)";
    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    ov::Shape input_tensor_shape = compiled_model.input().get_shape();
    cv::Mat blob_img = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0.0, true, false, CV_32F);
    auto input_port = compiled_model.input();
    ov::Tensor input_tensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), blob_img.ptr());
    
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    ov::Shape output_tensor_shape = compiled_model.output().get_shape();  // 1 x 1000
    size_t batch_num=output_tensor_shape[0], preds_num=output_tensor_shape[1];
    auto output_tensor = infer_request.get_output_tensor();
    const float* output_buff = output_tensor.data<const float>();
    cv::Mat m = cv::Mat(cv::Size(batch_num, preds_num), CV_32F, const_cast<float*>(output_buff));
    cv::Point max_pos;
    double max_score;
    cv::minMaxLoc(m, 0, &max_score, 0, &max_pos);
    std::cout << max_score << "  " << max_pos.y << std::endl;
    std::cout << "inference done" << std::endl;
}

void initModel(const char* onnx_pth, char* msg){    
    // OpenVINO 第一次推理耗时比较久，所以要做 WarmUp
#ifdef DEBUG_STAT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << ov::get_openvino_version() << "\n";
    std::cout << ov::get_openvino_version() << endl;
#endif
    std::stringstream msg_ss;
    msg_ss << "Call <initModel> Func\n";
    ov::Core core;

    try{
        // 创建模型
        auto model = core.read_model(onnx_pth);        
        gModelPtr = new ov::CompiledModel(core.compile_model(model, "CPU", 
                                                                      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), 
                                                                      ov::hint::num_requests(4), 
                                                                      ov::auto_batch_timeout(100)));
        msg_ss << "Create Compiled Model Success. Got Model Pointer: " << gModelPtr << "\n";
        #ifdef DEBUG_STAT
            fs << "Create Compiled Model Success. Got Model Pointer: " << gModelPtr << "\n";            
            std::cout << "Create Compiled Model Success. Got Model Pointer: " << gModelPtr << std::endl;            
        #endif
    }catch(std::exception &ex){
        msg_ss << "Create Model Failed\n";
        msg_ss << "Error Message: " << ex.what() << "\n";

        strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
        #ifdef DEBUG_STAT
            fs << "Create Model Failed\n";
            fs << "Error Message: " << ex.what() << "\n";
            fs.close();
        #endif
        return;
    }

    warmUp();
    strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
    #ifdef DEBUG_STAT
        fs.close();
    #endif
    return;
}

CLS_RES doInferenceByImgPth(const char* image_pth, const int* roi, char* msg){
#ifdef DEBUG_STAT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << getTimeNow() << " Got Image path: " << image_pth << "\n";
#endif
    try{
        cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
        if(img.empty()){
            throw std::runtime_error("Read Image Failed."+std::string(image_pth));
        }
        cv::Mat img_part;
        if(roi)
            img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
        else
            img.copyTo(img_part); 

        #ifdef DEBUG_STAT
            fs << getTimeNow() << " ROI Image size: " << img_part.size << "\n";
            fs.close();
        #endif
        return doInferenceByImgMat(img_part, msg);

    }catch(const std::exception& e){
        std::cerr << e.what() << std::endl;
        fs << getTimeNow() << " Error Message: " << e.what() << "\n";
        fs.close();
        return CLS_RES(-1, -1);
    }
}

CLS_RES doInferenceBy3chImg(uchar* image_arr, const std::int32_t height, const std::int32_t width, char* msg){
#ifdef DEBUG_STAT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    // 如果 width 和 height 与实际图像不符，出来的图像会扭曲，但不会报错
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);

#ifdef DEBUG_STAT
    fs << "Convert Image size: " << img.size << "\n";
    fs.close();
#endif
    return doInferenceByImgMat(img, msg);
}

CLS_RES doInferenceByImgMat(const cv::Mat& img_mat, char* msg){
#ifdef DEBUG_STAT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << "[" << getTimeNow() << "]" << "Call <doInferenceByImgMat> Func\n";
#endif
    std::stringstream msg_ss;
    
    ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(gModelPtr);
    if(model_ptr==nullptr){
        msg_ss << "Error, Got nullptr, Model pointer convert failed\n";
        #ifdef DEBUG_STAT
            fs << "Error, Got nullptr, Model pointer convert failed\n";
            fs.close();
        #endif
        return CLS_RES(-1, -1);
    }else{
        msg_ss << "Convert Model Pointer Success.\n";
        msg_ss << "Got Inference Model Pointer: " << model_ptr << "\n";    
    }
    // 前提假设模型只有一个输入节点
    ov::Shape input_tensor_shape = model_ptr->input().get_shape();
    msg_ss << "Model Input Shape: " << input_tensor_shape << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat blob_img = cv::dnn::blobFromImage(img_mat, 1.0/255.0, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0.0, true, false, CV_32F);
    auto input_port = model_ptr->input();
    ov::Tensor input_tensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), blob_img.ptr());
    auto img_preprocess_done = std::chrono::high_resolution_clock::now();

    ov::InferRequest infer_request = model_ptr->create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto infer_done = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> img_preprocess_cost = img_preprocess_done - start;
    std::chrono::duration<double, std::milli> inference_cost = infer_done - img_preprocess_done;
    msg_ss << "Image Preprocess cost: " << img_preprocess_cost.count() << "ms Infer cost: " << inference_cost.count() << "ms\n";

    ov::Shape output_tensor_shape = model_ptr->output().get_shape();
    msg_ss << "Model Output Shape: " << output_tensor_shape << "\n";

    size_t batch_num=output_tensor_shape[0], preds_num=output_tensor_shape[1];
    auto output_tensor = infer_request.get_output_tensor();
    const float* output_buff = output_tensor.data<const float>();
    cv::Mat m = cv::Mat(cv::Size(batch_num, preds_num), CV_32F, const_cast<float*>(output_buff));
    cv::Point max_pos;
    double max_score;
    cv::minMaxLoc(m, 0, &max_score, 0, &max_pos);
    CLS_RES ret = CLS_RES(max_pos.y, max_score);
    msg_ss << "Max confidence: " << ret.confidence << " Max Class Index: " << ret.cls << "\n";

    string t = getTimeNow();
    msg_ss << "[" << t << "]" << "---- Inference Over ----\n";
    strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());

    #ifdef DEBUG_STAT
        fs << msg_ss.str();
        fs.close();
    #endif   
    return ret;
}

void warmUp(){
    ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(gModelPtr);
    // 前提假设模型只有一个输入节点
    ov::Shape input_tensor_shape = model_ptr->input().get_shape();
    cv::Mat img_mat = cv::Mat::ones(cv::Size(224,224), CV_8UC3);
    cv::Mat blob_img = cv::dnn::blobFromImage(img_mat, 1.0/255.0, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0.0, true, false, CV_32F);
    auto input_port = model_ptr->input();
    ov::Tensor input_tensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), blob_img.ptr());
    ov::InferRequest infer_request = model_ptr->create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
}

void destroyModel(){
#ifdef DEBUG_STAT
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    if(gModelPtr!=nullptr){
        ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(gModelPtr);
        delete model_ptr;      
    }
#ifdef DEBUG_STAT
    fs << "[" << getTimeNow() << "]" << "Release Model Success.\n";
#endif
}


std::string getTimeNow() {
    std::stringstream ss;
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    
    // 将时间点转换为time_t以便进一步转换为本地时间
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    
    // 转换为本地时间并打印
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %X");
    return ss.str(); 
}
