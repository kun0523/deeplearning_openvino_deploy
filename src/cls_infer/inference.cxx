#include "inference.h"

#define DEBUG

void* initModel(const char* onnx_pth, char* msg){    
#ifdef DEBUG
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << ov::get_openvino_version() << "\n";
    std::cout << ov::get_openvino_version() << std::endl;
#endif
    std::stringstream msg_ss;
    msg_ss << "Call <initModel> Func\n";
    ov::Core core;
    void* compiled_model_ptr = nullptr;

    try{   
        // 验证openvino环境是否正确     
        auto devices = core.get_available_devices();
        auto version = core.get_versions(core.get_available_devices()[0]);
        for(auto& item:version){
            msg_ss << item.first << " : " << item.second << "\n";
            #ifdef DEBUG
                fs << item.first << " : " << item.second << "\n";
                std::cout << item.first << " : " << item.second << std::endl;
            #endif
        }
    }catch(std::exception &ex){
        msg_ss << "OpenVINO Error!\n";
        msg_ss << ex.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
        #ifdef DEBUG
            fs << ex.what() << "\n";
            fs.close();
        #endif
        return compiled_model_ptr;
    }    
    
    try{
        // 创建模型
        auto model = core.read_model(onnx_pth);        
        compiled_model_ptr = new ov::CompiledModel(core.compile_model(model, "CPU", 
                                                                      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), 
                                                                      ov::hint::num_requests(4), 
                                                                      ov::auto_batch_timeout(100)));
        msg_ss << "Create Compiled Model Success. Got Model Pointer: " << compiled_model_ptr << "\n";
        #ifdef DEBUG
            fs << "Create Compiled Model Success. Got Model Pointer: " << compiled_model_ptr << "\n";            
            std::cout << "Create Compiled Model Success. Got Model Pointer: " << compiled_model_ptr << std::endl;            
        #endif
    }catch(std::exception &ex){
        msg_ss << "Create Model Failed\n";
        msg_ss << "Error Message: " << ex.what() << "\n";

        strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
        #ifdef DEBUG
            fs << "Create Model Failed\n";
            fs << "Error Message: " << ex.what() << "\n";
            fs.close();
        #endif
        return compiled_model_ptr;
    }
    
    strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
    #ifdef DEBUG
        fs.close();
    #endif
    return compiled_model_ptr;
}

void warmUp(void* compiled_model, char* msg){
    std::stringstream msg_ss;
    msg_ss << "Call <warmUp> Func ...\n";

    try{        
        msg_ss << "WarmUp Model Pointer: " << compiled_model << "\n";
        cv::Mat blob_img = cv::Mat::ones(cv::Size(1024, 1024), CV_32FC3);
        char msg[1024];
        doInferenceByImgMat(blob_img, compiled_model, msg);

        // size_t input_w = input_shape[2], input_h = input_shape[3];
        // cv::Mat blob = cv::dnn::blobFromImage(blob_img, 1.0, cv::Size(input_w, input_h));
        // ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr());
        // infer_request.set_input_tensor(input_tensor);
        // infer_request.infer();
        msg_ss << "Inference Done\n" << "WarmUp Complete";
    }catch(std::exception ex){
        msg_ss << "Catch Error in Warmup Func\n";
        msg_ss << "Error Message: " << ex.what() << endl;
    }

    strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
}

CLS_RES doInferenceByImgPth(const char* image_pth, void* compiled_model, const int* roi, char* msg){
#ifdef DEBUG
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << "Got Image path: " << image_pth << "\n";
#endif
    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        img.copyTo(img_part); 
#ifdef DEBUG
    fs << "ROI Image size: " << img_part.size << "\n";
    fs.close();
#endif
    return doInferenceByImgMat(img_part, compiled_model, msg);
}

CLS_RES doInferenceBy3chImg(uchar* image_arr, const std::int32_t height, const std::int32_t width, void* compiled_model, char* msg){
#ifdef DEBUG
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << "Got Image size: " << width << "x" << height << "\n";
#endif
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    // cv::Mat img(cv::Size(200, 200), CV_8UC3, image_arr);

#ifdef DEBUG
    fs << "Convert Image size: " << img.size << "\n";
    fs.close();
#endif
    return doInferenceByImgMat(img, compiled_model, msg);
}

CLS_RES doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, char* msg){
#ifdef DEBUG
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << "Call <doInferenceByImgMat> Func\n";
#endif
    std::stringstream msg_ss;
    msg_ss << "Call <doInferenceByImgMat> Func\n";
        
    ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(compiled_model);
    if(model_ptr==nullptr){
        msg_ss << "Error, Got nullptr, Model pointer convert failed\n";
        #ifdef DEBUG
            fs << "Error, Got nullptr, Model pointer convert failed\n";
            fs.close();
        #endif
        return CLS_RES(-1, -1);
    }else{
        msg_ss << "Convert Model Pointer Success.\n";
        msg_ss << "Got Inference Model Pointer: " << model_ptr << "\n";                
        #ifdef DEBUG
            fs << "Got Inference Model Pointer: " << model_ptr << "\n";            
        #endif
    }
    // 前提假设模型只有一个输入节点
    ov::Shape input_tensor_shape = model_ptr->input().get_shape();
    msg_ss << "Model Input Shape: " << input_tensor_shape << "\n";
    #ifdef DEBUG
        fs << "Model Input Shape: " << input_tensor_shape << "\n";          
    #endif

    auto start = std::chrono::high_resolution_clock::now();
    // TODO: 图片前处理还有问题！！！
    // cv::Mat resized_img;
    // cv::resize(img_mat, resized_img, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0, 0, cv::INTER_AREA);
    // float* input_data = (float*)resized_img.data;
    // const ov::Tensor input_tensor = ov::Tensor(model_ptr->input().get_element_type(), model_ptr->input().get_shape(), input_data);

    // cv::cvtColor(img_mat, resized_img, cv::COLOR_BGR2RGB);
    // cv::resize(img_mat, resized_img, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0, 0, cv::INTER_AREA);
    // cv::Mat float_img;
    // resized_img.convertTo(float_img, CV_32F);
    // cv::Scalar mean, stdev;
    // cv::meanStdDev(float_img, mean, stdev);
    // std::cout << mean << std::endl;
    // std::cout << stdev << std::endl;
    // cv::Mat normalized_img = (float_img - mean) / stdev / 1.0;
    // std::cout << cv::typeToString(normalized_img.type()) << std::endl;
    // std::cout << normalized_img.at<double>(0, 0) << " " << normalized_img.at<double>(0, 3) << " " << normalized_img.at<double>(0, 6) << std::endl;
    // cv::Mat blob_img = cv::dnn::blobFromImage(normalized_img, 1.0, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0.0, true, true, CV_32F);

    cv::Mat blob_img = cv::dnn::blobFromImage(img_mat, 1.0/255.0, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0.0, true, false, CV_32F);
    ov::Tensor input_tensor;
    opencvMat2Tensor(blob_img, *model_ptr, input_tensor);
    auto img_preprocess_done = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG
        fs << "Image Convert to Tensor success \n";          
    #endif

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
    #ifdef DEBUG
        fs << "Max confidence: " << ret.confidence << " Max Class Index: " << ret.cls << "\n";
    #endif

    string t = getTimeNow();
    msg_ss << "[" << t << "]" << "---- Inference Over ----\n";

    strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());

    #ifdef DEBUG
        fs << "[" << t << "]" << "---- Inference Over ----\n";
        fs.close();
    #endif    
    return ret;
}

char* resizeImageAsYOLO(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& boarded_img){
    // 分类任务 resize 逻辑与 目标检测任务不同
    
    int scale_w = 320, scale_h = 320;
    cv::Mat resized_img;
    cv::resize(org_img, resized_img, cv::Size(scale_w, scale_h), cv::INTER_AREA);

    int new_width = compiled_model.input().get_shape()[2], new_height = compiled_model.input().get_shape()[3];
    int center_crop_x_start = (scale_w-new_width)/2, center_crop_y_start = (scale_h-new_height)/2;
    int center_crop_x_end = scale_w-center_crop_x_start, center_crop_y_end = scale_h-center_crop_y_start;
    boarded_img = resized_img(cv::Rect(cv::Point(center_crop_x_start, center_crop_y_start), cv::Point(center_crop_x_end, center_crop_y_end)));
    return "Resize Image as YOLO.";
}

char* opencvMat2Tensor(cv::Mat& img_mat, ov::CompiledModel& compiled_model, ov::Tensor& out_tensor){
    auto input_port = compiled_model.input();
    out_tensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), img_mat.ptr());
    return "Convert Success";
}

int doInferenceBatchImgs(const char* image_dir, int height, int width, void* compiled_model, const int* roi, const int roi_len, char* msg){
    return 1;
    ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(compiled_model);
    
    ov::Shape input_tensor_shape = model_ptr->input().get_shape();
    vector<cv::Mat> image_list;

    auto start = std::chrono::high_resolution_clock::now();
    for(const auto& f:std::filesystem::directory_iterator(image_dir)){
        cv::Mat tmp_img = cv::imread(f.path().string());
        cv::Mat resized_img;
        double scale_ratio;
        int left_padding_cols, top_padding_rows;
        resizeImageAsYOLO(*model_ptr, tmp_img, resized_img);
        image_list.push_back(tmp_img);
    }

    cv::Mat blob_img = cv::dnn::blobFromImages(image_list, 1.0/255.0, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0.0, true, false, CV_32F);
    ov::Tensor inputensor;
    opencvMat2Tensor(blob_img, *model_ptr, inputensor);
    auto img_preprocess_done = std::chrono::high_resolution_clock::now();
    
    ov::InferRequest infer_request = model_ptr->create_infer_request();
    infer_request.set_input_tensor(inputensor);
    infer_request.infer();
    auto infer_done = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> img_preprocess_cost = img_preprocess_done - start;
    std::chrono::duration<double, std::milli> inference_cost = infer_done - img_preprocess_done;
    std::stringstream msg_ss;
    msg_ss << "Image Preprocess cost: " << img_preprocess_cost.count() << "ms";
    msg_ss << " Infer cost: " << inference_cost.count() << "ms";

    ov::Shape output_tensor_shape = model_ptr->output().get_shape();
    size_t batch_num=output_tensor_shape[0], preds_num=output_tensor_shape[1];
    auto output_tensor = infer_request.get_output_tensor();
    const float* output_buff = output_tensor.data<const float>();
    cv::Mat m = cv::Mat(cv::Size(batch_num, preds_num), CV_32F, const_cast<float*>(output_buff));
    cv::Point max_pos;
    double max_score;
    cv::minMaxLoc(m, 0, &max_score, 0, &max_pos);
    msg_ss << " max score: " << max_score << " max index: " << max_pos.y << endl;
    strcpy_s(msg, msg_ss.str().length()+2, msg_ss.str().c_str());
    return max_pos.y;
}
 
string getTimeNow() {
    std::stringstream ss;
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    
    // 将时间点转换为time_t以便进一步转换为本地时间
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    
    // 转换为本地时间并打印
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %X");
    return ss.str(); 
}