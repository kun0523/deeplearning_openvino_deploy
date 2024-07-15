#include "inference.h"

void* initModel(const char* onnx_pth, char* msg, size_t msg_len){
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
        }
    }catch(std::exception ex){
        msg_ss << "OpenVINO Error!\n";
        msg_ss << ex.what() << "\n";
        strcpy_s(msg, msg_len, msg_ss.str().c_str());
        return compiled_model_ptr;
    }    
    
    try{
        // 创建模型
        compiled_model_ptr = new ov::CompiledModel(core.compile_model(onnx_pth, "CPU"));
        msg_ss << "Create Compiled Model Success. Got Model Pointer: " << compiled_model_ptr << "\n";
    }catch(std::exception ex){
        msg_ss << "Create Model Failed\n";
        msg_ss << "Error Message: " << ex.what() << "\n";

        strcpy_s(msg, msg_len, msg_ss.str().c_str());
        return compiled_model_ptr;
    }
    
    strcpy_s(msg, msg_len, msg_ss.str().c_str());
    return compiled_model_ptr;
}

void warmUp(void* compiled_model, char* msg, size_t msg_len){
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

    strcpy_s(msg, msg_len, msg_ss.str().c_str());
}

CLS_RES doInferenceByImgPth(const char* image_pth, void* compiled_model, const int* roi, char* msg, size_t msg_len){
    
    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        img.copyTo(img_part); 

    return doInferenceByImgMat(img_part, compiled_model, msg, msg_len);
}

CLS_RES doInferenceBy3chImg(uchar* image_arr, const int height, const int width, void* compiled_model, char* msg, size_t msg_len){
    
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return doInferenceByImgMat(img, compiled_model, msg, msg_len);
}

CLS_RES doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, char* msg, size_t msg_len){

    std::stringstream msg_ss;
    msg_ss << "Call <doInferenceByImgMat> Func\n";
        
    ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(compiled_model);
    if(model_ptr==nullptr){
        msg_ss << "Error, Got nullptr, Model pointer convert failed\n";
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
    ov::Tensor inputensor;
    opencvMat2Tensor(blob_img, *model_ptr, inputensor);
    auto img_preprocess_done = std::chrono::high_resolution_clock::now();
    
    ov::InferRequest infer_request = model_ptr->create_infer_request();
    infer_request.set_input_tensor(inputensor);
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
    strcpy_s(msg, msg_len, msg_ss.str().c_str());

    // std::fstream fs{"./log.txt", std::ios_base::out};
    // fs << msg_ss.str() << endl;
    // fs.close();
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

int doInferenceBatchImgs(const char* image_dir, int height, int width, void* compiled_model, const int* roi, const int roi_len, char* msg, size_t msg_len){
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
    // std::cout <<"scores: " << m << std::endl;
    cv::Point max_pos;
    double max_score;
    cv::minMaxLoc(m, 0, &max_score, 0, &max_pos);
    msg_ss << " max score: " << max_score << " max index: " << max_pos.y << endl;
    strcpy_s(msg, msg_len, msg_ss.str().c_str());
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