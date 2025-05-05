#include "inference.h"

void* gModelPtr = nullptr;

int DET_RES::get_area(){
    int width = std::abs(tl_x - br_x);
    int height = std::abs(tl_y - br_y);
    return width * height;
}

std::string DET_RES::get_info(){
    std::stringstream ss;
    ss << "BBOX Coord: [" << tl_x << ", " << tl_y << ", " << br_x << ", " << br_y << "]";
    ss << " | BBOX Area: " << get_area();
    ss << " | Class_id: " << cls;
    ss << " | Confidence: " << confidence;
    return ss.str();
}

int initModel(const char* onnx_pth, char* msg){
    std::stringstream msg_ss;
#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    std::cout << "[" << getTimeNow() << "] Use OpenVINO to Detect objects\n";

    msg_ss << "[" << getTimeNow() << "] Call <initModel> Func Model Path: " << onnx_pth << "\n";

    try{      
        ov::Core core;   
        gModelPtr = new ov::CompiledModel(core.compile_model(onnx_pth, "CPU", 
                                                                    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), 
                                                                    ov::hint::num_requests(4), ov::auto_batch_timeout(1000)));                       
        msg_ss << "[" << getTimeNow() << "] Create Compiled model Success. Got Model Pointer: " << gModelPtr << "\n";
    }catch(const std::exception& ex){
        msg_ss << "[" << getTimeNow() << "] Error Message: " << ex.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    #ifdef DEBUG_OPV
        fs << msg_ss.str();
        fs.close();
    #endif
        return 1;
    }

    warmUp(msg); 
    strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
#ifdef DEBUG_OPV
    fs << msg_ss.str();
    fs << "Model Init Success.\n";
    fs.close();
#endif
    return 0;
}

void warmUp(char* msg){
    cv::Mat blob_img = cv::Mat::ones(cv::Size(1024, 1024), CV_8UC3);
    int det_num;
    doInferenceByImgMat(blob_img, 0.5f, det_num, msg);
}

/*
模型推理结果解析，因为不同版本输出结果有差异
yolov8
- 需要NMS
- result  batch_num*[tlx, tly, brx, bry, cls_num]*bbox_num

yolov10
- 不需要NMS  后处理不需要转置
- result  batch_num*bbox_num*[tlx, tly, brx, bry, cls_num]

yolov11
- 需要NMS
- result:  batch_num*[center_x, center_y, weight, height, cls_num]*bbox_num 
*/
char* __postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double scale_ratio_, const int left_padding, const int top_padding, std::vector<DET_RES>& out_vec){

    // if(model_type!=8 && model_type!=10 && model_type != 11){
    //     throw std::runtime_error("Error Model Type " + std::to_string(model_type) + " Only support Model Type: v8 v10 v11.");
    // }

    // if(model_type!=10)
    //     det_result_mat = det_result_mat.t();    
    size_t pred_num = det_result_mat.cols;

    vector<cv::Rect2d> boxes;
    vector<float> scores;
    vector<int> indices;
    vector<int> class_idx;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.5f};

    for(int row=0; row<det_result_mat.rows; ++row){
        const float* ptr = det_result_mat.ptr<float>(row);
        vector<float> cls_conf = vector<float>(ptr+4, ptr+pred_num);
        cv::Point2i maxP;
        double maxV;
        cv::minMaxLoc(cls_conf, 0, &maxV, 0, &maxP);
        if (maxV<0.1) continue;  // 置信度非常低的直接跳过

        // switch (model_type)
        // {
        // case 8:
        // case 10:
        //     // 模型输出是 两个角点坐标
        //     tl_x = ptr[0], tl_y = ptr[1], br_x=ptr[2], br_y=ptr[3];
        //     boxes.emplace_back(tl_x, tl_y, br_x-tl_x, br_y-tl_y);
        //     break;
            
        // case 11:
            // 模型输出是 中心点 + 宽高
            cx = ptr[0], cy = ptr[1], w=ptr[2], h=ptr[3];  // 还不清楚是框架的问题还是有地方可以控制
            boxes.emplace_back(cx-w/2, cy-h/2, w, h);
            // break;
        // }
        scores.push_back(static_cast<float>(maxV));
        class_idx.push_back(maxP.x);
    }

    // switch(model_type){
    //     case 8:
    //     case 11:
            // 使用NMS过滤重复框
            cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);
            // break;

    //     case 10:
    //         // 不需要使用NMS过滤
    //         indices = vector<int>(boxes.size());
    //         std::iota(indices.begin(), indices.end(), 0);
    //         break;
    // }           

    for(auto it=indices.begin(); it!=indices.end(); ++it){
        float score = scores[*it];
        if (score < conf_threshold) continue;
        int cls = class_idx[*it];
        cv::Rect2d tmp = boxes[*it];
        tmp.x -= left_padding;
        tmp.x /= scale_ratio_;
        tmp.y -= top_padding;
        tmp.y /= scale_ratio_;
        tmp.width /= scale_ratio_;
        tmp.height /= scale_ratio_;
        out_vec.emplace_back(tmp, cls, score);     
    }

    return "Post Process Complete.";
}

DET_RES* doInferenceByImgPth(const char* img_pth, const int* roi, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceByImgPth> Func Image Name: " << img_pth << "\n";
    if(roi)
        msg_ss << "[" << getTimeNow() << "] ROI: [" << roi[0] << ", " << roi[1] << ", " << roi[2] << ", " << roi[3] << "]\n";

    try{
        cv::Mat org_img = cv::imread(img_pth, cv::IMREAD_COLOR);
        cv::Mat img_part;
        if(roi)
            img_part = org_img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
        else
            org_img.copyTo(img_part); 
        
        #ifdef DEBUG_OPV
            fs << msg_ss.str();
            fs.close();
        #endif
        return doInferenceByImgMat(img_part, score_threshold, det_num, msg);

    }catch(const std::exception& e){
        msg_ss << "[" << getTimeNow() << "] " << e.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());

        #ifdef DEBUG_OPV
            fs << msg_ss.str();
            fs.close();
        #endif
        return new DET_RES[1];
    }
}

DET_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceBy3chImg> Func Image Size: " << height << "x" << width << "\n";

    // 如果 width 和 height 与实际图像不符，出来的图像会扭曲，但不会报错
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
#ifdef DEBUG_OPV
    fs << msg_ss.str();
    fs.close();
#endif
    return doInferenceByImgMat(img, score_threshold, det_num, msg);
}

DET_RES* doInferenceByImgMat(const cv::Mat& img_mat, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceByImgMat> Func\n";

    try{
        if(gModelPtr == nullptr)
            throw std::runtime_error("Error Model Pointer Convert Failed!");

        ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(gModelPtr);

        ov::Shape input_tensor_shape = model_ptr->input().get_shape();
        msg_ss << "Model Input Shape: " << input_tensor_shape << "\n";
        cv::Mat blob_img;
        preProcess(*model_ptr, img_mat, blob_img);
        auto input_port = model_ptr->input();
        ov::Tensor inputensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), blob_img.ptr());
        auto img_preprocess_done = std::chrono::high_resolution_clock::now();
        
        ov::InferRequest infer_request = model_ptr->create_infer_request();    
        infer_request.set_input_tensor(inputensor);
        infer_request.infer();  // 同步推理    
        auto infer_done = std::chrono::high_resolution_clock::now();

        ov::Shape output_tensor_shape = model_ptr->output().get_shape();
        msg_ss << "Model Output Shape: " << output_tensor_shape << "\n";

        size_t res_height=output_tensor_shape[1], res_width=output_tensor_shape[2];
        auto output_tensor = infer_request.get_output_tensor();
        const float* output_buff = output_tensor.data<const float>();
        cv::Mat m = cv::Mat(cv::Size(res_width, res_height), CV_32F, const_cast<float*>(output_buff));

        vector<DET_RES> out_res_vec;
        double r = std::min((double)input_tensor_shape[2]/img_mat.rows, (double)input_tensor_shape[3]/img_mat.cols);

        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
        DET_RES* result = postProcess(score_threshold, m, r, det_num);

        #ifdef DEBUG_OPV
            fs << msg_ss.str();
            fs << "[" << getTimeNow() << "] ------------Inference Success.-------------------\n";
            fs.close();
        #endif
        return result;

    }catch(const std::exception& e){
        msg_ss << "[" << getTimeNow() << "] ERROR Message: " << e.what() << "\n";
        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    #ifdef DEBUG_OPV
        fs << msg_ss.str();
        fs << "[" << getTimeNow() << "] ------------Inference Failed.-------------------\n";
        fs.close();
    #endif
        return new DET_RES[1];
    }
}

char* resizeImageAsYOLO(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& boarded_img, double& scale_ratio, int& left_padding, int& top_padding){
    
    double new_height = compiled_model.input().get_shape()[2], new_width = compiled_model.input().get_shape()[3];
    double org_width = org_img.cols, org_height = org_img.rows;
    scale_ratio = new_width/org_width > new_height/org_height ? new_height/org_height : new_width/org_width;

    cv::Mat resized_img;
    cv::resize(org_img, resized_img, cv::Size(), scale_ratio, scale_ratio, cv::INTER_LINEAR);

    double dw = new_width - resized_img.cols;
    double dh = new_height - resized_img.rows;

    left_padding = static_cast<int>(dw/2);
    top_padding = static_cast<int>(dh/2);
    int right_padding = static_cast<int>(dw) - left_padding;
    int bottom_padding = static_cast<int>(dh) - top_padding;
    cv::copyMakeBorder(resized_img, boarded_img, top_padding, bottom_padding, left_padding, right_padding, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));

    return "Resize Image as YOLO.";
}

char* opencvMat2Tensor(cv::Mat& img_mat, ov::CompiledModel& compiled_model, ov::Tensor& out_tensor){
    auto input_port = compiled_model.input();
    out_tensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), img_mat.ptr());
    return "Convert Success";
}

int destroyModel(){
#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    if(gModelPtr!=nullptr){
        auto compiled_model_ptr = static_cast<ov::CompiledModel*>(gModelPtr);
        delete compiled_model_ptr;
    }
#ifdef DEBUG_OPV
    fs << "[" << getTimeNow() << "] Release Model Success.\n";
#endif
    return 0;
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

void preProcess(const ov::CompiledModel& compiled_model_ptr, const cv::Mat& org_img, cv::Mat& blob){
    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto input_tensor_shape = compiled_model_ptr.input().get_shape();
    // auto input_tensor_shape = infer_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int board_h = input_tensor_shape[2], board_w = input_tensor_shape[3];
    auto boarded_img = cv::Mat(cv::Size(board_w, board_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min(board_h/org_h, board_w/org_w);
    cv::Mat resize_img;
    cv::resize(org_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));

    blob = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(board_w, board_h), cv::Scalar(0,0,0), true, false);
}

DET_RES* postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double& scale_ratio_, int& det_num){
    det_result_mat = det_result_mat.t();
    size_t pred_num = det_result_mat.cols;

    vector<cv::Rect2d> boxes;
    vector<float> scores;
    vector<int> indices;
    vector<int> class_idx;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.5f};

    for(int row=0; row<det_result_mat.rows; ++row){
        const float* ptr = det_result_mat.ptr<float>(row);
        vector<float> cls_conf = vector<float>(ptr+4, ptr+pred_num);
        cv::Point2i maxP;
        double maxV;
        cv::minMaxLoc(cls_conf, 0, &maxV, 0, &maxP);
        if (maxV<0.1) continue;  // 置信度非常低的直接跳过

        // 模型输出是 中心点 + 宽高
        cx = ptr[0], cy = ptr[1], w=ptr[2], h=ptr[3];  // 还不清楚是框架的问题还是有地方可以控制
        boxes.emplace_back(cx-w/2, cy-h/2, w, h);

        scores.push_back(static_cast<float>(maxV));
        class_idx.push_back(maxP.x);
    }
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);

    det_num = indices.size();
    DET_RES* result = new DET_RES[det_num];
    int counter{};
    for(auto it=indices.begin(); it!=indices.end(); ++it){
        float score = scores[*it];
        if (score < conf_threshold) continue;
        int cls = class_idx[*it];
        cv::Rect2d tmp = boxes[*it];
        tmp.x /= scale_ratio_;
        tmp.y /= scale_ratio_;
        tmp.width /= scale_ratio_;
        tmp.height /= scale_ratio_;
        // out_vec.emplace_back(tmp, cls, score);     

        result[counter].tl_x = tmp.tl().x;
        result[counter].tl_y = tmp.tl().y;
        result[counter].br_x = tmp.br().x;
        result[counter].br_y = tmp.br().y;
        result[counter].cls = cls;
        result[counter].confidence = score;
        ++counter;
    }

    return result;
}

int freeResult(void* res_ptr, int num){
    if(!res_ptr){
        return 1;
    }
    
    delete[] static_cast<DET_RES*>(res_ptr);
    return 0;
}
