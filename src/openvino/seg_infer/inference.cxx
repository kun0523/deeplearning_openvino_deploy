#include "inference.h"

void* gModelPtr = nullptr;

int SEG_RES::get_area(){
    int width = std::abs(tl_x - br_x);
    int height = std::abs(tl_y - br_y);
    return width * height;
}

std::string SEG_RES::get_info(){
    std::stringstream ss;
    ss << "BBOX Coord: [" << tl_x << ", " << tl_y << ", " << br_x << ", " << br_y << "]";
    ss << " | BBOX Area: " << get_area();
    ss << " | Class_id: " << cls;
    ss << " | Confidence: " << confidence;
    return ss.str();
}

SEG_RES::~SEG_RES(){
    if(mask_data){
        delete[] mask_data;
        mask_data = nullptr;
    }
}

int initModel(const char* onnx_pth, char* msg){
    std::stringstream msg_ss;
#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    std::cout << "[" << getTimeNow() << "] Use OpenVINO to Segment objects\n";

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

SEG_RES* doInferenceByImgPth(const char* img_pth, const int* roi, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceByImgPth> Func Image Path: " << img_pth << "\n";
    if(roi)
        msg_ss << "[" << getTimeNow() << "] ROI: [" << roi[0] << ", " << roi[1] << ", " << roi[2] << ", " << roi[3] << "]\n";
    
    try{
        // 对三通道的图进行推理
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
        return new SEG_RES[1];
    }
}

SEG_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceBy3chImg> Func Image Size: " << height << "x" << width << "\n";
    
    // 对三通道的图进行推理
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
#ifdef DEBUG_OPV
    fs << msg_ss.str();
    fs.close();
#endif
    return doInferenceByImgMat(img, score_threshold, det_num, msg);
}

SEG_RES* doInferenceByImgMat(const cv::Mat& img_mat, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

#ifdef DEBUG_OPV
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    msg_ss << "[" << getTimeNow() << "] Call <doInferenceByImgMat> Func\n";

    try{
        if(gModelPtr == nullptr)
            throw std::runtime_error("Error Model Pointer Convert Failed!");
    
        ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(gModelPtr);

        cv::Mat blob_img;
        preProcess(*model_ptr, img_mat, blob_img);

        auto input_port = model_ptr->input();
        auto input_shape = model_ptr->input().get_shape();
        ov::Tensor inputensor = ov::Tensor(input_port.get_element_type(), input_shape, blob_img.data);
        msg_ss << "Input Shape: " << input_shape.to_string() << "\n";
        
        ov::InferRequest infer_request = model_ptr->create_infer_request();    
        infer_request.set_input_tensor(inputensor);
        infer_request.infer();  // 同步推理    

        // 已知模型有两个输出节点
        auto outputs = model_ptr->outputs();
        auto pred_shape = outputs[0].get_shape();
        auto proto_shape = outputs[1].get_shape();
        msg_ss << "Output Pred Shape: " << pred_shape.to_string() << "\n";
        msg_ss << "Output Proto Shape: " << proto_shape.to_string() << "\n";

        auto pred_tensor = infer_request.get_output_tensor(0);
        auto proto_tensor = infer_request.get_output_tensor(1);
        cv::Mat pred = cv::Mat(pred_shape[1], pred_shape[2], CV_32F, pred_tensor.data<float>());
        pred = pred.t();
        cv::Mat proto = cv::Mat(proto_shape[1], proto_shape[2]*proto_shape[3], CV_32F, proto_tensor.data<float>());

        strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
        SEG_RES* result = postProcess(score_threshold, pred, proto, cv::Size(img_mat.cols, img_mat.rows), cv::Size(input_shape[2], input_shape[3]), det_num);
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
        return new SEG_RES[1];
    }
}

void preProcess(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& blob){
    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto input_tensor_shape = compiled_model.input().get_shape();
    int board_h = input_tensor_shape[2], board_w = input_tensor_shape[3];
    auto boarded_img = cv::Mat(cv::Size(board_w, board_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min(board_h/org_h, board_w/org_w);
    cv::Mat resize_img;
    cv::resize(org_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));

    blob = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(board_w, board_h), cv::Scalar(0,0,0), true, false);
}

SEG_RES* postProcess(const float conf_threshold, const cv::Mat& pred_mat, const cv::Mat& proto_mat, const cv::Size& org_size, const cv::Size& infer_size, int& det_num){
    size_t pred_num = pred_mat.cols;

    vector<cv::Rect2d> boxes;
    vector<float> scores;
    vector<int> indices;
    vector<int> class_idx;
    vector<cv::Mat> masks;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.7f};

    for(int row=0; row<pred_mat.rows; ++row){
        const float* ptr = pred_mat.ptr<float>(row);
        vector<float> cls_conf = vector<float>(ptr+4, ptr+pred_num-32);
        cv::Point2i maxP;
        double maxV;
        cv::minMaxLoc(cls_conf, 0, &maxV, 0, &maxP);
        if (maxV<0.1) continue;  // 置信度非常低的直接跳过

        // 模型输出是 中心点 + 宽高
        cx = ptr[0], cy = ptr[1], w=ptr[2], h=ptr[3];  // Anchor Free 下 模型直接预测中心点和宽高
        boxes.emplace_back(cx-w/2, cy-h/2, w, h);

        cv::Mat m = pred_mat.row(row).colRange(pred_num-32, pred_num);
        masks.push_back(m);
        scores.push_back(static_cast<float>(maxV));
        class_idx.push_back(maxP.x);
    }
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);

    double r = std::min((double)infer_size.height/org_size.height, (double)infer_size.width/org_size.width);

    det_num = indices.size();
    SEG_RES* result = new SEG_RES[det_num];
    int counter{};
    for(auto it=indices.begin(); it!=indices.end(); ++it){
        float score = scores[*it];
        if (score < conf_threshold) continue;
        int cls = class_idx[*it];
        cv::Rect2d tmp = boxes[*it];
        tmp.x /= r;
        tmp.y /= r;
        tmp.width /= r;
        tmp.height /= r;

        cv::Mat m = masks[*it];
        cv::Mat mask = m * proto_mat;
        mask = mask.reshape(1, 160);

        int n_h = r*org_size.height, n_w = r*org_size.width;
        cv::resize(mask, mask, infer_size);
        mask = mask(cv::Rect(cv::Point(0,0), cv::Point(n_w, n_h)));
        cv::resize(mask, mask, org_size);

        cv::threshold(mask, mask, 0, 114, cv::THRESH_BINARY);  // 前景>0, 背景<0;
        cv::Mat instance_mask;
        mask(tmp).copyTo(instance_mask);

        result[counter].tl_x = tmp.tl().x;
        result[counter].tl_y = tmp.tl().y;
        result[counter].br_x = tmp.br().x;
        result[counter].br_y = tmp.br().y;
        result[counter].confidence = score;
        result[counter].cls = cls;
        result[counter].mask_h = instance_mask.rows;
        result[counter].mask_w = instance_mask.cols;
        result[counter].mask_type = instance_mask.type();
        result[counter].mask_data = new uchar[instance_mask.total()*instance_mask.elemSize()];
        memcpy(result[counter].mask_data, instance_mask.data, instance_mask.total()*instance_mask.elemSize());
        ++counter;
    }

    return result;
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
    fs.close();
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
    