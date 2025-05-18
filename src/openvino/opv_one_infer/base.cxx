#include "base.h"

using std::vector;

std::string CLS_RES::get_info(){
    std::stringstream ss;
    ss << "Class_id: " << cls << " | Confidence: " << confidence;
    return ss.str();
}

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

Base::Base(const char* model_pth_, char* msg):model_pth(model_pth_), my_msg(msg){
    // 加载模型文件
    my_model_ptr = new ov::CompiledModel(my_core.compile_model(model_pth, "CPU", 
                                                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), 
                                                                ov::hint::num_requests(4), ov::auto_batch_timeout(1000)));                       

}

Base::~Base(){
    if(my_model_ptr){
        delete my_model_ptr;
        my_model_ptr = nullptr;
    }
    std::cout << "------------- Model Destroyed ------------------------\n";
}

Classify::Classify(const char* model_pth_, char* msg):Base(model_pth_, msg){
    // warmup

}

void* Classify::inferByMat(cv::Mat& img_mat, const float conf_threshold, int& num, char* msg) {

    checkImageChannel(img_mat);
    img_mat.copyTo(infer_img);

    if(result_ptr){
        delete[] static_cast<CLS_RES*>(result_ptr);
        result_ptr = nullptr;
        result_len = 0;
    }

    if(my_model_ptr == nullptr){
        throw std::runtime_error("No Valid Model");
    }

    // 前提假设模型只有一个输入节点
    ov::Shape input_tensor_shape = my_model_ptr->input().get_shape();
    std::cout << "Model Input Shape: " << input_tensor_shape << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat blob_img = cv::dnn::blobFromImage(img_mat, 1.0/255.0, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0.0, true, false, CV_32F);
    auto input_port = my_model_ptr->input();
    ov::Tensor input_tensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), blob_img.ptr());
    auto img_preprocess_done = std::chrono::high_resolution_clock::now();

    ov::InferRequest infer_request = my_model_ptr->create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto infer_done = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> img_preprocess_cost = img_preprocess_done - start;
    std::chrono::duration<double, std::milli> inference_cost = infer_done - img_preprocess_done;
    std::cout << "Image Preprocess cost: " << img_preprocess_cost.count() << "ms Infer cost: " << inference_cost.count() << "ms\n";

    ov::Shape output_tensor_shape = my_model_ptr->output().get_shape();
    std::cout << "Model Output Shape: " << output_tensor_shape << "\n";

    size_t batch_num=output_tensor_shape[0], preds_num=output_tensor_shape[1];
    auto output_tensor = infer_request.get_output_tensor();
    const float* output_buff = output_tensor.data<const float>();
    cv::Mat m = cv::Mat(cv::Size(batch_num, preds_num), CV_32F, const_cast<float*>(output_buff));
    cv::Point max_pos;
    double max_score;
    cv::minMaxLoc(m, 0, &max_score, 0, &max_pos);
    CLS_RES result{-1, -1};
    result.cls = max_pos.y;
    result.confidence = max_score;

    std::cout << "Max confidence: " << result.confidence << " Max Class Index: " << result.cls << "\n";

    result_ptr = new CLS_RES[1]{result};
    result_len = 1;

    return result_ptr;
}

void* Classify::inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& num, char* msg){
    std::cout << ">>>>>>>>>>>>>>>>>>>>>> Classify infer by image path <<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        img.copyTo(img_part); 

    return inferByMat(img_part, conf_threshold, num, msg);
}

void* Classify::inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& num, char* msg){
    std::cout << "Classify infer by uchar array" << std::endl;
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return inferByMat(img, conf_threshold, num, msg);
}

Classify::~Classify(){
}

void Classify::drawResult(const short stop_period, const bool is_save)const {
    if(!result_ptr)
        return;

    int org_h = infer_img.rows;
    int org_w = infer_img.cols;

    cv::Mat new_img(org_h+30, org_w, CV_8UC3, cv::Scalar(0, 100, 200));
    infer_img.copyTo(new_img(cv::Rect(0, 0, org_w, org_h)));

    CLS_RES result = static_cast<CLS_RES*>(result_ptr)[0];
    std::string label{"Class_id: "};
    cv::putText(new_img, label+std::to_string(result.cls), cv::Point(0, new_img.rows-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.0);
    cv::imshow("Show Result", new_img);
    cv::waitKey(stop_period);

    if(is_save){
        std::string save_file_name = getTimeNow() + ".jpg";
        std::replace(save_file_name.begin(), save_file_name.end(), ':', '-');
        std::replace(save_file_name.begin(), save_file_name.end(), ' ', '-');
        std::cout << "Save to: " << save_file_name << std::endl;
        if(cv::imwrite(save_file_name, new_img)){
            std::cout << "Save Image Success.\n";
        }else{
            std::cout << "Save Image Failed.\n";
        }
    }

}


Detection::Detection(const char* model_pth_, char* msg):Base(model_pth_, msg){
    // warmup

}

void* Detection::inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& det_num, char* msg){
    std::cout << ">>>>>>>>>>>>>>>>>>>>>> Detection infer by image path <<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        img.copyTo(img_part); 

    return inferByMat(img_part, conf_threshold, det_num, msg);
}

void* Detection::inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& det_num, char* msg){
    std::cout << "Detection infer by uchar array" << std::endl;
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return inferByMat(img, conf_threshold, det_num, msg);
}

void* Detection::inferByMat(cv::Mat& img_mat, const float conf_threshold, int& det_num, char* msg) {
    
    checkImageChannel(img_mat);
    img_mat.copyTo(infer_img);

    if(result_ptr){
        delete[] static_cast<DET_RES*>(result_ptr);
        result_ptr = nullptr;
        result_len = 0;
    }

    if(my_model_ptr == nullptr){
        throw std::runtime_error("No Valid Model");
    }

    ov::Shape input_tensor_shape = my_model_ptr->input().get_shape();
    std::cout << "Model Input Shape: " << input_tensor_shape << "\n";
    
    double org_h = infer_img.rows, org_w = infer_img.cols;
    // 前提假设，模型只有一个输入节点
    // auto input_tensor_shape = infer_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int board_h = input_tensor_shape[2], board_w = input_tensor_shape[3];
    auto boarded_img = cv::Mat(cv::Size(board_w, board_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min(board_h/org_h, board_w/org_w);
    cv::Mat resize_img;
    cv::resize(infer_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));
    cv::Mat blob_img = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(board_w, board_h), cv::Scalar(0,0,0), true, false);
    
    auto input_port = my_model_ptr->input();
    ov::Tensor inputensor = ov::Tensor(input_port.get_element_type(), input_port.get_shape(), blob_img.ptr());
    auto img_preprocess_done = std::chrono::high_resolution_clock::now();
    
    ov::InferRequest infer_request = my_model_ptr->create_infer_request();    
    infer_request.set_input_tensor(inputensor);
    infer_request.infer();  // 同步推理    
    auto infer_done = std::chrono::high_resolution_clock::now();

    ov::Shape output_tensor_shape = my_model_ptr->output().get_shape();
    std::cout << "Model Output Shape: " << output_tensor_shape << "\n";

    size_t res_height=output_tensor_shape[1], res_width=output_tensor_shape[2];
    auto output_tensor = infer_request.get_output_tensor();
    const float* output_buff = output_tensor.data<const float>();
    cv::Mat m = cv::Mat(cv::Size(res_width, res_height), CV_32F, const_cast<float*>(output_buff));

    vector<DET_RES> out_res_vec;
    double r = std::min((double)input_tensor_shape[2]/img_mat.rows, (double)input_tensor_shape[3]/img_mat.cols);

    DET_RES* result = postProcess(conf_threshold, m, r, det_num);
    
    result_ptr = result;
    result_len = det_num;
    
    return result;
}

void Detection::preProcess(const cv::Mat& org_img, cv::Mat& blob){
    int board_h = 640;
    int board_w = 640;

    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto boarded_img = cv::Mat(cv::Size(board_w, board_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min((double)board_h/org_h, (double)board_w/org_w);
    cv::Mat resize_img;
    cv::resize(org_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));
    blob = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(board_w, board_h), cv::Scalar(0,0,0), true, false);
}

DET_RES* Detection::postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double& scale_ratio_, int& det_num){
    det_result_mat = det_result_mat.t();
    size_t pred_num = det_result_mat.cols;

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> class_idx;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.5f};

    for(int row=0; row<det_result_mat.rows; ++row){
        const float* ptr = det_result_mat.ptr<float>(row);
        std::vector<float> cls_conf = std::vector<float>(ptr+4, ptr+pred_num);
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

        result[counter].br_x = tmp.br().x;
        result[counter].br_y = tmp.br().y;
        result[counter].tl_x = tmp.tl().x;
        result[counter].tl_y = tmp.tl().y;
        result[counter].confidence = score;
        result[counter].cls = cls;
        ++counter;
    }

    return result;
}

Detection::~Detection(){

}

void Detection::drawResult(const short stop_period, const bool is_save) const {
    if(!result_ptr)
        return;

    cv::Mat new_img;
    infer_img.copyTo(new_img);
    auto box_color = cv::Scalar(0, 100, 200);
    auto font_color = cv::Scalar(200, 200, 200);
    for(int i{}; i<result_len; i++){
        DET_RES result = static_cast<DET_RES*>(result_ptr)[i];
        cv::Rect r(cv::Point(result.tl_x, result.tl_y), cv::Point(result.br_x, result.br_y));
        cv::rectangle(new_img, r, box_color, 1);
        std::stringstream label;
        label << "Class_id: " << result.cls << " conf: " << std::setprecision(2) << result.confidence;
        new_img(cv::Rect(cv::Point(result.tl_x, result.br_y), cv::Point(result.br_x, result.br_y+30))) = box_color;
        cv::putText(new_img, label.str(), cv::Point(result.tl_x, result.br_y+20), cv::FONT_HERSHEY_SIMPLEX, 0.6, font_color);
    }
    cv::imshow("Show Result", new_img);
    cv::waitKey(stop_period);

    if(is_save){
        std::string save_file_name = getTimeNow() + ".jpg";
        std::replace(save_file_name.begin(), save_file_name.end(), ':', '-');
        std::replace(save_file_name.begin(), save_file_name.end(), ' ', '-');
        std::cout << "Save to: " << save_file_name << std::endl;
        if(cv::imwrite(save_file_name, new_img)){
            std::cout << "Save Image Success.\n";
        }else{
            std::cout << "Save Image Failed.\n";
        }    
    }
}


Segmentation::Segmentation(const char* model_pth_, char* msg):Base(model_pth_, msg){
    // warmup

}

void* Segmentation::inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& det_num, char* msg){
    std::cout << ">>>>>>>>>>>>>>>>>>>>>> Segment infer by image path <<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
    cv::Mat img = cv::imread(image_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        img.copyTo(img_part); 

    
    return inferByMat(img_part, conf_threshold, det_num, msg);
}

void* Segmentation::inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& det_num, char* msg){
    std::cout << "Segmentation infer by uchar array" << std::endl;
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return inferByMat(img, conf_threshold, det_num, msg);
}

void* Segmentation::inferByMat(cv::Mat& img_mat, const float conf_threshold, int& det_num, char* msg) {
    
    checkImageChannel(img_mat);
    img_mat.copyTo(infer_img);
    
    if(result_ptr){
        delete[] static_cast<SEG_RES*>(result_ptr);
        result_ptr = nullptr;
        result_len = 0;
    }

    if(my_model_ptr==nullptr){
        throw std::runtime_error("No Valid Model");
    }
    
    double org_h = infer_img.rows, org_w = infer_img.cols;
    // 前提假设，模型只有一个输入节点
    auto input_tensor_shape = my_model_ptr->input().get_shape();
    int board_h = input_tensor_shape[2], board_w = input_tensor_shape[3];
    auto boarded_img = cv::Mat(cv::Size(board_w, board_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min(board_h/org_h, board_w/org_w);
    cv::Mat resize_img;
    cv::resize(infer_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));
    cv::Mat blob_img = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(board_w, board_h), cv::Scalar(0,0,0), true, false);

    auto input_port = my_model_ptr->input();
    auto input_shape = my_model_ptr->input().get_shape();
    ov::Tensor inputensor = ov::Tensor(input_port.get_element_type(), input_shape, blob_img.data);
    std::cout << "Input Shape: " << input_shape.to_string() << "\n";
    
    ov::InferRequest infer_request = my_model_ptr->create_infer_request();    
    infer_request.set_input_tensor(inputensor);
    infer_request.infer();  // 同步推理    

    // 已知模型有两个输出节点
    auto outputs = my_model_ptr->outputs();
    auto pred_shape = outputs[0].get_shape();
    auto proto_shape = outputs[1].get_shape();
    std::cout << "Output Pred Shape: " << pred_shape.to_string() << "\n";
    std::cout << "Output Proto Shape: " << proto_shape.to_string() << "\n";

    auto pred_tensor = infer_request.get_output_tensor(0);
    auto proto_tensor = infer_request.get_output_tensor(1);
    cv::Mat pred = cv::Mat(pred_shape[1], pred_shape[2], CV_32F, pred_tensor.data<float>());
    pred = pred.t();
    cv::Mat proto = cv::Mat(proto_shape[1], proto_shape[2]*proto_shape[3], CV_32F, proto_tensor.data<float>());

    SEG_RES* result = postProcess(conf_threshold, pred, proto, cv::Size(img_mat.cols, img_mat.rows), cv::Size(input_shape[2], input_shape[3]), det_num);

    result_ptr = result;
    result_len = det_num;

    return result;
}

void Segmentation::preProcess(const cv::Mat& org_img, cv::Mat& blob){
    int input_h = 640;
    int input_w = 640;

    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto boarded_img = cv::Mat(cv::Size(input_w, input_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min((double)input_h/org_h, (double)input_w/org_w);
    cv::Mat resize_img;
    cv::resize(org_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));

    blob = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(input_w, input_h), cv::Scalar(0,0,0), true, false);
}

SEG_RES* Segmentation::postProcess(const float conf_threshold, const cv::Mat& pred_mat, const cv::Mat& proto_mat, const cv::Size& org_size, const cv::Size& infer_size, int& det_num){
    size_t pred_num = pred_mat.cols;

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> class_idx;
    std::vector<cv::Mat> masks;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.5f};

    for(int row=0; row<pred_mat.rows; ++row){
        const float* ptr = pred_mat.ptr<float>(row);
        std::vector<float> cls_conf = std::vector<float>(ptr+4, ptr+pred_num-32);
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

    // TODO: 有BUG
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

        cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);  // 前景>0, 背景<0;
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

Segmentation::~Segmentation(){

}

void Segmentation::drawResult(const short stop_period, const bool is_save)const{
    if(!result_ptr)
        return;

    cv::Mat new_img;
    infer_img.copyTo(new_img);
    auto box_color = cv::Scalar(0, 100, 200);
    auto font_color = cv::Scalar(200,200,200);
    SEG_RES* seg_res_ptr = static_cast<SEG_RES*>(result_ptr);
    for(int i{}; i<result_len; i++){
        cv::Rect r(cv::Point(seg_res_ptr[i].tl_x, seg_res_ptr[i].tl_y), cv::Point(seg_res_ptr[i].br_x, seg_res_ptr[i].br_y));
        cv::Mat patch(new_img(r));
        cv::Mat binary_mask(seg_res_ptr[i].mask_h, seg_res_ptr[i].mask_w, seg_res_ptr[i].mask_type, seg_res_ptr[i].mask_data);
        binary_mask.convertTo(binary_mask, CV_8U);
        cv::Mat color_mask;
        cv::cvtColor(binary_mask, color_mask, cv::COLOR_GRAY2BGR);

        auto mask_color = getRandomColor();
        color_mask.setTo(mask_color, binary_mask==255);

        cv::imshow("single mask", color_mask);
        cv::waitKey(0);

        cv::resize(binary_mask, binary_mask, cv::Size(patch.cols, patch.rows));
        cv::resize(color_mask, color_mask, cv::Size(patch.cols, patch.rows));
        cv::Mat tmp;
        cv::addWeighted(patch, 0.5, color_mask, 0.5, 0, tmp);
        tmp.copyTo(patch, binary_mask);

        cv::rectangle(new_img, r, box_color, 1);
        std::stringstream label;
        label << "Class_id: " << seg_res_ptr[i].cls << " conf: " << std::setprecision(2) << seg_res_ptr[i].confidence;
        new_img(cv::Rect(cv::Point(seg_res_ptr[i].tl_x, seg_res_ptr[i].br_y), cv::Point(seg_res_ptr[i].br_x, seg_res_ptr[i].br_y+30))) = box_color;
        cv::putText(new_img, label.str(), cv::Point(seg_res_ptr[i].tl_x, seg_res_ptr[i].br_y+20), cv::FONT_HERSHEY_SIMPLEX, 0.6, font_color);
    }
    cv::imshow("Show Result", new_img);
    cv::waitKey(stop_period);

    if(is_save){
        std::string save_file_name = getTimeNow() + ".jpg";
        std::replace(save_file_name.begin(), save_file_name.end(), ':', '-');
        std::replace(save_file_name.begin(), save_file_name.end(), ' ', '-');
        std::cout << "Save to: " << save_file_name << std::endl;
        if(cv::imwrite(save_file_name, new_img)){
            std::cout << "Save Image Success.\n";
        }else{
            std::cout << "Save Image Failed.\n";
        }    
    }
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

void checkImageChannel(cv::Mat& input_mat){
    if(input_mat.channels()>3){
        throw std::runtime_error("Error, image has more than 3 channels");
    }else if(input_mat.channels()==1){
        cv::cvtColor(input_mat, input_mat, cv::COLOR_GRAY2BGR);
    }
}

cv::Scalar getRandomColor(){
    cv::RNG rng((unsigned)cv::getTickCount());
    int b = rng.uniform(0, 256);
    int g = rng.uniform(0, 256);
    int r = rng.uniform(0, 256);
    return cv::Scalar(b, g, r);
}