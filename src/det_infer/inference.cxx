#include "inference.h"

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

// TODO: 还需要写一个回收的接口
void* initModel(const char* onnx_pth, char* msg){
    // 根据模型onnx文件路径，返回模型指针
    ov::Core core;
    ov::CompiledModel* compiled_model_ptr = nullptr;
    string msg_s;
    try{
        compiled_model_ptr = new ov::CompiledModel(core.compile_model(onnx_pth, "CPU"));
        msg_s = "Create Compiled model Success.";
    }catch(std::exception ex){
        msg_s = string(ex.what());
    }

    strcpy_s(msg, 1024, msg_s.c_str());
    return compiled_model_ptr;
}

void warmUp(void* model_ptr, char* msg){
    string msg_s;
    try{
        ov::CompiledModel* model = static_cast<ov::CompiledModel*>(model_ptr);
        ov::InferRequest infer_request = model->create_infer_request();
        auto input_port = model->input();  // 假设模型只有一个输入
        ov::Shape input_shape = input_port.get_shape();

        cv::Mat blob_img = cv::Mat::ones(cv::Size(1024, 1024), CV_32FC3);
        size_t input_w = input_shape[2], input_h = input_shape[3];
        cv::Mat blob = cv::dnn::blobFromImage(blob_img, 1.0, cv::Size(input_w, input_h));
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        msg_s = "WarmUp Complete.";
    }catch(std::exception ex){
        cout << ex.what() << endl;
        msg_s = string(ex.what());
    }

    strcpy_s(msg, 1024, msg_s.c_str());
}

char* postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double scale_ratio_, const int left_padding, const int top_padding, bool do_nms, std::vector<DET_RES>& out_vec){
    if (do_nms)
        det_result_mat = det_result_mat.t();    
    size_t pred_num = det_result_mat.size().width;

    vector<cv::Rect2d> boxes;
    vector<float> scores;
    vector<int> indices;
    vector<int> class_idx;
    if(do_nms){
        for(int row=0; row<det_result_mat.size[0]; ++row){
            const float* ptr = det_result_mat.ptr<float>(row);
            float tl_x = ptr[0], tl_y = ptr[1], br_x=ptr[2], br_y=ptr[3];
            // float cx = ptr[0], cy = ptr[1], w=ptr[2], h=ptr[3];  // 不清楚什么时候接口变了，直接返回对角坐标
            vector<float> cls_conf = vector<float>(ptr+4, ptr+pred_num);
            cv::Point2i maxP;
            double maxV;
            cv::minMaxLoc(cls_conf, 0, &maxV, 0, &maxP);
            // boxes.emplace_back(cx-w/2, cy-h/2, w, h);
            boxes.emplace_back(tl_x, tl_y, br_x-tl_x, br_y-tl_y);
            scores.push_back(static_cast<float>(maxV));
            class_idx.push_back(maxP.x);
        }
        float iou_threshold = 0.5;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);
    }else{
        for(int row=0; row<det_result_mat.size[0]; ++row){
            const float* ptr = det_result_mat.ptr<float>(row);
            float tl_x = ptr[0], tl_y = ptr[1], br_x=ptr[2], br_y=ptr[3];
            vector<float> cls_conf = vector<float>(ptr+4, ptr+pred_num);
            cv::Point2i maxP;
            double maxV;
            cv::minMaxLoc(cls_conf, 0, &maxV, 0, &maxP);
            boxes.emplace_back(tl_x, tl_y, br_x-tl_x, br_y-tl_y);
            scores.push_back(static_cast<float>(maxV));
            class_idx.push_back(maxP.x);
        }
        indices = vector<int>(boxes.size());
        std::iota(indices.begin(), indices.end(), 0);
    }

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

DET_RES* doInferenceBy3chImg(uchar* image_arr, int height, int width, void* model_ptr, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg){
    // 对三通道的图进行推理
    cv::Mat img(height, width, CV_8UC3, image_arr);
    return doInferenceByImgMat(img, model_ptr, score_threshold, is_use_nms, det_num, msg);
}

DET_RES* doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg){
    ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(compiled_model);
    ov::Shape input_tensor_shape = model_ptr->input().get_shape();
    auto start = std::chrono::high_resolution_clock::now();

    // // TODO: 增强
    // img_mat.convertTo(img_mat, -1, 1, -20);

    cv::Mat resized_img;
    double scale_ratio;
    int left_padding_cols, top_padding_rows;
    resizeImageAsYOLO(*model_ptr, img_mat, resized_img, scale_ratio, left_padding_cols, top_padding_rows);
    cv::Mat blob_img = cv::dnn::blobFromImage(resized_img, 1.0/255.0, cv::Size(input_tensor_shape[2], input_tensor_shape[3]), 0.0, true, false, CV_32F);
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
    strcpy_s(msg, 1024, msg_ss.str().c_str());

    ov::Shape output_tensor_shape = model_ptr->output().get_shape();
    size_t batch_num=output_tensor_shape[0], res_height=output_tensor_shape[1], res_width=output_tensor_shape[2];
    auto output_tensor = infer_request.get_output_tensor();
    const float* output_buff = output_tensor.data<const float>();
    cv::Mat m = cv::Mat(cv::Size(res_width, res_height), CV_32F, const_cast<float*>(output_buff));

    vector<DET_RES> out_res_vec;
    // YOLOV10 需要指定 false  YOLOV8 指定 true；
    postProcess(score_threshold, m, scale_ratio, left_padding_cols, top_padding_rows, is_use_nms, out_res_vec);
    DET_RES* det_res = new DET_RES[out_res_vec.size()];
    det_num = out_res_vec.size();
    int counter = 0;
    for(auto it=out_res_vec.begin(); it!=out_res_vec.end(); ++it){
        it->tl_x = std::max(0, it->tl_x);
        it->tl_y = std::max(0, it->tl_y);
        it->br_x = std::min(it->br_x, img_mat.cols);
        it->br_y = std::min(it->br_y, img_mat.rows);
        det_res[counter++] = *it;
    }

    return det_res;
}

char* resizeImageAsYOLO(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& boarded_img, double& scale_ratio, int& left_padding, int& top_padding){
    
    double new_width = compiled_model.input().get_shape()[2], new_height = compiled_model.input().get_shape()[3];
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

DET_RES* doInferenceByImgPth(const char* img_pth, const int* roi, const int roi_len, void* model_ptr, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg){
    // 对三通道的图进行推理
    cv::Mat img_org = cv::imread(img_pth);
    cv::Mat img;
    img_org(cv::Rect2i(0, 0, 100, 100));

    return doInferenceByImgMat(img, model_ptr, score_threshold, is_use_nms, det_num, msg);
}