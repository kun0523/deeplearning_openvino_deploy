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

void* initModel(const char* onnx_pth, char* msg){
    std::stringstream msg_ss;
    msg_ss << "Call <initModel> Func\n";
    ov::Core core;
    ov::CompiledModel* compiled_model_ptr = nullptr;

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
        strcpy_s(msg, 1024, msg_ss.str().c_str());
        return compiled_model_ptr;
    }   

    try{
        // 创建模型
        // std::shared_ptr<ov::Model> model = core.read_model(onnx_pth);
        // model->get_parameters()[0]->set_layout("NCHW");
        // ov::set_batch(model, 5);
        // cout << model->input().get_shape() << endl;

        compiled_model_ptr = new ov::CompiledModel(core.compile_model(onnx_pth, "CPU"));
        core.set_property("CPU", ov::num_streams(6));
        msg_ss << "Create Compiled model Success. Got Model Pointer: " << compiled_model_ptr << "\n";
    }catch(std::exception ex){
        msg_ss << "Create Model Failed\n";
        msg_ss << "Error Message: " << ex.what() << "\n";

        strcpy_s(msg, 1024, msg_ss.str().c_str());
        return compiled_model_ptr;
    }

    warmUp(compiled_model_ptr, msg); 
    msg_ss << msg; 
    strcpy_s(msg, 1024, msg_ss.str().c_str());
    return compiled_model_ptr;
}

void warmUp(void* model_ptr, char* msg){
    cout << "msg init: " << msg << endl;
    std::stringstream msg_ss;
    msg_ss << "Call <warmUp> Func ...\n";

    try{
        msg_ss << "WarmUp Model Pointer: " << model_ptr << "\n";        
        cv::Mat blob_img = cv::Mat::ones(cv::Size(1024, 1024), CV_32FC3);
        size_t det_num;
        doInferenceByImgMat(blob_img, model_ptr, 0.5f, true, det_num, msg);
        msg_ss << msg;
        msg_ss << "WarmUp Complete.";
    }catch(std::exception ex){
        msg_ss << "Catch Error in Warmup Func\n";
        msg_ss << "Error Message: " << ex.what() << endl;
    }

    strcpy_s(msg, 1024, msg_ss.str().c_str());
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

DET_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, void* model_ptr, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg){
    // 对三通道的图进行推理
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return doInferenceByImgMat(img, model_ptr, score_threshold, is_use_nms, det_num, msg);
}

DET_RES* doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg){

    std::stringstream msg_ss;
    msg_ss << "Call <doInferenceByImgMat> Func\n";

    ov::CompiledModel* model_ptr = static_cast<ov::CompiledModel*>(compiled_model);
    if(model_ptr==nullptr){
        msg_ss << "Error, Got nullptr, Model pointer convert failed\n";
        return nullptr;
    }else{
        msg_ss << "Convert Model Pointer Success.\n";
        msg_ss << "Got Inference Model Pointer: " << model_ptr << "\n";                
    }
    ov::Shape input_tensor_shape = model_ptr->input().get_shape();
    msg_ss << "Model Input Shape: " << input_tensor_shape << "\n";

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
    msg_ss << "Image Preprocess cost: " << img_preprocess_cost.count() << "ms Infer cost: " << inference_cost.count() << "ms\n";

    ov::Shape output_tensor_shape = model_ptr->output().get_shape();
    msg_ss << "Model Output Shape: " << output_tensor_shape << "\n";

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
        cout << it->tl_x << " " << it->tl_y << " " << it->br_x << " " << it->br_y << " cls: " << it->cls << " conf: " << it->confidence << endl;
    }
    msg_ss << "Detect Object Num: " << det_num << "\n";
    msg_ss << "---- Inference Over ----\n";
    strcpy_s(msg, 1024, msg_ss.str().c_str());
    return det_res;
    // return nullptr;
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

DET_RES* doInferenceByImgPth(const char* img_pth, void* model_ptr, const int* roi, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg){
    // 对三通道的图进行推理
    cv::Mat org_img = cv::imread(img_pth, cv::IMREAD_COLOR);
    cv::Mat img_part;
    if(roi)
        img_part = org_img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    else
        org_img.copyTo(img_part); 

    return doInferenceByImgMat(img_part, model_ptr, score_threshold, is_use_nms, det_num, msg);
}

void multiThreadInference(const cv::Mat& org_img, const int* roi, void* model_ptr, const float& score_threshold, const bool& is_use_nms, std::vector<DET_RES>* res_vec, std::mutex& mtx, char* msg){
    // cout << "test threads" << endl;  // 只是打印信息 基础耗时 5ms/perThread
    // TODO 可能会有roi异常  需要加异常处理
    // cout << "roi:" << cv::Point(roi[0], roi[1]) << cv::Point(roi[2], roi[3]) << endl;
    cv::Mat img_part = org_img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
    size_t det_num{0};
    DET_RES* res = doInferenceByImgMat(img_part, model_ptr, score_threshold, is_use_nms, det_num, msg);
    // for(int i=0; i<det_num; ++i){
    //     std::lock_guard<std::mutex> lock(mtx);
    //     res_vec->push_back(res[i]);
    // }

}


DET_RES* doInferenceBy3chImgPatches(uchar* image_arr, const int height, const int width, const int patch_size, const int overlap_size, void* model_ptr, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg){
    std::vector<std::array<int, 4> > roiList;
    int x_end = 0, y_end = 0;
    for(int y_start=0; y_start<height; y_start+=(patch_size-overlap_size)){
        for(int x_start=0; x_start<width; x_start+=(patch_size-overlap_size)){
            x_end = std::min({width, x_start+patch_size});
            y_end = std::min(height, y_start+patch_size);
            x_start = x_end - patch_size;
            y_start = y_end - patch_size;
            roiList.push_back({x_start, y_start, x_end, y_end});          

            if(x_end == width) break;
        }
        if(y_end==height) break;
    }

    std::mutex mtx;
    std::vector<DET_RES>* det_res_vec = new std::vector<DET_RES>();
    std::vector<std::thread> threads;
    cv::Mat org_img(cv::Size(width, height), CV_8UC3, image_arr);
    cout << "Patch num: " << roiList.size() << endl;
    for(const auto& roi:roiList){
        threads.push_back(std::thread(multiThreadInference, std::ref(org_img), roi.data(), model_ptr, std::ref(score_threshold), std::ref(is_use_nms), det_res_vec, std::ref(mtx), msg));
        // multiThreadInference(std::ref(org_img), roi.data(), model_ptr, std::ref(score_threshold), std::ref(is_use_nms), det_res_vec, std::ref(mtx), msg);
    }
    // threads.push_back(std::thread(multiThreadInference, org_img, roiList[0].data(), model_ptr, score_threshold, is_use_nms, det_res_vec, std::ref(mtx), msg));
    // threads.push_back(std::thread(multiThreadInference, org_img, roiList[1].data(), model_ptr, score_threshold, is_use_nms, det_res_vec, std::ref(mtx), msg));
    // threads.push_back(std::thread(multiThreadInference, org_img, roiList[2].data(), model_ptr, score_threshold, is_use_nms, det_res_vec, std::ref(mtx), msg));
    // threads.push_back(std::thread(multiThreadInference, org_img, roiList[3].data(), model_ptr, score_threshold, is_use_nms, det_res_vec, std::ref(mtx), msg));
    // threads.push_back(std::thread(multiThreadInference, org_img, roiList[4].data(), model_ptr, score_threshold, is_use_nms, det_res_vec, std::ref(mtx), msg));

    for(auto& t:threads){
        t.join();
    }
    det_num = det_res_vec->size();

    return det_res_vec->data();
}