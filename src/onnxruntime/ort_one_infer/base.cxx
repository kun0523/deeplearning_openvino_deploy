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

Base::Base(const char* model_pth_, char* msg):my_msg(msg){
    // 加载模型文件
    std::ifstream file(model_pth_, std::ios::binary);
    if (!file){
        // 模型文件读取失败
        throw std::runtime_error("Read Model Failed!");
    }

    std::string onnxpath{model_pth_};
    std::wstring model_wpath = std::wstring(onnxpath.begin(), onnxpath.end());    
    static Ort::Env ORT_ENV(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntime");
    model_ptr = std::make_unique<Ort::Session>(ORT_ENV, model_wpath.c_str(), Ort::SessionOptions());
}

Base::~Base(){
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

    if(model_ptr == nullptr)
        throw std::runtime_error("Model Pointer Convert Failed!");

    vector<std::string> input_node_names;
    vector<std::string> output_node_names;
    size_t numInputNodes = model_ptr->GetInputCount();
    if (numInputNodes != 1){
        std::cout << "Input Nodes Num: " << numInputNodes << std::endl;
        throw std::runtime_error("InputNodes != 1");
    }
    input_node_names.reserve(numInputNodes);

    size_t numOutputNodes = model_ptr->GetOutputCount();
    if (numOutputNodes != 1){
        std::cout << "Output Nodes Num: " << numOutputNodes << std::endl;
        throw std::runtime_error("OutputNodes != 1");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    // 获取输入信息
    auto input_name = model_ptr->GetInputNameAllocated(0, allocator);
    input_node_names.push_back(input_name.get());
    Ort::TypeInfo input_type_info = model_ptr->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();
    int input_w = input_dims[3];
    int input_h = input_dims[2];
    msg_ss << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;

    // 获取输出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = model_ptr->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[0]; // 1
    output_w = output_dims[1]; // 1000
    msg_ss << "output format : HxW = " << output_h << "x" << output_w << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = model_ptr->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    msg_ss << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

    // format frame
    cv::Mat blob = cv::dnn::blobFromImage(img_mat, 1.0 / 255.0, cv::Size(input_w, input_h), 0.0, true, false);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
    vector<Ort::Value> ort_outputs;
    std::cout << msg_ss.str() << std::endl;

    ort_outputs = model_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    // output data
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);

    cv::Point maxP;
    double maxScore;
    cv::minMaxLoc(dout, 0, &maxScore, 0, &maxP);
    msg_ss << "cls: " << maxP.x << " score: " << maxScore << std::endl;

    result_ptr = new CLS_RES[1]{CLS_RES(maxP.x, maxScore)};
    result_len = 1;
    num = result_len;

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

    if(model_ptr == nullptr)
        throw std::runtime_error("Model Pointer Convert Failed!");

    vector<std::string> input_node_names;
    vector<std::string> output_node_names;
    size_t numInputNodes = model_ptr->GetInputCount();
    size_t numOutputNodes = model_ptr->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 获取输入信息
    int input_w{}, input_h{};
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = model_ptr->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = model_ptr->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        msg_ss << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 获取输出信息  1*84*8400
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = model_ptr->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1]; // 84
    output_w = output_dims[2]; // 8400
    msg_ss << "output format : HxW = " << output_h << "x" << output_w << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = model_ptr->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    msg_ss << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

    cv::Mat blob;
    preProcess(img_mat, blob);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
    vector<Ort::Value> ort_outputs;
    ort_outputs = model_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());

    // output data
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
    double scale_r = std::min((double)input_h/img_mat.rows, (double)input_w/img_mat.cols);

    result_ptr = postProcess(conf_threshold, dout, scale_r, det_num);
    result_len = det_num;

    return result_ptr;
}

void Detection::preProcess(const cv::Mat& org_img, cv::Mat& blob){
    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto input_tensor_shape = model_ptr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int board_h = input_tensor_shape[2], board_w = input_tensor_shape[3];
    auto boarded_img = cv::Mat(cv::Size(board_w, board_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min(board_h/org_h, board_w/org_w);
    cv::Mat resize_img;
    cv::resize(org_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));

    blob = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(board_w, board_h), cv::Scalar(0,0,0), true, false);
}

DET_RES* Detection::postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double& scale_ratio_, int& det_num){
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

Detection::~Detection(){

}

void Detection::drawResult(const short stop_period, const bool is_save) const {
    if(!result_ptr)
        return;

    cv::Mat new_img;
    infer_img.copyTo(new_img);
    auto box_color = cv::Scalar(0, 100, 200);
    auto font_color = cv::Scalar(200,200,200);
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

    if(model_ptr==nullptr)
        throw std::runtime_error("Model Pointer Convert Failed!");

    vector<std::string> input_node_names;
    vector<std::string> output_node_names;
    size_t numInputNodes = model_ptr->GetInputCount();
    size_t numOutputNodes = model_ptr->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 获取输入信息  1x3x640x640
    auto input_shape = model_ptr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int input_h = input_shape[2], input_w = input_shape[3]; 
    auto input_name = model_ptr->GetInputNameAllocated(0, allocator);
    input_node_names.push_back(input_name.get());

    // 获取输出信息  output0 1x116x8400    output1  1x32x160x160
    auto pred_shape = model_ptr->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int pred_h = pred_shape[1], pred_w = pred_shape[2];
    auto proto_shape = model_ptr->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    int proto_c = proto_shape[1], proto_h = proto_shape[2], proto_w = proto_shape[3];

    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = model_ptr->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    msg_ss << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

    cv::Mat blob;
    preProcess(img_mat, blob);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 2> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str() };
    vector<Ort::Value> ort_outputs;
    ort_outputs = model_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());

    // output data
    const float* pred_data = ort_outputs[0].GetTensorMutableData<float>();
    const float* proto_data = ort_outputs[1].GetTensorMutableData<float>();
    cv::Mat preds(pred_h, pred_w, CV_32F, (float*)pred_data);
    preds = preds.t();
    cv::Mat proto(proto_c, proto_h*proto_w, CV_32F, (float*)proto_data);

    result_ptr = postProcess(conf_threshold, preds, proto, cv::Size(img_mat.cols, img_mat.rows), cv::Size(input_w, input_h), det_num);
    result_len = det_num;
    return result_ptr;
}

void Segmentation::preProcess(const cv::Mat& org_img, cv::Mat& blob){
    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto input_tensor_shape = model_ptr->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int board_h = input_tensor_shape[2], board_w = input_tensor_shape[3];
    auto boarded_img = cv::Mat(cv::Size(board_w, board_h), CV_8UC3, cv::Scalar(114, 114, 114));

    double ratio = std::min(board_h/org_h, board_w/org_w);
    cv::Mat resize_img;
    cv::resize(org_img, resize_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
    resize_img.copyTo(boarded_img(cv::Rect(cv::Point(0,0), cv::Point(resize_img.cols, resize_img.rows))));

    blob = cv::dnn::blobFromImage(boarded_img, 1/255.0, cv::Size(board_w, board_h), cv::Scalar(0,0,0), true, false);
}

SEG_RES* Segmentation::postProcess(const float conf_threshold, const cv::Mat& pred_mat, const cv::Mat& proto_mat, const cv::Size& org_size, const cv::Size& infer_size, int& det_num){
    size_t pred_num = pred_mat.cols;

    vector<cv::Rect2d> boxes;
    vector<float> scores;
    vector<int> indices;
    vector<int> class_idx;
    vector<cv::Mat> masks;
    float tl_x{}, tl_y{}, br_x{}, br_y{}, cx{}, cy{}, w{}, h{}, iou_threshold{0.3f};

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
        // std::cout << "mask shape: " << mask.rows << " " << mask.cols << std::endl;
        mask = mask.reshape(1, 160);
        // std::cout << "after reshape mask shape: " << mask.rows << " " << mask.cols << std::endl;

        // double s = std::min((double)input_h/frame.rows, (double)input_w/frame.cols);
        int n_h = r*org_size.height, n_w = r*org_size.width;
        // std::cout << "cls: " << cls << std::endl;
        // std::cout << tmp.tl() << " ||| " << tmp.br() << std::endl;
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
        std::cout << "mask size: " << color_mask.size << std::endl;
        std::cout << "mask channel: " << color_mask.channels() << std::endl;

        auto mask_color = getRandomColor();
        color_mask.setTo(mask_color, binary_mask==255);

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