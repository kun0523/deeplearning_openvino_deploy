#include "inference.h"

using std::endl;
using std::string;
using std::vector;

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

void printInfo(){
    std::cout << "OnnxRuntime Detection Inference Demo" << std::endl;
}

void run(const char* image_path, const char* onnx_path) {
    cv::Mat frame = cv::imread(image_path);
    int ih = frame.rows;
    int iw = frame.cols;

    auto env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Detection");
    // 创建InferSession, 查询支持硬件设备
    std::string onnxpath{onnx_path};
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    auto session_ptr = new Ort::Session(env, modelPath.c_str(), Ort::SessionOptions());

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    size_t numInputNodes = session_ptr->GetInputCount();
    size_t numOutputNodes = session_ptr->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 获取输入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session_ptr->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session_ptr->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 获取输出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session_ptr->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1]; // 84
    output_w = output_dims[2]; // 8400
    std::cout << "output format : HxW = " << output_h << "x" << output_w << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session_ptr->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

    cv::Mat blob;
    preProcess(session_ptr, frame, blob);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    // output data
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);

    std::cout << dout.size << std::endl;
    double scale_r = std::min(frame.rows/640.0, frame.cols/640.0);
    int det_num{};
    DET_RES* result = postProcess(0.5f, dout, scale_r, det_num);
    for(int i{}; i<det_num; ++i){
        std::cout << result[i].get_info() << std::endl;
    }

    // session_options.release();
    session_ptr->release();
    return ;
}

void initModel(const char* model_pth, char* msg){
    // OnnxRuntime 第一次推理和后续推理的耗时差不多，所以不做 WarmUp
    std::string onnxpath{model_pth};
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    try{
        ENV_PTR = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Classification");
        gModelPtr = new Ort::Session(*ENV_PTR, modelPath.c_str(), Ort::SessionOptions());
        std::cout << "Init Session Success." << std::endl;
        return;
    }catch(const std::exception& e){
        std::cout << e.what() << std::endl;
        return;
    }
}

MY_DLL DET_RES* doInferenceByImgPth(const char* img_pth, const int* roi, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

    try{
        cv::Mat img = cv::imread(img_pth);
        cv::Mat img_part;
        if(roi)
            img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
        else
            img.copyTo(img_part); 

        return doInferenceByImgMat(img_part, score_threshold, det_num, msg);
    }catch(const std::exception& e){
        msg_ss << e.what() << std::endl;
        std::cout << e.what() << std::endl;
    }
}

DET_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;
    // 如果 width 和 height 与实际图像不符，出来的图像会扭曲，但不会报错
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return doInferenceByImgMat(img, score_threshold, det_num, msg);
}

DET_RES* doInferenceByImgMat(const cv::Mat& img_mat, const float score_threshold, int& det_num, char* msg){
#ifdef DEBUG_ORT_
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << "[" << getTimeNow() << "]" << "Call <doInferenceByImgMat> Func\n";
#endif

    std::stringstream msg_ss;
    try{
        if(gModelPtr==nullptr)
            throw std::runtime_error("Error Model Pointer Convert Failed!");
        Ort::Session* session_ptr = static_cast<Ort::Session*>(gModelPtr);

        std::vector<std::string> input_node_names;
        std::vector<std::string> output_node_names;
        size_t numInputNodes = session_ptr->GetInputCount();
        size_t numOutputNodes = session_ptr->GetOutputCount();
        Ort::AllocatorWithDefaultOptions allocator;
        input_node_names.reserve(numInputNodes);

        // 获取输入信息
        int input_w{}, input_h{};
        for (int i = 0; i < numInputNodes; i++) {
            auto input_name = session_ptr->GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            Ort::TypeInfo input_type_info = session_ptr->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            input_w = input_dims[3];
            input_h = input_dims[2];
            msg_ss << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
        }

        // 获取输出信息  1*84*8400
        int output_h = 0;
        int output_w = 0;
        Ort::TypeInfo output_type_info = session_ptr->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_h = output_dims[1]; // 84
        output_w = output_dims[2]; // 8400
        msg_ss << "output format : HxW = " << output_h << "x" << output_w << std::endl;
        for (int i = 0; i < numOutputNodes; i++) {
            auto out_name = session_ptr->GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(out_name.get());
        }
        msg_ss << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

        cv::Mat blob;
        preProcess(session_ptr, img_mat, blob);
        size_t tpixels = input_h * input_w * 3;
        std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

        // set input data and inference
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
        const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
        const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
        std::vector<Ort::Value> ort_outputs;
        try {
            ort_outputs = session_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
        }
        catch (const std::exception& e) {
            msg_ss << e.what() << std::endl;
        }
        
        // output data
        const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
        cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
        double scale_r = std::min((double)input_h/img_mat.rows, (double)input_w/img_mat.cols);
        return postProcess(score_threshold, dout, scale_r, det_num);

        // msg_ss << "cls: " << maxP.x << " score: " << maxScore << std::endl;
        // strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    }catch(const std::exception& e){
        msg_ss << e.what() << std::endl;
        strcpy_s(msg, msg_ss.str().length(), msg_ss.str().c_str());
        return new DET_RES[1];
    }

#ifdef DEBUG_ORT_
    fs << "[" << getTimeNow() << "]" << msg_ss.str() << "\n";
    if(fs.is_open())
        fs.close();
#endif
}

void destroyModel(){
#ifdef DEBUG_ORT_
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    if(gModelPtr!=nullptr){
        Ort::Session* session_ptr = static_cast<Ort::Session*>(gModelPtr);
        session_ptr->release();        
        ENV_PTR->release();
    }
#ifdef DEBUG_ORT_
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

void preProcess(const Ort::Session* infer_session, const cv::Mat& org_img, cv::Mat& blob){
    double org_h = org_img.rows, org_w = org_img.cols;
    // 前提假设，模型只有一个输入节点
    auto input_tensor_shape = infer_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
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
