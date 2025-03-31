#include "inference.h"

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
    std::cout << "Call SEG_RES destroy func" << std::endl;
    if(mask_data){
        delete[] mask_data;
        mask_data = nullptr;
    }
}

void printInfo(){
    std::cout << "OnnxRuntime Detection Inference Demo" << std::endl;
}

SEG_RES* run(const char* image_path, const char* onnx_path, int& det_num) {
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
    const std::array<const char*, 2> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str() };
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "outputname size: " << output_node_names.size() << std::endl;
    std::cout << "ort_outputs size: " << ort_outputs.size() << std::endl;

    // output data
    const float* pred_data = ort_outputs[0].GetTensorMutableData<float>();
    const float* proto_data = ort_outputs[1].GetTensorMutableData<float>();
    cv::Mat preds(output_h, output_w, CV_32F, (float*)pred_data);
    preds = preds.t();
    cv::Mat proto(32, 160*160, CV_32F, (float*)proto_data);

    double scale_r = std::min((double)input_h/frame.rows, (double)input_w/frame.cols);

    SEG_RES* result = postProcess(0.5f, preds, proto, cv::Size(frame.cols, frame.rows), cv::Size(640, 640), det_num);

    env.release();
    session_ptr->release();
    return result;
}


void* initModel(const char* onnx_pth, char* msg){
    // OnnxRuntime 第一次推理和后续推理的耗时差不多，所以不做 WarmUp
    std::string onnxpath{onnx_pth};
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    try{
        ENV_PTR = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Classification");
        auto session_ptr = new Ort::Session(*ENV_PTR, modelPath.c_str(), Ort::SessionOptions());
        std::cout << "Init Session Success." << std::endl;
        return session_ptr;
    }catch(const std::exception& e){
        std::cout << e.what() << std::endl;
        return nullptr;
    }
}

MY_DLL SEG_RES* doInferenceByImgPth(const char* img_pth, void* model_ptr, const int* roi, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;

    try{
        cv::Mat img = cv::imread(img_pth);
        cv::Mat img_part;
        if(roi)
            img_part = img(cv::Rect(cv::Point(roi[0], roi[1]), cv::Point(roi[2], roi[3])));
        else
            img.copyTo(img_part); 

        return doInferenceByImgMat(img_part, model_ptr, score_threshold, det_num, msg);
    }catch(const std::exception& e){
        msg_ss << e.what() << std::endl;
        std::cout << e.what() << std::endl;
    }
}

SEG_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, void* model_ptr, const float score_threshold, int& det_num, char* msg){
    std::stringstream msg_ss;
    // 如果 width 和 height 与实际图像不符，出来的图像会扭曲，但不会报错
    cv::Mat img(cv::Size(width, height), CV_8UC3, image_arr);
    return doInferenceByImgMat(img, model_ptr, 0.5f, det_num, msg);
}

SEG_RES* doInferenceByImgMat(const cv::Mat& img_mat, void* model_ptr, const float score_threshold, int& det_num, char* msg){
#ifdef DEBUG_ORT_
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
    fs << "[" << getTimeNow() << "]" << "Call <doInferenceByImgMat> Func\n";
#endif

    std::stringstream msg_ss;
    try{
        if(model_ptr==nullptr)
            throw std::runtime_error("Error Model Pointer Convert Failed!");
        Ort::Session* session_ptr = static_cast<Ort::Session*>(model_ptr);

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
        const std::array<const char*, 2> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str() };
        std::vector<Ort::Value> ort_outputs;
        try {
            ort_outputs = session_ptr->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
        }
        catch (const std::exception& e) {
            std::cout << e.what() << std::endl;
        }
        // output data
        const float* pred_data = ort_outputs[0].GetTensorMutableData<float>();
        const float* proto_data = ort_outputs[1].GetTensorMutableData<float>();
        cv::Mat preds(output_h, output_w, CV_32F, (float*)pred_data);
        preds = preds.t();
        cv::Mat proto(32, 160*160, CV_32F, (float*)proto_data);
    
        // double scale_r = std::min((double)input_h/img_mat.rows, (double)input_w/img_mat.cols);
        return postProcess(0.5f, preds, proto, cv::Size(img_mat.cols, img_mat.rows), cv::Size(input_w, input_h), det_num);

        // postProcess(score_threshold, dout, scale_r, result);
        // det_num = result.size();

        // return result.data();
        // msg_ss << "cls: " << maxP.x << " score: " << maxScore << std::endl;
        // strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    }catch(const std::exception& e){
        msg_ss << e.what() << std::endl;
        strcpy_s(msg, msg_ss.str().length(), msg_ss.str().c_str());
        return {};
    }

#ifdef DEBUG_ORT_
    fs << "[" << getTimeNow() << "]" << msg_ss.str() << "\n";
    if(fs.is_open())
        fs.close();
#endif
}

void warmUp(Ort::Session& model){
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes = model.GetInputCount();
    size_t numOutputNodes = model.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 获取输入信息
    int input_w{}, input_h{};
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = model.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = model.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        // msg_ss << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 获取输出信息  1*84*8400
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = model.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1]; // 84
    output_w = output_dims[2]; // 8400
    // msg_ss << "output format : HxW = " << output_h << "x" << output_w << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = model.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    // msg_ss << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

    cv::Mat blob;
    preProcess(&model, cv::Mat(640, 640, CV_8UC3, cv::Scalar(1)), blob);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 2> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str() };
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = model.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

void destroyModel(void* model_ptr){
#ifdef DEBUG_ORT_
    std::fstream fs{"./debug_log.txt", std::ios_base::app};
#endif
    if(model_ptr!=nullptr){
        Ort::Session* session_ptr = static_cast<Ort::Session*>(model_ptr);
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

void* preProcess(const Ort::Session* infer_session, const cv::Mat& org_img, cv::Mat& blob){
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

SEG_RES* postProcess(const float conf_threshold, const cv::Mat& pred_mat, const cv::Mat& proto_mat, const cv::Size& org_size, const cv::Size& infer_size, int& det_num){
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
