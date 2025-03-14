#include "inference.h"

void printInfo(char* ret_msg, size_t msg_len){

    cout << "Paddle Det Show Info..." << endl;
    std::stringstream msg;
    auto backends = fastdeploy::GetAvailableBackends();
    msg << "Available Backends: \n";
    for(auto e:backends){
        msg << e << '\n';
    }    
    std::memcpy(ret_msg, msg.str().c_str(), msg_len);

    ov::Core core;
    auto devices = core.get_available_devices();
    auto version = core.get_versions(core.get_available_devices()[0]);
    for(const auto& item:version){
        cout << item.first << " : " << item.second << endl;
    }
}

#ifdef USE_FASTDEPLOY

void* initModel(const char* pdmodel_dir, short backend_type, char* msg, size_t msg_len){
    fastdeploy::RuntimeOption option;
    option.UseCpu();
    if(backend_type==PADDLE){
        cout << "Use Paddle Backend\n";
        option.UsePaddleBackend();
    }else if(backend_type==ONNX){
        cout << "Use ONNX Backend\n";
        option.UseOrtBackend();
    }else if(backend_type==OPENVINO){
        cout << "Use OpenVINO Backend\n";
        option.UseOpenVINOBackend();
    }else{
        cout << "Error, Wrong Backend Type: " << backend_type << endl;
    }

    std::string pdmodel_dir_str{pdmodel_dir};
    auto model_file = pdmodel_dir_str + sep + "inference.pdmodel";
    auto params_file = pdmodel_dir_str + sep + "inference.pdiparams";
    auto config_file = pdmodel_dir_str + sep + "inference.yml";
    // auto model = fastdeploy::vision::classification::PaddleClasModel(model_file, params_file, config_file, option);
    
    auto model_ptr = new fastdeploy::vision::classification::PaddleClasModel{model_file, params_file, config_file, option};

    

    if (!model_ptr->Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return nullptr;
    }

    cout << "Model Initialize Success." << endl;

    return model_ptr;
}


DET_RES doInferenceByImgPth(const char* image_pth, void* model_ptr, char* msg, size_t msg_len){
    cout << "Call doInferenceByImgPth Func !!" << endl;
    
    auto pdcls_model_ptr = static_cast<fastdeploy::vision::classification::PaddleClasModel*>(model_ptr);
    if(pdcls_model_ptr == nullptr){
        cout << "Error! model pointer is nullptr";
        return DET_RES{};
    }

    auto im = cv::imread(image_pth);
    fastdeploy::vision::ClassifyResult res;
    try{
        if(!pdcls_model_ptr->Predict(im, &res)){
            std::cerr << "Failed to predict." << std::endl;
            return DET_RES{};
        }

    }catch(const std::exception& ex){
        std::cerr << ex.what() << endl;
    }
    
    cout << "Num: " << res.label_ids.size() << endl;
    cout << "Result: " << res.label_ids.at(0) << " Scores: " << res.scores.at(0) << endl;

    return DET_RES{};
}

DET_RES doInferenceBy3chImg(uchar* image_arr, const int height, const int width, void* model_ptr, char* msg, size_t msg_len){
    cout << "Call doInferenceBy3chImg Func !!" << endl;

    auto pdcls_model_ptr = static_cast<fastdeploy::vision::classification::PaddleClasModel*>(model_ptr);
    if(pdcls_model_ptr == nullptr){
        cout << "Error! model pointer is nullptr";
        return DET_RES{};
    }

    cv::Mat im{cv::Size(width, height), CV_8UC3, image_arr};
    fastdeploy::vision::ClassifyResult res;
    try{
        if(!pdcls_model_ptr->Predict(im, &res)){
            std::cerr << "Failed to predict." << std::endl;
            return DET_RES{};
        }

    }catch(const std::exception& ex){
        std::cerr << ex.what() << endl;
    }
    
    cout << "Num: " << res.label_ids.size() << endl;
    cout << "Result: " << res.label_ids.at(0) << " Scores: " << res.scores.at(0) << endl;

    return DET_RES{};
}


#else

void* initModel(const char* onnx_pth, char* msg, size_t msg_len){
    std::stringstream msg_ss;
    msg_ss << "Call <initModel> Func\n";
    ov::Core core;  
    ov::CompiledModel* compiled_model_ptr = nullptr;  

    try{      
        // auto model = core.read_model(R"(E:\le_trt\models\yolo11s01_int8_openvino_model\yolo11s01.xml)", 
        //                             R"(E:\le_trt\models\yolo11s01_int8_openvino_model\yolo11s01.bin)");
        // compiled_model_ptr = new ov::CompiledModel(core.compile_model(model, "CPU", 
        //                                                             ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), 
        //                                                             ov::hint::num_requests(4), ov::auto_batch_timeout(1000)));                       

        compiled_model_ptr = new ov::CompiledModel(core.compile_model(onnx_pth, "CPU"));
        if(compiled_model_ptr==nullptr){
            throw std::runtime_error("Create Model Failed!");
        }
        msg_ss << "Create Compiled model Success. Got Model Pointer: " << compiled_model_ptr << "\n";
    }catch(const std::exception& ex){
        msg_ss << "Create Model Failed\n";
        msg_ss << "Error Message: " << ex.what() << "\n";
        cout << "-------------" << ex.what() << "-------------" << endl;

        // strcpy_s(msg, msg_len, msg_ss.str().c_str());
        return compiled_model_ptr;
    }

    // // warmUp(compiled_model_ptr, msg_ss); 
    // strcpy_s(msg, msg_len, msg_ss.str().c_str());
    return compiled_model_ptr;

}


DET_RES* doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, const float score_threshold, const short model_type, size_t& det_num, char* msg){

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

    // // TODO: 增强
    // img_mat.convertTo(img_mat, -1, 1.2, 3);
    cv::Mat resized_img;
    double scale_ratio;
    int left_padding_cols, top_padding_rows;
    resizeImageAsYOLO(*model_ptr, img_mat, resized_img, scale_ratio, left_padding_cols, top_padding_rows);

    cv::Mat blob_img = cv::dnn::blobFromImage(resized_img, 1.0/255.0, cv::Size(input_tensor_shape[3], input_tensor_shape[2]), 0.0, true, false, CV_32F);
    ov::Tensor inputensor;
    opencvMat2Tensor(blob_img, *model_ptr, inputensor);
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
    // YOLOV10 需要指定 false  YOLOV8 v11 指定 true；
    postProcess(score_threshold, m, scale_ratio, left_padding_cols, top_padding_rows, model_type, out_res_vec);
    DET_RES* det_res = new DET_RES[out_res_vec.size()];
    det_num = out_res_vec.size();
    int counter = 0;
    for(auto it=out_res_vec.begin(); it!=out_res_vec.end(); ++it){
        it->tl_x = std::max(0, it->tl_x);
        it->tl_y = std::max(0, it->tl_y);
        it->br_x = std::min(it->br_x, img_mat.cols);
        it->br_y = std::min(it->br_y, img_mat.rows);
        det_res[counter++] = *it;
        // cout << it->tl_x << " " << it->tl_y << " " << it->br_x << " " << it->br_y << " cls: " << it->cls << " conf: " << it->confidence << endl;
    }
    msg_ss << "Detect Object Num: " << det_num << "\n";
    msg_ss << "---- Inference Over ----\n";
    strcpy_s(msg, msg_ss.str().length()+1, msg_ss.str().c_str());
    return det_res;
}


#endif
