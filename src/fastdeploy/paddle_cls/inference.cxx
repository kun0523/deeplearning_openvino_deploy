#include "inference.h"

void printInfo(char* ret_msg, size_t msg_len){

    std::stringstream msg;
    auto backends = fastdeploy::GetAvailableBackends();
    msg << "Available Backends: \n";
    for(auto e:backends){
        msg << e << '\n';
    }    
    std::memcpy(ret_msg, msg.str().c_str(), msg_len);
}

// TODO: 待做batch推理

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

CLS_RES doInferenceByImgPth(const char* image_pth, void* model_ptr, char* msg, size_t msg_len){
    cout << "Call doInferenceByImgPth Func !!" << endl;
    
    auto pdcls_model_ptr = static_cast<fastdeploy::vision::classification::PaddleClasModel*>(model_ptr);
    if(pdcls_model_ptr == nullptr){
        cout << "Error! model pointer is nullptr";
        return CLS_RES{0, -1.0};
    }

    auto im = cv::imread(image_pth);
    fastdeploy::vision::ClassifyResult res;
    try{
        if(!pdcls_model_ptr->Predict(im, &res)){
            std::cerr << "Failed to predict." << std::endl;
            return CLS_RES{0, -1.0};
        }

    }catch(const std::exception& ex){
        std::cerr << ex.what() << endl;
    }
    
    cout << "Num: " << res.label_ids.size() << endl;
    cout << "Result: " << res.label_ids.at(0) << " Scores: " << res.scores.at(0) << endl;

    return CLS_RES{res.label_ids.at(0), res.scores.at(0)};
}

CLS_RES doInferenceBy3chImg(uchar* image_arr, const int height, const int width, void* model_ptr, char* msg, size_t msg_len){
    cout << "Call doInferenceBy3chImg Func !!" << endl;

    auto pdcls_model_ptr = static_cast<fastdeploy::vision::classification::PaddleClasModel*>(model_ptr);
    if(pdcls_model_ptr == nullptr){
        cout << "Error! model pointer is nullptr";
        return CLS_RES{0, -1.0};
    }

    cv::Mat im{cv::Size(width, height), CV_8UC3, image_arr};
    fastdeploy::vision::ClassifyResult res;
    try{
        if(!pdcls_model_ptr->Predict(im, &res)){
            std::cerr << "Failed to predict." << std::endl;
            return CLS_RES{0, -1.0};
        }

    }catch(const std::exception& ex){
        std::cerr << ex.what() << endl;
    }
    
    cout << "Num: " << res.label_ids.size() << endl;
    cout << "Result: " << res.label_ids.at(0) << " Scores: " << res.scores.at(0) << endl;

    return CLS_RES{res.label_ids.at(0), res.scores.at(0)};
}



