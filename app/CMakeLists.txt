project(test_model)

add_executable(${PROJECT_NAME} do_infer.cpp)
if(COMPILE_CLS_LIB)
    target_link_libraries(${PROJECT_NAME} PUBLIC cls_infer_lib)
elseif(COMPILE_DET_LIB)
    target_link_libraries(${PROJECT_NAME} PUBLIC det_infer_lib)
elseif(COMPILE_SEG_LIB)
    target_link_libraries(${PROJECT_NAME} PUBLIC seg_infer_lib)
elseif(COMPILE_OCR_LIB)
    target_link_libraries(${PROJECT_NAME} PUBLIC ocr_infer_lib)
else()
    message(WARNING "No Lib Selected!")
endif()

if(COMPILE_HTTP_SERVER)
    # find_package()
    add_executable(Server http_server/server.cxx)
    target_link_libraries(Server PUBLIC )
endif()


 