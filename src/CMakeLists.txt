project(test_model)

add_subdirectory(cls_infer)
# add_subdirectory(det_infer)
# add_subdirectory(ocr_infer)

add_executable(${PROJECT_NAME} do_infer.cpp)
# target_include_directories(${PROJECT_NAME} PUBLIC ${Cls_Infer_Dir})
target_link_libraries(${PROJECT_NAME} PUBLIC cls_infer_lib)
# target_link_libraries(${PROJECT_NAME} PUBLIC det_infer_lib)
# target_link_libraries(${PROJECT_NAME} PUBLIC ocr_infer_lib)
