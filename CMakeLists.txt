cmake_minimum_required(VERSION 2.8)

project(acupas)
find_package(Threads)

add_executable(acupa svmocas.c lib_svmlight_format.c
    evaluate_testing.c evaluate_multiclass.c sparse_mat.c
    ocas_helper.c features_double.c libocas.c libqp_splx.c)
target_link_libraries(acupa m ${CMAKE_THREAD_LIBS_INIT})

add_executable(acupam msvmocas.c lib_svmlight_format.c
    evaluate_testing.c evaluate_multiclass.c sparse_mat.c
    ocas_helper.c features_double.c libocas.c libqp_splx.c)
target_link_libraries(acupam m ${CMAKE_THREAD_LIBS_INIT})
