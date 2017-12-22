project (classifier)

add_definitions(-std=c++11 -g)
set(CMAKE_EXE_LINKER_FLAGS "-pthread -lrt")

file(GLOB_RECURSE classifier_SOURCES "src/*.cpp")

set (classifier_INCLUDE_DIRS "")
foreach (_headerFile ${classifier_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND classifier_INCLUDE_DIRS ${_dir})
endforeach()

#set caffe include path, such as "/home/work/caffe_ssd"
set (caffe_ssd_PATH "/home/work/caffe_ssd")

set (caffe_ssd_INCLUDE_DIRS ${caffe_ssd_PATH}/include)
set (caffe_ssd_BUILD_DIRS ${caffe_ssd_PATH}/.build_release/src)

set (cuda_INCLUDE_DIRS "/usr/local/cuda-8.0/include")
set (gflags_INCLUDE_DIRS "/usr/local/include/gflags")
set (gtest_INCLUDE_DIRS "/usr/local/include/gtest")
set (glog_INCLUDE_DIRS "/usr/include/glog")
set (boost_system_INCLUDE_DIRS "/usr/include/boost")

include_directories(${classifier_INCLUDE_DIRS} ${caffe_ssd_INCLUDE_DIRS} ${caffe_ssd_BUILD_DIRS} ${cuda_INCLUDE_DIRS} ${gflags_INCLUDE_DIRS}  ${gtest_INCLUDE_DIRS} ${glog_INCLUDE_DIRS} ${boost_system_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
add_executable (classifier ${classifier_SOURCES})

set (caffe_ssd_LIBS ${caffe_ssd_PATH}/build/lib/libcaffe.so)
set (gflags_LIBS "/usr/local/lib/libgflags.so")
set (gtest_LIBS "/usr/local/lib/libgtest.so")
set (glog_LIBS "/usr/lib/x86_64-linux-gnu/libglog.so")
set (boost_system_LIBS "/usr/lib/x86_64-linux-gnu/libboost_system.so")

target_link_libraries( classifier ${OpenCV_LIBS} ${caffe_ssd_LIBS} ${gflags_LIBS} ${gtest_LIBS} ${glog_LIBS} ${boost_system_LIBS}) 


