// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//

#ifndef _SSD_DETECT_HPP_
#define _SSD_DETECT_HPP_


#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "highgui.h"  
#include "cxcore.h"
#include "cv.h" 


using namespace caffe;  // NOLINT(build/namespaces)




class Detector {
    public:
        Detector();
        ~Detector();
        void Set(const string& model_file, const string& weights_file, const string& mean_file, 
            const string& mean_value, const int isMobilenet);
        std::vector<vector<float> > Detect(const cv::Mat& img);
        void Postprocess(std::string file, cv::Mat& img, const float confidence_threshold, 
            std::vector<vector<float> > detections, std::vector<string> &results);

    private:
        void SetMean(const string& mean_file, const string& mean_value);
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);   
        std::string flt2str(float f);    
        std::string int2str(int n) ;

    private:
        std::shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        float scale_;
        
        std::string CLASSES[21] = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
};
#endif


