#include "ssd_detect.hpp"

#include <sstream>
#include <iostream>

const int isMobilenet = 1;

DEFINE_string(mean_file, "", "The mean file used to subtract from the input image.");


#if isMobilenet
DEFINE_string(mean_value, "127", "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
#else
DEFINE_string(mean_value, "104,117,123", "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
#endif

DEFINE_string(file_type, "image", "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "result/out.txt", "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.7, "Only store detections with score higher than the threshold.");

int main(int argc, char** argv) {
    std::cout << "input ssd_main ..."  << std::endl;
    if (argc < 4) {
        std::cout << " Usage: classifier model_file weights_file list_file\n" << std::endl;
        return -1;
    }

    const string& model_file = argv[1];
    const string& weights_file = argv[2];
    const string& mean_file = FLAGS_mean_file;
    const string& mean_value = FLAGS_mean_value;
    const string& file_type = FLAGS_file_type;
    const string& out_file = FLAGS_out_file;
    const float confidence_threshold = FLAGS_confidence_threshold;

    std::cout << "ssd_main model_file=" << model_file << " weights_file=" << weights_file << 
        " mean_file=" << mean_file << " mean_value=" << mean_value << " file_type=" << file_type << 
        " out_file=" << out_file << " confidence_threshold=" << confidence_threshold << std::endl;

    // Initialize the network.
    Detector detector;
    detector.Set(model_file, weights_file, mean_file, mean_value, isMobilenet);

    // Set the output mode.
    std::streambuf* buf = std::cout.rdbuf();
    std::ofstream outfile;
    if (!out_file.empty()) {
        outfile.open(out_file.c_str());
        if (outfile.good()) {
            buf = outfile.rdbuf();
        }
    }
    std::ostream out(buf);

    // Process image one by one.
    std::ifstream infile(argv[3]);
    std::string file;
        
    while (infile >> file) {
        string::size_type found = file.find("#");
        if (found != string::npos) {
            return 0;
        }
        
        if (file_type == "image") {
            cv::Mat img = cv::imread(file, -1);
            CHECK(!img.empty()) << "Unable to decode image from file=" << file;
            
            std::vector<vector<float> > detections = detector.Detect(img);
            std::vector<string> results;
            detector.Postprocess(file, img, confidence_threshold, detections, results);
            
            std::cout << "ssd_main file=" << file << " results.size=" << results.size() << std::endl;    
            for (int i = 0; i < results.size(); ++i) {        
                string result= results[i];
                out << result;
            }            
        } else if (file_type == "video") {
            cv::VideoCapture cap(file);
            if (!cap.isOpened()) {
                std::cout << "Failed to open video: " << file << std::endl;
            }
            
            cv::Mat img;
            int frame_count = 0;
            while (true) {
                bool success = cap.read(img);
                if (!success) {
                    std::cout << "Process " << frame_count << " frames from file=" << file << std::endl;
                    break;
                }
                CHECK(!img.empty()) << "Error when read frame";
                std::vector<vector<float> > detections = detector.Detect(img);
                
                std::vector<string> results;
                detector.Postprocess(file, img, confidence_threshold, detections, results);
                
                std::cout << "ssd_main file=" << file << " results.size=" << results.size() << std::endl;    
                for (int i = 0; i < results.size(); ++i) {        
                    string result= results[i];
                    out << result;
                }   
                ++frame_count;
            }
            
            if (cap.isOpened()) {
                cap.release();
            }
        } else {
            std::cout << "Unknown file_type: " << file_type << std::endl;
        }
    }

    std::cout << "output ssd_main" << std::endl;
    return 0;
}


