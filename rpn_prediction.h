#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <vector>
#include <string>
#include <iostream>
#include <caffe/caffe.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/videoio.hpp>
#include <iosfwd>
#include <memory>
#include <ctime>
#include <cassert>
#include <algorithm>
#include <ctype.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
using namespace std;
using namespace cv;
using namespace caffe;


void createDocList_R(vector<string> &doc_list, const string path0);

struct Bbox {
	int size_index;
	float confidence;
	Rect rect;
	bool deleted;

};
bool mycmp(struct Bbox b1, struct Bbox b2) {
	return b1.confidence > b2.confidence;
}

class RPN_detector {
public:
	RPN_detector(const string& model_file,
			const string& trained_file,
			const bool use_GPU,
			const int batch_size,
			const int device_id);

	vector<Blob<float>* > forward(vector<Mat> imgs);
	void nms(vector<struct Bbox>& p, float threshold);
	vector<struct Bbox> get_detection(vector<Mat> images, vector<Blob<float>* >& outputs, 
			int anchor_width[],int anchor_height[],  
			float rpn_thres[], float nms_thres, float enlarge_ratiow, float enlarge_ratioh);
	void get_input_size(int& batch_size, int& num_channels, int& height, int& width);

private:
	shared_ptr<Net<float> > net_;
	int batch_size_;
	int num_channels_;
	cv::Size input_geometry_;	
	bool useGPU_;
	
	int sliding_window_stride_;
};
