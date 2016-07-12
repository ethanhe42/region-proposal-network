#include "rpn_prediction.h"
#define ANCHOR_NUMBER 8	

RPN_detector::RPN_detector(const string& model_file,
		const string& trained_file,
		const bool use_GPU,
		const int batch_size, const int devide_id) {
	if (use_GPU) {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(devide_id);
		useGPU_ = true;
	}
	else {
		Caffe::set_mode(Caffe::CPU);
		useGPU_ = false;
	}

	/* Set batchsize */
	batch_size_ = batch_size;

	/* Load the network. */
	cout<<"loading "<<model_file<<endl;
	net_.reset(new Net<float>(model_file, TEST));
	cout<<"loading "<<trained_file<<endl;
	net_->CopyTrainedLayersFrom(trained_file);
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
	<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	sliding_window_stride_ = 8;
}

// predict single frame forward function
vector<Blob<float>* > RPN_detector::forward(vector< cv::Mat > imgs) {

	Blob<float>* input_layer = net_->input_blobs()[0];  
	input_geometry_.height = imgs[0].rows;
	input_geometry_.width = imgs[0].cols;
	input_layer->Reshape(batch_size_, num_channels_,
			input_geometry_.height,
			input_geometry_.width);

	float* input_data = input_layer->mutable_cpu_data();
	int cnt = 0;
	for(int i = 0; i < imgs.size(); i++) {
		Mat sample;
		Mat img = imgs[i];

		if (img.channels() == 3 && num_channels_ == 1)
			cvtColor(img, sample, CV_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cvtColor(img, sample, CV_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cvtColor(img, sample, CV_RGBA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cvtColor(img, sample, CV_GRAY2BGR);
		else
			sample = img;

		if((sample.rows != input_geometry_.height) || (sample.cols != input_geometry_.width)) {
			resize(sample, sample, Size(input_geometry_.width, input_geometry_.height));
		}
		for(int k = 0; k < sample.channels(); k++) {
			for(int i = 0; i < sample.rows; i++) {
				for(int j = 0; j < sample.cols; j++) {
					input_data[cnt] = float(sample.at<uchar>(i,j*3+k))/255.0 - 0.5;
					cnt += 1;
				}
			}
		}		
	}

	/* Forward dimension change to all layers. */
	net_->Reshape();

	//	struct timeval start;
	//	gettimeofday(&start, NULL);

	net_->ForwardPrefilled();
	if(useGPU_) {
		cudaDeviceSynchronize();
	}

	//	struct timeval end;
	//	gettimeofday(&end, NULL);
	//	double t = 1000.0 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000.0;
	//	cout<<"Forward time: "<<t<<" ms"<<endl;


	/* Copy the output layer to a std::vector */
	vector<Blob<float>* > outputs;
	for(int i = 0; i < net_->num_outputs(); i++) {
		Blob<float>* output_layer = net_->output_blobs()[i];
		outputs.push_back(output_layer);
	}
	return outputs;
}



void RPN_detector::nms(vector<struct Bbox>& p, float threshold) {

	sort(p.begin(), p.end(), mycmp);
	int cnt = 0;
	for(int i = 0; i < p.size(); i++) {

		if(p[i].deleted) continue;
		cnt += 1;
		for(int j = i+1; j < p.size(); j++) {

			if(!p[j].deleted) {
				cv::Rect intersect = p[i].rect & p[j].rect;
				float iou = intersect.area() * 1.0/p[j].rect.area(); 
				if (iou > threshold) {
					p[j].deleted = true;
				}
			}
		}
	}
}

void RPN_detector::get_input_size(int& batch_size, int& num_channels, int& height, int& width)
{
	batch_size = batch_size_;
	num_channels = num_channels_;
	height = input_geometry_.height;
	width = input_geometry_.width;
}

vector<struct Bbox> RPN_detector::get_detection(vector<Mat> images, vector<Blob<float>* >& outputs, 
		int sliding_window_width[], int sliding_window_height[],
		float rpn_thres[], float nms_thres, float enlarge_ratiow, float enlarge_ratioh) {

	//	struct timeval start;
	//	gettimeofday(&start, NULL);

	Blob<float>* cls = outputs[0];
	Blob<float>* reg = outputs[1];

	cls->Reshape(cls->num(), cls->channels(), cls->height(), cls->width());
	reg->Reshape(reg->num(), reg->channels(), reg->height(), reg->width());

	assert(cls->num() == reg->num());

	assert(cls->channels() == ANCHOR_NUMBER * 2);
	assert(reg->channels() == ANCHOR_NUMBER * 4);


	assert(cls->height() == reg->height());
	assert(cls->width() == reg->width());

	assert(images.size() == 1);

	vector<struct Bbox> vbbox;
	const float* cls_cpu = cls->cpu_data();
	const float* reg_cpu = reg->cpu_data();	
	int img_height = images[0].rows;
	int img_width = images[0].cols;
	float w,h;
	int skip = cls->height() * cls->width();
	float log_thres[ANCHOR_NUMBER];
	for (int i=0; i<ANCHOR_NUMBER; i++)
		log_thres[i]=log(rpn_thres[i]/(1.0-rpn_thres[i]));
	float rect[4];
	for(int i = 0; i < cls->num(); i++) 
	{  
		for(int j=0;j<ANCHOR_NUMBER;j++)
		{
			h = sliding_window_height[j];
			w = sliding_window_width[j]; 
			for (int y_index=0; y_index<int(img_height/sliding_window_stride_);y_index++)
			{
				int y = y_index*sliding_window_stride_ + sliding_window_stride_/2-1 - h/2;
				for (int x_index=0; x_index<int(img_width/sliding_window_stride_); x_index++)
				{
					int x = x_index*sliding_window_stride_ + sliding_window_stride_/2-1 - w/2;

					float x0=cls_cpu[2*j*skip + y_index*cls->width() + x_index];
					float x1=cls_cpu[(2*j+1)*skip+y_index*cls->width()+x_index];					
					if(x1 - x0 > log_thres[j])
					{
						rect[2]=exp(reg_cpu[(j*4+2)*skip+y_index*reg->width()+x_index])*w;
						rect[3]=exp(reg_cpu[(j*4+3)*skip+y_index*reg->width()+x_index])*h;

						rect[0]=reg_cpu[j*4*skip+y_index*reg->width()+x_index];   
						rect[1]=reg_cpu[(j*4+1)*skip+y_index*reg->width()+x_index];

						rect[0]=rect[0]*w+w/2-rect[2]/2+x;
						rect[1]=rect[1]*h+h/2-rect[3]/2+y; 

						struct Bbox bbox;
						bbox.confidence = 1.0/(1.0 + exp(x0-x1));;
						bbox.size_index = j;
						bbox.rect = Rect(rect[0], rect[1], rect[2], rect[3]);							
						bbox.rect &= Rect(0,0, img_width, img_height);
						bbox.deleted = false;
						vbbox.push_back(bbox);
					}
				}
			}
		} 
	}

	//	struct timeval end;
	//	gettimeofday(&end, NULL);
	//	double t = 1000.0 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000.0;
	//	cout<<"bb collection: "<<t<<" ms"<<endl;

	if (vbbox.size()!=0)
		nms(vbbox, nms_thres);

	vector<struct Bbox> final_vbbox;

	for(int i = 0; i < vbbox.size(); i++) {
		if(!vbbox[i].deleted) {
			struct Bbox box = vbbox[i];
			float x = box.rect.x * enlarge_ratiow;
			float y = box.rect.y * enlarge_ratioh;
			float w = box.rect.width *enlarge_ratiow;
			float h = box.rect.height* enlarge_ratioh;
			box.rect.x = x;
			box.rect.y = y;
			box.rect.width = w;
			box.rect.height = h;
			final_vbbox.push_back(box);
		}
	}
	return final_vbbox;
}

//reading  images from one file
void createDocList(vector<string> &doc_list, const string path){
	DIR *dpdf;
	struct dirent *epdf;
	dpdf = opendir(path.c_str());

	if (dpdf != NULL){
		epdf = readdir(dpdf);
		while (epdf){
			string name=string(epdf->d_name);
			epdf = readdir(dpdf);
			// for jpg format frame			
			if (name[name.length()-1]!='g')
				continue;
			doc_list.push_back(path+string(name));

		}
		closedir(dpdf);
	}else{
		cout<<"the path is empty"<<endl;
	}   

}

void createDocList_R(vector<string> &doc_list, const string path0){
	cout<<"NEW DIR: "<<path0<<endl;
	string path(path0);
	if(path[path.length()-1]!='/')
		path.append("/");
	DIR *pDir;
	struct dirent *ent;
	pDir = opendir(path.c_str());

	if (pDir != NULL){
		while ( (ent = readdir(pDir))!= NULL){
			string name = string(ent->d_name);
			if(ent->d_type == 4){
				if(name[0] == '.')
					continue;
				createDocList_R(doc_list, path + name);
			}
			else{
				// for jpg format frame			
				if (name[name.length()-1]!='g')
					continue;
				doc_list.push_back(path + name);
			}
		}
		closedir(pDir);
	}else{
		cout<<"the path is empty"<<endl;
	}   

}




void DetectionForVideo(string& model_file, string& trained_file, string& video, string& save_dir,bool x_folder)
{	
	bool x_show_prediction = true;
	bool x_save = true;
	if(save_dir.compare("") == 0)
		x_save = false;

	cout<<"opening "<<video<<"..."<<endl;
	vector<string> imagelist;
	vector<string>::iterator iter;
	cv::VideoCapture capture;

	if (x_folder){		
		createDocList_R(imagelist,video);		
		sort(imagelist.begin(),imagelist.end());
		iter = imagelist.begin();
	}
	else{		
		if(video.compare("/dev/video0") == 0)
			capture.open(0);
		else if(video.compare("/dev/video1") == 0){
			capture.open(1);
			capture.set(3, 1920);
			capture.set(4, 1080);
		}
		else
			capture.open(video);

		if(!capture.isOpened())
		{
			cout<<"Cannot open "<<video<<endl;
			return;
		}		
	}

	cout<<"loading model..."<<endl;	
	RPN_detector rpn_det(model_file, trained_file, true, 1,0);

	int batch_size = 0, num_channels = 0, resize_width = 0, resize_height = 0;
	rpn_det.get_input_size(batch_size, num_channels, resize_height, resize_width);
	cout<<"input size: ("<<resize_height<<", "<<resize_width<<")"<<endl;

#if ANCHOR_NUMBER == 9	
	int sliding_window_width[] =  {10,10,10, 30,30,30, 50,50,50};
	int sliding_window_height[] = {10,15,20, 30,45,60, 50,75,100}; 
	float rpn_thres[] = {0.98,0.98,0.98, 0.98,0.98,0.98, 0.98,0.98,0.98};
	float min_thres = 0.9;
#elif ANCHOR_NUMBER == 8
	int sliding_window_width[] =  {20,20, 30,30, 40,40, 50,50};
	int sliding_window_height[] = {30,40, 45,60, 60,80, 75,100}; 
	float rpn_thres[] = {0.95,0.95, 0.95,0.95, 0.95,0.95, 0.95,0.95};
	float min_thres = 0.0;
#endif
	float nms_thres = 0.4;

	cv::namedWindow("Cam",CV_WINDOW_NORMAL);
	bool stop = false;

	int frame_count = 1;
	struct timeval start, end;

	double sum_time_forward = 0.0, aver_time_forward = 0.0;
	double sum_time_nms = 0.0, aver_time_nms = 0.0;
	double sum_time_show = 0.0, aver_time_show = 0.0;

	int start_frame_no = 0;
	int jump_frames = 0;
	int jump_count = 0;

	while (!stop) {		
		cv::Mat frame; 

		if (x_folder){
			if (iter==imagelist.end())
				break;
			frame = imread(*iter,1);
		}
		else
			if (!capture.read(frame))
				break;

		if (frame.empty()) {
			cout << "Wrong Image" << endl;
			continue;
		}
		if (frame_count < start_frame_no)
		{
			if (start_frame_no % 100 == 0)
				cout<<start_frame_no<<" frames left"<<endl;
			start_frame_no--;
			continue;
		}
		if (jump_count < jump_frames)
		{
			jump_count++;
			continue;
		}
		else
			jump_count = 0;


		int output_width = frame.cols;
		int output_height = frame.rows;
		float enlarge_ratioh = output_height*1.0/resize_height  ,enlarge_ratiow=output_width*1.0/resize_width;
		if(frame_count==1)
			cout<<"output size: ("<<output_height<<", "<<output_width<<")"<<endl;

		gettimeofday(&start, NULL);			

		Mat img = frame.clone();
		Mat norm_img;
		resize(img, norm_img, Size(resize_width, resize_height));
		cvtColor(norm_img, norm_img, CV_RGB2BGR);
		vector<Mat> images;
		images.push_back(norm_img);
		vector<Blob<float>* > outputs = rpn_det.forward(images);

		gettimeofday(&end, NULL);
		sum_time_forward += double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1.0e6;
		aver_time_forward = sum_time_forward / double(frame_count);

		gettimeofday(&start, NULL);

		vector<struct Bbox> result =rpn_det.get_detection(images, outputs,
				sliding_window_width,sliding_window_height, 
				rpn_thres, nms_thres, enlarge_ratiow, enlarge_ratioh);	

		gettimeofday(&end, NULL);
		sum_time_nms += double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1.0e6;	
		aver_time_nms = sum_time_nms / double(frame_count);

		gettimeofday(&start, NULL);

		if(x_show_prediction)
		{
			char str_info[100];		
			for(int bbox_id = 0; bbox_id < result.size(); bbox_id ++) {
				cv::Mat frame_tag = frame.clone();
				rectangle(frame_tag, result[bbox_id].rect, Scalar(0,255,0),2);
//				sprintf(str_info,"%.3f,[%d,%d],(%d,%d)",result[bbox_id].confidence, 
//						sliding_window_width[result[bbox_id].size_index],
//						sliding_window_height[result[bbox_id].size_index],
//						result[bbox_id].rect.width,
//						result[bbox_id].rect.height);
				sprintf(str_info,"%.3f",result[bbox_id].confidence);
				string prob_info(str_info); 
				putText(frame_tag, prob_info, Point(result[bbox_id].rect.x, result[bbox_id].rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.3,  Scalar(0,0,255));
				cv::addWeighted(frame, 1-(result[bbox_id].confidence-min_thres)/(1.0-min_thres), 
						frame_tag, (result[bbox_id].confidence-min_thres)/(1.0-min_thres), 0, frame);
			}
		}

		cv::imshow("Cam",frame);

		gettimeofday(&end, NULL);
		sum_time_show += double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1.0e6;
		aver_time_show = sum_time_show / double(frame_count);

		if (frame_count % 1 == 0)
			cout<<"Frame."<<frame_count<<" "
			<<"forward = "<<aver_time_forward*1000.0<<" ms, "
			<<"nms = "<<aver_time_nms*1000.0<<" ms, "
			<<"show = "<<aver_time_show*1000.0<<" ms"<<endl;

		if (x_save)
		{
			char fn[512];
			sprintf(fn,"%s/img-%06d.jpg",save_dir.c_str(),frame_count);
			cv::imwrite(fn, frame);
		}


		frame_count++;
		if ( (cv::waitKey(x_folder ? 10 : 10) & 0xff) == 27)//Esc
			stop = true;
		
		if(x_folder)
			iter++;
		//below from hus
		if (x_folder){
			string picPath=*iter;
			for(int bbox_id = 0; bbox_id < result.size(); bbox_id ++) {
				double x=result[bbox_id].rect.x;
				double y=result[bbox_id].rect.y;
				double width=result[bbox_id].rect.width;
				double height=result[bbox_id].rect.height;
				double score=result[bbox_id].confidence*100;
			}
			std::cout<<picPath<<" "<<x<<" "<<y<<" "<<width<<" "<<height<<"_"<<score<<std::endl;
	}
}

void HUB_DetectionForVideo(string& model_file, string& trained_file, string& video, string& save_dir,bool x_folder)
{	
	bool x_show_prediction = true;
	bool x_save = true;
	if(save_dir.compare("") == 0)
		x_save = false;

	cout<<"opening "<<video<<"..."<<endl;
	vector<string> imagelist;
	vector<string>::iterator iter;
	cv::VideoCapture capture;

	if (x_folder){		
		createDocList_R(imagelist,video);		
		sort(imagelist.begin(),imagelist.end());
		iter = imagelist.begin();
	}
	else{		
		if(video.compare("/dev/video0") == 0)
			capture.open(0);
		else if(video.compare("/dev/video1") == 0){
			capture.open(1);
			capture.set(3, 1920);
			capture.set(4, 1080);
		}
		else
			capture.open(video);

		if(!capture.isOpened())
		{
			cout<<"Cannot open "<<video<<endl;
			return;
		}		
	}

	cout<<"loading model..."<<endl;	
	RPN_detector rpn_det(model_file, trained_file, true, 1,0);

	int batch_size = 0, num_channels = 0, resize_width = 0, resize_height = 0;
	rpn_det.get_input_size(batch_size, num_channels, resize_height, resize_width);
	cout<<"input size: ("<<resize_height<<", "<<resize_width<<")"<<endl;

#if ANCHOR_NUMBER == 8
	int sliding_window_width[] =  {10,10, 30,30, 50,50,  70,70};
	int sliding_window_height[] = {10,20, 30,60, 50,100, 70,140}; 
	float rpn_thres[] = {0.95,0.95, 0.95,0.95, 0.95,0.95, 0.95,0.95};
	float min_thres = 0.8;
#endif
	float nms_thres = 0.4;

	cv::namedWindow("Cam",CV_WINDOW_NORMAL);
	bool stop = false;

	int frame_count = 1;
	struct timeval start, end;

	double sum_time_forward = 0.0, aver_time_forward = 0.0;
	double sum_time_nms = 0.0, aver_time_nms = 0.0;
	double sum_time_show = 0.0, aver_time_show = 0.0;

	int start_frame_no = 0;
	int jump_frames = 0;
	int jump_count = 0;

	while (!stop) {		
		cv::Mat frame; 

		if (x_folder){
			if (iter==imagelist.end())
				break;
			frame = imread(*iter,1);
		}
		else
			if (!capture.read(frame))
				break;

		if (frame.empty()) {
			cout << "Wrong Image" << endl;
			continue;
		}
		if (frame_count < start_frame_no)
		{
			if (start_frame_no % 100 == 0)
				cout<<start_frame_no<<" frames left"<<endl;
			start_frame_no--;
			continue;
		}
		if (jump_count < jump_frames)
		{
			jump_count++;
			continue;
		}
		else
			jump_count = 0;


		int output_width = frame.cols;
		int output_height = frame.rows;
		float enlarge_ratioh = output_height*1.0/resize_height  ,enlarge_ratiow=output_width*1.0/resize_width;
		if(frame_count==1)
			cout<<"output size: ("<<output_height<<", "<<output_width<<")"<<endl;

		gettimeofday(&start, NULL);			

		Mat img = frame.clone();
		Mat norm_img;
		resize(img, norm_img, Size(resize_width, resize_height));
		cvtColor(norm_img, norm_img, CV_RGB2BGR);
		vector<Mat> images;
		images.push_back(norm_img);
		vector<Blob<float>* > outputs = rpn_det.forward(images);


		vector<Blob<float>* > outputs_head;
		outputs_head.push_back(outputs[0]);
		outputs_head.push_back(outputs[1]);

		vector<Blob<float>* > outputs_upper;
		outputs_upper.push_back(outputs[2]);
		outputs_upper.push_back(outputs[3]);

		vector<Blob<float>* > outputs_body;
		outputs_body.push_back(outputs[4]);
		outputs_body.push_back(outputs[5]);

		gettimeofday(&end, NULL);
		sum_time_forward += double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1.0e6;
		aver_time_forward = sum_time_forward / double(frame_count);

		gettimeofday(&start, NULL);

		vector<struct Bbox> result_head = rpn_det.get_detection(images, outputs_head, sliding_window_width,sliding_window_height, rpn_thres, nms_thres, enlarge_ratiow, enlarge_ratioh);	
		vector<struct Bbox> result_upper =rpn_det.get_detection(images, outputs_upper,sliding_window_width,sliding_window_height, rpn_thres, nms_thres, enlarge_ratiow, enlarge_ratioh);
		vector<struct Bbox> result_body = rpn_det.get_detection(images, outputs_body, sliding_window_width,sliding_window_height, rpn_thres, nms_thres, enlarge_ratiow, enlarge_ratioh);

		gettimeofday(&end, NULL);
		sum_time_nms += double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1.0e6;	
		aver_time_nms = sum_time_nms / double(frame_count);

		gettimeofday(&start, NULL);

		if(x_show_prediction)
		{		
			char str_info[100];		
			for(int bbox_id = 0; bbox_id < result_head.size(); bbox_id ++) {
				cv::Mat frame_tag = frame.clone();
				rectangle(frame_tag, result_head[bbox_id].rect, Scalar(0,0,255),2);
				sprintf(str_info,"%.3f,[%d,%d],(%d,%d)",result_head[bbox_id].confidence, 
						sliding_window_width[result_head[bbox_id].size_index],
						sliding_window_height[result_head[bbox_id].size_index],
						result_head[bbox_id].rect.width,
						result_head[bbox_id].rect.height);
				//sprintf(str_info,"%.3f",result_head[bbox_id].confidence);
				string prob_info(str_info); 
				putText(frame_tag, prob_info, Point(result_head[bbox_id].rect.x, result_head[bbox_id].rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.3,  Scalar(0,0,255));
				cv::addWeighted(frame, 1-(result_head[bbox_id].confidence-min_thres)/(1.0-min_thres), 
						frame_tag, (result_head[bbox_id].confidence-min_thres)/(1.0-min_thres), 0, frame);
			}
			for(int bbox_id = 0; bbox_id < result_upper.size(); bbox_id ++) {
				cv::Mat frame_tag = frame.clone();
				rectangle(frame_tag, result_upper[bbox_id].rect, Scalar(0,255,0),2);
				sprintf(str_info,"%.3f,[%d,%d],(%d,%d)",result_upper[bbox_id].confidence, 
						sliding_window_width[result_upper[bbox_id].size_index],
						sliding_window_height[result_upper[bbox_id].size_index],
						result_upper[bbox_id].rect.width,
						result_upper[bbox_id].rect.height);
				//sprintf(str_info,"%.3f",result_upper[bbox_id].confidence);
				string prob_info(str_info); 
				putText(frame_tag, prob_info, Point(result_upper[bbox_id].rect.x, result_upper[bbox_id].rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.3,  Scalar(0,255,0));
				cv::addWeighted(frame, 1-(result_upper[bbox_id].confidence-min_thres)/(1.0-min_thres), 
						frame_tag, (result_upper[bbox_id].confidence-min_thres)/(1.0-min_thres), 0, frame);
			}
			for(int bbox_id = 0; bbox_id < result_body.size(); bbox_id ++) {
				cv::Mat frame_tag = frame.clone();
				rectangle(frame_tag, result_body[bbox_id].rect, Scalar(255,0,0),2);
				sprintf(str_info,"%.3f,[%d,%d],(%d,%d)",result_body[bbox_id].confidence, 
						sliding_window_width[result_body[bbox_id].size_index],
						sliding_window_height[result_body[bbox_id].size_index],
						result_body[bbox_id].rect.width,
						result_body[bbox_id].rect.height);
				//sprintf(str_info,"%.3f",result_body[bbox_id].confidence);
				string prob_info(str_info); 
				putText(frame_tag, prob_info, Point(result_body[bbox_id].rect.x, result_body[bbox_id].rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.3,  Scalar(255,0,0));
				cv::addWeighted(frame, 1-(result_body[bbox_id].confidence-min_thres)/(1.0-min_thres), 
						frame_tag, (result_body[bbox_id].confidence-min_thres)/(1.0-min_thres), 0, frame);
			}
		}

		cv::imshow("Cam",frame);

		gettimeofday(&end, NULL);
		sum_time_show += double(end.tv_sec - start.tv_sec) + double(end.tv_usec - start.tv_usec) / 1.0e6;
		aver_time_show = sum_time_show / double(frame_count);

		if (frame_count % 1 == 0)
			cout<<"Frame."<<frame_count<<" "
			<<"forward = "<<aver_time_forward*1000.0<<" ms, "
			<<"nms = "<<aver_time_nms*1000.0<<" ms, "
			<<"show = "<<aver_time_show*1000.0<<" ms"<<endl;

		if (x_save)
		{
			char fn[512];
			sprintf(fn,"%s/img-%06d.jpg",save_dir.c_str(),frame_count);
			cv::imwrite(fn, frame);
		}


		frame_count++;
		if ( (cv::waitKey(x_folder ? 10 : 10) & 0xff) == 27)//Esc
			stop = true;

		if(x_folder)
			iter++;
	}
}

int main(int argc, char** argv )
{
	//	string model_file   = "./models/fasterRCNN/rpn_16layer_small_da_ft/rpn_16layer_small_da_ft_deploy.prototxt";
	//	string trained_file = "./models/fasterRCNN/rpn_16layer_small_da_ft/rpn_16layer_small_da_ft_iter_400000.caffemodel";

		string model_file   = "/home/huyangyang/caffe-master/models/fasterRCNN/rpn_drn/rpn_drn_deploy.prototxt";
		string trained_file = "/home/huyangyang/caffe-master/models/fasterRCNN/rpn_drn/rpn_drn_iter_750000.caffemodel";

//	string model_file   = "./models/fasterRCNN/rpn_hub/rpn_hub_deploy.prototxt";
//	string trained_file = "./models/fasterRCNN/rpn_hub/rpn_hub_iter_310000.caffemodel";

//	string model_file   = "./models/fasterRCNN/rpn_thin16/rpn_thin16_deploy.prototxt";
//	string trained_file = "./models/fasterRCNN/rpn_thin16/rpn_thin16_iter_130000.caffemodel";

	if(argc<2)
	{
		cout<<"need video file param"<<endl;
		return 0;
	}
	string video(argv[1]);
	string save_dir = "";
	if(argc==3)
		save_dir.append(argv[2]);

	google::InitGoogleLogging(argv[0]);

	bool x_folder = true;
	DetectionForVideo(model_file, trained_file, video, save_dir, x_folder);
}
