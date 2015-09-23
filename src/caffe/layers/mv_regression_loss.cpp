#include "caffe/custom_layers.hpp"
#include <opencv2/opencv.hpp>

#include <atomic>
#include <mutex>
#include <iostream>
#include <stdarg.h>
#include <jvl/jvl.hpp>
#include <jvl/blob_io.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace jvl;

namespace caffe
{ 
  // available if useful
  template <typename Dtype>
  void MVRegressionLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
					     const vector<Blob<Dtype>*>& top)
  {
    if(top.size() > 0)      
      top[0]->Reshape(1, 1, 1, 1);        
  }
  
  template <typename Dtype>
  MVRegressionLossLayer<Dtype>::MVRegressionLossLayer(const LayerParameter& param) :
    Layer<Dtype>(param)
  {
  }

  constexpr double sf = 8;
  
  template<typename Dtype>
  static std::vector<Point2d> drawKeypoints(Blob<Dtype>* blob,int nIter,Mat&vis,Scalar color)
  {
    // check geometry
    CHECK_LE(nIter-1,blob->num());
    CHECK_EQ(blob->channels(),2*jvl::active_keypoints.size());
    CHECK_EQ(blob->height(),1);
    CHECK_EQ(blob->width(),1);
    
    const Dtype* label_data = blob->cpu_data() + blob->offset(nIter);
    Mat label_mat(2*jvl::active_keypoints.size(),1,DataType<Dtype>::type);
    caffe_copy(blob->channels()*blob->width()*blob->height(),label_data,label_mat.ptr<Dtype>());

    std::vector<Point2d> points;
    for(int iter = 0; iter < 2*jvl::active_keypoints.size(); iter +=2 )
    {
      Dtype u = sf*label_mat.at<Dtype>(iter);
      Dtype v = sf*label_mat.at<Dtype>(iter+1);
      cv::putText(vis,jvl::printfpp("%d",iter),Point2d(u,v),FONT_HERSHEY_SIMPLEX,.5,color);
      points.push_back(Point2d(u,v));
    }
    return points;
  }  
  
  template <typename Dtype>
  void MVRegressionLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
						 const vector<Blob<Dtype>*>& top)
  {
    double average_mean = 0;
    //for(int nIter = 0; nIter < bottom.at(0)->num(); ++nIter)
    int nIter = rand() % bottom.at(0)->num();
    {
      // extract the image
      Blob<Dtype> * image = bottom.at(0);
      const Dtype* image_data = image->cpu_data() + image->offset(nIter);
      CHECK_EQ(image->channels(), 1);
      Mat image_mat(image->height(),image->width(),DataType<Dtype>::type);
      caffe_copy(image->count()/image->num(),image_data,image_mat.ptr<Dtype>());
      cv::resize(image_mat,image_mat,Size(),sf,sf,jvl::DEPTH_INTER_STRATEGY);
      Mat viz = jvl::eq(image_mat);
      
      // extract the gt labels
      vector<Point2d> pts_gt = drawKeypoints(bottom.at(1),nIter,viz,Scalar(0,255,0));
      // extract
      vector<Point2d> pts_dt = drawKeypoints(bottom.at(2),nIter,viz,Scalar(255,0,0));
      
      // compute the euclidean distance.
      double dist = 0;
      double max_dist = -inf;
      vector<double> dists;
      for(int iter = 0; iter < pts_gt.size(); ++iter)
      {
	double u_gt = pts_gt.at(iter).x;
	double v_gt = pts_gt.at(iter).y;
	double u_dt = pts_dt.at(iter).x;
	double v_dt = pts_dt.at(iter).y;
	double d_here = (jvl::metric_size/static_cast<double>(image->width()))*
	  std::sqrt(std::pow(u_gt - u_dt,2) + std::pow(v_gt - v_dt,2));
	dist += d_here;
	max_dist = std::max<double>(max_dist,d_here);
	dists.push_back(d_here);
      }
      double mean_dist = dist/pts_gt.size();
      average_mean += mean_dist;
      
      // write the result
      static atomic<int> out_count(0);
      //cout << "\033[1;31m !MVRegressionLossLayer!  \033[0m" << endl;
      cv::imwrite(jvl::printfpp("out/regression_result_%d_%d.png",out_count++,nIter),viz);
      
      {
	static mutex m; lock_guard<mutex> l(m);
	static ofstream mean_errors("out/mean_errors.txt");
	mean_errors << mean_dist << endl;
	static ofstream max_errors("out/max_errors.txt");
	max_errors << max_dist << endl;
	static ofstream all_errors("out/all_errors.txt");
	for(auto && err : dists)
	  all_errors << err << " ";
	all_errors << endl;
      }
    }
    
    if(top.size() > 0)
      *top.at(0)->mutable_cpu_data() = average_mean/bottom.at(0)->num();    
  }

  template <typename Dtype>
  void MVRegressionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			    const vector<bool>& propagate_down,
			    const vector<Blob<Dtype>*>& bottom)
  {
    // doesn't matter for dataa layers
  }
  
  template <typename Dtype>
  void MVRegressionLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		       const vector<Blob<Dtype>*>& top)
  {
    // only matters if we have a bttom apparently
  }
  
  INSTANTIATE_CLASS(MVRegressionLossLayer);
  REGISTER_LAYER_CLASS(MVRegressionLoss);
}
