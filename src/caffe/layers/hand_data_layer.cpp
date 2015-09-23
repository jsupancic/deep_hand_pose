#include "caffe/custom_layers.hpp"
#include <opencv2/opencv.hpp>

#include <mutex>
#include <iostream>
#include <stdarg.h>
#include <thread>
#include <jvl/jvl.hpp>
#include "jvl/blob_io.hpp"
#include "tbb/concurrent_queue.h"
#include <memory>

using namespace std;
using namespace cv;
using namespace jvl;

//

namespace caffe
{
  template<typename Dtype>
  struct TrainingHandData
  {
    std::shared_ptr<Blob<Dtype> > data;
    std::shared_ptr<Blob<Dtype> > label;
    std::shared_ptr<Blob<Dtype> > index;
    std::shared_ptr<Blob<Dtype> > heatmap;
  };
  
  template<typename Dtype>
  class HandDataLayerPrivate
  {
  public:
    vector<std::thread> t;
    int batch_size;
    string src_name;
    
    tbb::concurrent_bounded_queue<TrainingHandData<Dtype>> output_queue;
    void producer();
    
    HandDataLayerPrivate(int batch_size, const string&src_name) :
      batch_size(batch_size),
      src_name(src_name)
    {
      output_queue.set_capacity(4*4);
      for(int iter = 0; iter < 4; ++iter)
      {
	t.push_back(std::move(std::thread([this]()
	{	  
	  while(true)
	  {
	    try
	    {
	      this->producer();
	    }
	    catch(tbb::user_abort&abrt)
	    {
	      return;
	    }
	  }
	})));
      }
    }

    ~HandDataLayerPrivate()
    {
      output_queue.abort();
    }
  };
  
  // available if useful
  template <typename Dtype>
  void HandDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top)
  {   
    // image
    cout << "batch_size = " << batch_size << endl;
    int channels = 1;
    int height = image_res;
    int width  = image_res;
    top[0]->Reshape(
      batch_size, channels, height,width);
    LOG(INFO) << "output data size: " << top[0]->num() << ","
	      << top[0]->channels() << "," << top[0]->height() << ","
	      << top[0]->width();
    // labels
    int hmHeight = 1;
    int hmWidth  = 1;
    int hmChans  = jvl::active_keypoints.size()*2;
    top[1]->Reshape(batch_size, hmChans, hmHeight, hmWidth);

    // index
    if(top.size() >= 3)
      top[2]->Reshape(batch_size,1,1,1);

    if(top.size() >= 4)
      top[3]->Reshape(batch_size,jvl::active_keypoints.size(),jvl::heat_map_res,jvl::heat_map_res);
  }

  template <typename Dtype>
  HandDataLayer<Dtype>::~HandDataLayer()
  {
    delete private_;
  }
  
  template <typename Dtype>
  HandDataLayer<Dtype>::HandDataLayer(const LayerParameter& param) :
    Layer<Dtype>(param),
    transform_param_(param.transform_param())
  {
    batch_size = param.data_param().batch_size();
    src_name = param.data_param().source();
    private_ = new HandDataLayerPrivate<Dtype>(batch_size,src_name);
    cout << "batch_size = " << batch_size << endl;
    cout << "\033[1;31m !HandDataLayer::HandDataLayer!  \033[0m\n" << endl;
  }

  template<typename Dtype>
  static Mat update_mean(const Mat&m)
  {
    static mutex mx; lock_guard<mutex> l(mx);
    if(m.type() != DataType<Dtype>::type)
      throw std::runtime_error("type mismatch");
    
    Mat x(m.rows,m.cols,DataType<double>::type);
    for(int yIter = 0; yIter < m.rows; ++yIter)
      for(int xIter = 0; xIter < m.cols; ++xIter)
	x.at<double>(yIter,xIter) = m.at<Dtype>(yIter,xIter);
    //m.clone().convertTo(x,DataType<double>::type);       
    //if(x.type() != DataType<double>::type)
    //throw std::runtime_error("type mismatch");
    static Mat mean, stdv;
    bool re_compute_mean = true;
    if(re_compute_mean) 
    {
      static long count = 0;
      static Mat moment1;
      static Mat moment2;
      if(count == 0)
      {
	count = 1;
	moment1 = x.clone();
	moment2 = x.mul(x);
      }
      else
      {
	count ++;
	if(moment1.type() != DataType<double>::type)
	  throw std::runtime_error("type mismatch");
	if(x.type() != DataType<double>::type)
	  throw std::runtime_error("type mismatch");
	moment1 += x;
	moment2 += x.mul(x);
      }
      mean = moment1/count;
      Mat var  = moment2/count - mean.mul(mean);
      cv::sqrt(cv::abs(var),stdv);
      stdv += .005;
      if(count % 1024 == 0)
      {
	jvl::writeMatToFile(mean, "examples/NYU_HANDS/mu.bin");
	jvl::writeMatToFile(stdv, "examples/NYU_HANDS/stdv.bin");
      }
      mean.convertTo(mean,DataType<Dtype>::type);
      stdv.convertTo(stdv,DataType<Dtype>::type);
    }
    else
    {
      if(mean.empty())
      {
	jvl::readFileToMat(mean,"examples/NYU_HANDS/mu_save.bin");
	mean.convertTo(mean,DataType<Dtype>::type);
      }
      if(stdv.empty())
      {
	jvl::readFileToMat(stdv,"examples/NYU_HANDS/stdv_save.bin");
	stdv.convertTo(stdv,DataType<Dtype>::type);      
      }           
    }
    
    //
    Mat normalized = (m - mean);
    cv::divide(normalized,stdv,normalized);
    //cout << normalized << endl;
    return normalized;
  }

  template<typename Dtype>
  void augment(Mat&Z,std::vector<cv::Vec3d>&uvd)
  {
    // rot
    int angle_deg = (rand() % 120) - 60;
    Mat at;
    Z = imrotate(Z,deg2rad(angle_deg),at);
    for(auto && v : uvd)
      v = vec_affine(v,at);
    // scale
    double sf = sample_in_range(.9,1.1);
    Z = imscale(Z,sf,at);
    for(auto && v : uvd)
    v = vec_affine(v,at);    
    // translate
    int t_range = (1.5/10.)*image_res;
    int tx = rand() % t_range - t_range/2;
    int ty = rand() % t_range - t_range/2;
    for(auto && v : uvd)
    {
      v[0] -= tx;
      v[1] -= ty;
    }
    Z = imroi(Z,Rect(Point(tx,ty),Z.size()));
  }

  // Jarret 2009
  //  sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8    
  template<typename Dtype>
  void LocalContrastNormalization(Mat&X)
  {
    //
    Size kSize(5,5);
    
    // mean subtractive normalization
    Mat XB; cv::GaussianBlur(X,XB,kSize,0,0);
    Mat V = X.clone() - XB;

    // compute normalizers
    Mat V2 = V.mul(V);
    cv::GaussianBlur(V2,V2,kSize,0,0);
    Mat sigma; cv::sqrt(V2,sigma);
    // mean
    Scalar mean = cv::mean(sigma);
    // normalizer
    Mat N = cv::max(sigma,mean[0]);
    
    // divisive normalization
    X = V/N;
  }
  
  template <typename Dtype>
  void HandDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				       const vector<Blob<Dtype>*>& top)
  {
    TrainingHandData<Dtype> cur_render;
    private_->output_queue.pop(cur_render);

    if(top.size() > 0)
    top.at(0)->CopyFrom(*cur_render.data);
    if(top.size() > 1)
    {
      // has the label been corrupted here?
      top.at(1)->CopyFrom(*cur_render.label);
    }
    if(top.size() > 2)
      top.at(2)->CopyFrom(*cur_render.index);
    if(top.size() > 3)
      top.at(3)->CopyFrom(*cur_render.heatmap);
  }

  template <typename Dtype>
  void HandDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			    const vector<bool>& propagate_down,
			    const vector<Blob<Dtype>*>& bottom)
  {
    // doesn't matter for dataa layers
  }

  static string accum_uvd_filename = "out/accum_uvd.dat";
  
  template <typename Dtype>
  void HandDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		       const vector<Blob<Dtype>*>& top)
  {
    // only matters if we have a bttom apparently
  }

  static void do_PCA()
  {
    static mutex m; lock_guard<mutex> l(m);
    static bool PCA_done = false;
    if(PCA_done)
      return;

    // check the file
    ifstream ifs(accum_uvd_filename);
    if(ifs)
      ifs.close();
    else
      assert(false);
    Mat accum_uvds; readFileToMat(accum_uvds, "examples/NYU_HANDS/accum_uvd.dat");

    // now compute the PCA
    cout << "accum_size = " << accum_uvds.size() << endl;
    cv::PCA pca(accum_uvds,cv::noArray()/*compute from data*/,CV_PCA_DATA_AS_ROW);
    Mat eigns;
    
    // read out the components
    cout << "EIGEN_VEC_SIZE = " << pca.eigenvectors.size() << endl;
    cout << "EIGEN_VEC = " << pca.eigenvectors << endl;
    cout << "EIGEN_VAL = " << pca.eigenvalues  << endl;
    cout << "mean = " << pca.mean           << endl;
    writeMatToFile(pca.eigenvectors, "out/pca_eigenvectors.dat");
    writeMatToFile(pca.eigenvalues, "out/pca_eigenvalues.dat");
    writeMatToFile(pca.mean, "out/pca_mean.dat");

    // write them out as a caffe blob for the IP layer.
    
    
    PCA_done = true;
    assert(false);
  }

  // accumulate the uvds for later analysis
  static void accum_uvd(std::vector<cv::Vec3d>&uvd)
  {
    static mutex m; lock_guard<mutex> l(m);
    static long next_pos = 0;
    assert(active_keypoints.size() == uvd.size());
    static Mat accum_uvds(50000,2*jvl::active_keypoints.size(),DataType<double>::type,Scalar::all(0));    
    if(next_pos >= accum_uvds.rows)
      return;    
    for(size_t iter = 0; iter < uvd.size(); ++iter)
    {
      accum_uvds.at<double>(next_pos,2*iter)   = uvd.at(iter)[0];
      accum_uvds.at<double>(next_pos,2*iter+1) = uvd.at(iter)[1];
    }
    next_pos++;
    if(next_pos % 1000 == 0)
    {
      cout << "(INFO) accumulated " << next_pos << " uvds" << endl;
    }
    
    if(next_pos >= accum_uvds.rows)
    {
      cout << "(INFO) writting accum_uvd" << endl;
      writeMatToFile(accum_uvds,accum_uvd_filename);
    }
  }
  
  template<typename Dtype>
  void HandDataLayerPrivate<Dtype>::producer()
  {
    TrainingHandData<Dtype> render;
    int hmChans = jvl::active_keypoints.size();
    render.data = std::make_shared<Blob<Dtype> >(batch_size, 1, image_res, image_res);
    render.label = std::make_shared<Blob<Dtype> >(batch_size,2*hmChans,1,1);
    render.index = std::make_shared<Blob<Dtype> >(batch_size,1,1,1);
    render.heatmap = std::make_shared<Blob<Dtype> >(batch_size,hmChans,jvl::heat_map_res,jvl::heat_map_res);

    for(int nIter = 0; nIter < batch_size; ++nIter)
    {
      // generate top and bottom
      jvl::RegExample ex;
      if(src_name == "NYU_HANDS_TRAIN")
	ex = jvl::load_nyu_datum("training");
      else if(src_name == "NYU_HANDS_TEST")
	ex = jvl::load_nyu_datum("testing");
      else if(src_name == "ICL_HANDS_TRAIN")
	ex = jvl::load_icl_datum("training");
      else if(src_name == "ICL_HANDS_TEST")
	ex = jvl::load_icl_datum("testing");
      else
	ex = jvl::load_direc_datum(src_name);
      
      // write the image evidence
      ex.Z.convertTo(ex.Z,DataType<Dtype>::type);
      if(ex.Z.type() != DataType<Dtype>::type)
	throw std::runtime_error("Bad conversion");
      double sx = render.data->width()/static_cast<double>(ex.Z.cols);
      double sy = render.data->height()/static_cast<double>(ex.Z.rows);
      for(auto && v : ex.uvd)
      {
	v[0] *= sx;
	v[1] *= sy;
      }
      cv::resize(ex.Z,ex.Z,Size(render.data->width(),render.data->height()),jvl::DEPTH_INTER_STRATEGY);
      if(src_name == "NYU_HANDS_TRAIN")
	augment<Dtype>(ex.Z,ex.uvd);
      //LocalContrastNormalization<Dtype>(ex.Z);
      if(ex.Z.type() != DataType<Dtype>::type)
	throw std::runtime_error("Bad conversion");      
      ex.Z = update_mean<Dtype>(ex.Z);
      Dtype* blob_image = render.data->mutable_cpu_data();
      //cout << "copying to blob_image count = " << render.data->count() << endl;    
      caffe_copy(ex.Z.size().area(),ex.Z.ptr<Dtype>(),
		 blob_image + render.data->offset(nIter));
      
      // write the label
      //accum_uvd(ex.uvd);
      //do_PCA();
      jvl::copy(ex.uvd,render.label.get(),nIter,1,1);
      
      // write index
      render.index->mutable_cpu_data()[nIter] = ex.index;

      // generate and write the heatmap
      for(int kpIter = 0; kpIter < jvl::active_keypoints.size(); ++kpIter)
      {
	Mat hm(jvl::heat_map_res,jvl::heat_map_res,DataType<Dtype>::type,Scalar::all(0));
	Vec3d uvd = ex.uvd.at(jvl::active_keypoints.at(kpIter));
	int u = (heat_map_res/image_res)*uvd[0];
	int v = (heat_map_res/image_res)*uvd[1];
	u = jvl::clamp<int>(0,u,hm.cols-1);
	v = jvl::clamp<int>(0,v,hm.rows-1);
	hm.at<Dtype>(v,u) = 1;
	cv::GaussianBlur(hm,hm,Size(),.25);
	caffe_copy(hm.size().area(),hm.ptr<Dtype>(),
		   render.heatmap->mutable_cpu_data() + render.heatmap->offset(nIter,kpIter,0,0));
      }
    }

    output_queue.push(render);
  }
  
  INSTANTIATE_CLASS(HandDataLayer);
  REGISTER_LAYER_CLASS(HandData);
}
