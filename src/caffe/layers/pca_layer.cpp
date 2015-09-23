#include "caffe/common.hpp"
#include "caffe/custom_layers.hpp"
#include <opencv2/opencv.hpp>
#include <jvl/jvl.hpp>

using namespace jvl;
using namespace cv;
using namespace std;

namespace caffe
{
  template<typename Dtype>
  PCALayer<Dtype>::PCALayer(const LayerParameter& param) :
    InnerProductLayer<Dtype>(param)
  {
  }
  
  template<typename Dtype>
  void PCALayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				   const vector<Blob<Dtype>*>& top)
  {
    InnerProductLayer<Dtype>::LayerSetUp(bottom,top);

    Mat eigenvectors; readFileToMat(eigenvectors,"examples/deep_hand_pose/pca_eigenvectors.dat");
    Mat eigenvalues; readFileToMat(eigenvalues,"examples/deep_hand_pose/pca_eigenvalues.dat");
    Mat mean; readFileToMat(mean,"examples/deep_hand_pose/pca_mean.dat");

    //
    assert(this->blobs_.size() == 2);
    Blob<Dtype>*weights = this->blobs_[0].get();
    cout << "weights_shape: " << weights->num() << " " << weights->channels() << " " <<
      weights->height() << " " << weights->width() << endl;
    Blob<Dtype>*biases  = this->blobs_[1].get();
    cout << "biases_shape: " << biases->num() << " " << biases->channels() << " " <<
      biases->height() << " " << biases->width() << endl;    
    for(int output_iter = 0; output_iter < this->N_; ++output_iter)
    {
      // set the input weights
      for(int input_iter = 0; input_iter < this->K_; ++input_iter)
      {
	weights->mutable_cpu_data()[weights->offset(output_iter,input_iter)] =
	  //eigenvalues.at<double>(input_iter)*
	  eigenvectors.at<double>(output_iter,input_iter);
      }
      
      // set the bias term
      biases->mutable_cpu_data()[output_iter] = mean.at<double>(output_iter);
    }
  }

#ifdef CPU_ONLY
STUB_GPU(PCALayer);
#endif

INSTANTIATE_CLASS(PCALayer);
REGISTER_LAYER_CLASS(PCA);  
}
