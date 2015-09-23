#include <caffe/blob.hpp>

namespace jvl
{
  namespace
  {
    //std::vector<int> active_keypoints{0,3,6,8,12,15,18,21,24,26,28,34,30,29};
    std::vector<int> active_keypoints{0,3,6,9,12,15,18,21,24,25,27,30,31,32};
    std::vector<int> icl_correspondence{
      15,15,14,14,13,13,
	12,12,11,11,10,10,
	9,9,8,8,7,7,
	6,6,5,5,4,4,
	3,3,2,2,1,1,
	0,0,0,0,0,0};
    //std::vector<int> active_keypoints{0,2,4,6,8,10,12,14,16,18,20,22,24,26};
  }
  
  template<typename Dtype>
  void copy(const std::vector<cv::Vec3d>&uvd,Dtype* label_data,int nIter,double sx,double sy)
  {    
    //cout << "copying to label_blob count = " << label_blob->count() << endl;
    //assert(label_blob->count() == label_blob->num()*active_keypoints.size()*2);
    std::vector<Dtype> labels(active_keypoints.size()*2);
    for(int uOv = 0; uOv < 2; ++uOv)
      for(int kpIter = 0; kpIter < active_keypoints.size(); ++kpIter)
      {
	double sf = ((uOv==0)?sx:sy);
	labels.at(2*kpIter + uOv) = sf*uvd.at(active_keypoints.at(kpIter))[uOv];
      }
    caffe::caffe_copy(2*active_keypoints.size(),
		      &labels.at(0),
		      label_data);    
  }
}
