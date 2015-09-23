#ifndef CAFFE_RENDER_LAYERS
#define CAFFE_RENDER_LAYERS

#include "caffe/layer.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe
{
  template<typename Dtype>
  class HandDataLayerPrivate;
  template <typename Dtype>
  class HandDataLayer : public Layer<Dtype> {
  public:
    // construct
    explicit HandDataLayer(const LayerParameter& param);
    virtual ~HandDataLayer();
    // generate data via a rendering engine
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			     const vector<Blob<Dtype>*>& top);
    // not sure what this does yet?
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			      const vector<bool>& propagate_down,
			      const vector<Blob<Dtype>*>& bottom);
    // it's important yo
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			 const vector<Blob<Dtype>*>& top);

    // load stuff
    void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);

  protected:
    string src_name;
    size_t batch_size;
    HandDataLayerPrivate<Dtype>*private_;
    
    TransformationParameter transform_param_;
  };

  template <typename Dtype>
  class MVRegressionLossLayer : public Layer<Dtype>
  {
  public:
    // construct
    explicit MVRegressionLossLayer(const LayerParameter& param);
    // generate data via a rendering engine
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			     const vector<Blob<Dtype>*>& top);
    // not sure what this does yet?
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			      const vector<bool>& propagate_down,
			      const vector<Blob<Dtype>*>& bottom);
    // it's important yo
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			 const vector<Blob<Dtype>*>& top);

    // load stuff
    void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		    const vector<Blob<Dtype>*>& top);

  protected:
  };  
  
  template<typename Dtype>
  class PCALayer : public InnerProductLayer<Dtype>
  {
  public:
    // load stuff
    explicit PCALayer(const LayerParameter& param);
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			    const vector<Blob<Dtype>*>& top);    
  };
}

#endif
