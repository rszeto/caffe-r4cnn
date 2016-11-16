#ifndef CAFFE_SOFTMAX_WITH_VIEW_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_WITH_VIEW_LOSS_LAYER_HPP_

#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/loss_layer.hpp"
// #include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
class SoftmaxWithViewLossLayer : public LossLayer<Dtype> {
 public:
  explicit SoftmaxWithViewLossLayer(const LayerParameter& param)
  	  : LossLayer<Dtype>(param),
  	  softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxWithViewLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; } 
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  // sum of weights
  Dtype weights_sum_;
};

}

#endif // CAFFE_SOFTMAX_WITH_VIEW_LOSS_LAYER_HPP_