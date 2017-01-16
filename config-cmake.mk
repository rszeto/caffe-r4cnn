SHELL=bash -i
CAFFE_INSTALL_DIR:= /z/home/szetor/sw/caffe-r4cnn
CAFFE_SOURCE_DIR:= /z/home/szetor/Documents/DENSO_VCA/RenderForCNN/caffe-r4cnn
caffe:
	module load cuda cudnn opencv gflags glog lmdb boost && \
	cmake $(CAFFE_SOURCE_DIR) \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_INSTALL_PREFIX=$(CAFFE_INSTALL_DIR) \
			-DUSE_OPENCV=ON \
			-DOpenCV_DIR=$${OPENCV_ROOT}/share/OpenCV \
			-DGLOG_ROOT_DIR="$${GLOG_INC}\;$${GLOG_ROOT}" \
			-DGFLAGS_ROOT_DIR="$${GFLAGS_INC}" \
			-DCUDNN_ROOT=$${CUDNN_ROOT}/include\;$${CUDNN_ROOT}/lib64 \
			-DCUDA_TOOLKIT_ROOT_DIR=$${CUDA_ROOT} \
			-DCUDA_ARCH_NAME=Manual \
			-DCUDA_ARCH_BIN="20 21(20) 30 35 50 60 61(60)" \
			-DCUDA_ARCH_PTX="60" \
			-DALLOW_LMDB_NOLOCK=ON \
			-DBUILD_matlab=ON
