#!/bin/sh
if ! test -f example/voc/MobileNetSSD_train.prototxt ;then
	echo "error: example/voc/MobileNetSSD_train.prototxt does not exist."
  exit 1
fi

echo $(ls -t snapshot/voc/*.caffemodel | head -n 1)

#set your caffe path, such as '/home/work/caffe_ssd/build/tools/caffe'
/home/work/caffe_ssd/build/tools/caffe train --solver="example/voc/solver_train.prototxt" \
--weights=$(ls -t snapshot/voc/*.caffemodel | head -n 1) \
--gpu 0 2>&1 | tee example/voc/MobileNetSSD_voc.log  


