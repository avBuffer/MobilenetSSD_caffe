#!/bin/sh

latest=$(ls -t snapshot/voc/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi

echo lastest
../../build/tools/caffe train -solver="example/voc/solver_test.prototxt" \
--weights=$(ls -t snapshot/voc/*.caffemodel | head -n 1) \
-gpu 0
