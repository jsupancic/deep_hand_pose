#!/usr/bin/env sh
# Draw the Net
# ./python/draw_net.py examples/NYU_HANDS/train_test.prototxt out/net.png && okular out/net.png
#
# Plot the performance
# python2 ./tools/extra/plot_training_log.py.example 6 out/plot6.png out/caffe.loki.jsupanci.log.INFO.*
#
# Resume option
# --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate
#
# watch 'nvidia-smi -q -d UTILIZATION'
#
# See plot_cdfs.m

export FINETUNE=-weights=/home/jsupanci/Dropbox/out/2015.03.18-PCA-init/network_iter_15000.caffemodel

export GLOG_minloglevel=0
export GLOG_log_dir=out/
rm -v ./out/* ;
pushd build/;
make -j16 &&
    popd &&
    gdb ./build/tools/caffe -ex 'catch throw' -ex "r train --solver=examples/NYU_HANDS/solver.prototxt $FINETUNE " -ex 'q' | tee out/log.txt
    #valgrind ./build/tools/caffe train --solver=examples/NYU_HANDS/solver.prototxt
