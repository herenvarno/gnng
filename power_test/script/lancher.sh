#!/usr/bin/env bash
if [ $# -ge '3' ]; then
	MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
	PREFIX=$MYDIR/../..
	BINDIR=$PREFIX/conv_test
	DATADIR=$PREFIX/tb/$1
	ORIGINAL_CAFFE_LIB=/usr/lib
	MODIFIED_CAFFE_LIB=$PREFIX/caffe/.build_release/lib
	if [ $3 -le '0' ]; then
		export LD_LIBRARY_PATH=$ORIGINAL_CAFFE_LIB
		if [ -e $DATADIR/deploy.prototxt ]; then
			cp -f $DATADIR/deploy.prototxt $DATADIR/net.prototxt
		else
			exit -1
		fi
	else
		export LD_LIBRARY_PATH=$MODIFIED_CAFFE_LIB
		if [ -e $DATADIR/deploy.prototxt ]; then
			cp -f $DATADIR/deploy.prototxt $DATADIR/net.prototxt
		else
			exit -1
		fi
		sed -i 's/convolution_param\s*{/convolution_param {\n\tset: '$3'/g' $DATADIR/net.prototxt
	fi
	echo $LD_LIBRARY_PATH
	$BINDIR/test $2 $DATADIR/net.prototxt $DATADIR/net.caffemodel $DATADIR/test.png
fi
