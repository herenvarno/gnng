#!/usr/bin/env bash
if [ $# -gt '0' ]; then
	MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
	LOGDIR=$MYDIR/../data
	mkdir -p $LOGDIR
fi

