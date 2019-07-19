#!/bin/sh
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
echo Using $LD_PRELOAD
bin/worker.py
