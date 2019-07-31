#!/bin/sh
if [ "${ENABLE_TCMALLOC}" = "true" ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
  echo Using tcmalloc
fi

bin/job.py
