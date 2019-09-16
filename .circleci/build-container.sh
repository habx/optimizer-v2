#!/bin/sh -ex
ls -lh
# export BUILD_REF=$(if [ -z "$CIRCLE_TAG" ]; then echo ${CIRCLE_BRANCH/\//-}-${CIRCLE_SHA1:0:7}; else echo $CIRCLE_TAG; fi)
# export BUILD_REF=$(cat version.txt)
export PLUGIN_TAGS=$(if [ -z "$CIRCLE_TAG" ]; then echo $(cat version.txt); else echo $CIRCLE_TAG; fi)
export PLUGIN_REPO=724009402066.dkr.ecr.eu-west-1.amazonaws.com/worker-optimizer
/usr/local/bin/dockerd-entrypoint.sh /bin/drone-docker-ecr
