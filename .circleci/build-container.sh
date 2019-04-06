#!/bin/sh -ex

export BUILD_REF=$(if [ -z "$CIRCLE_TAG" ]; then echo ${CIRCLE_BRANCH/\//-}-${CIRCLE_SHA1:0:7}; else echo $CIRCLE_TAG; fi)
export PLUGIN_TAGS=${BUILD_REF}
export PLUGIN_REPO=724009402066.dkr.ecr.eu-west-1.amazonaws.com/worker-optimizer-v2
/usr/local/bin/dockerd-entrypoint.sh /bin/drone-docker-ecr
