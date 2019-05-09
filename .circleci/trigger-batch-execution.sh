#!/bin/sh -ex

COMMIT_MESSAGE=$(git log --format=oneline -n 1 $CIRCLE_SHA1)
BUILD_REF=$(cat version.txt)
REGEXP="TEST"
SERVICE_OPTIMIZER="https://www.habx-dev.fr/api/optimizer-v2/job"


echo "Commit message is ${COMMIT_MESSAGE}" 
if [[ $COMMIT_MESSAGE =~ $REGEXP ]] ; then
    cmd=curl -H "Content-Type: application/json" -H "cookie: jwt=YOURS;" -XPOST $SERVICE_OPTIMIZER -d '{"batchExecution": {"sampleRequestGroupId": 1,"version": "'${BUILD_REF}'","usage": "test","meta":{"params":{}}}}'
    echo $cmd
else
    echo "Regexp ${REGEXP} didn't match"
fi