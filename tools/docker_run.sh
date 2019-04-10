#!/bin/sh -ex

# Example usage:
# tools/docker_cli.sh bin/cli.py -l resources/blueprints/009.json -s resources/specifications/009_setup1.json -o out

DOCKER_IMAGE=optimizer-v2:$(cat requirements.txt Dockerfile | shasum |cut -f 1 -d' ')

if [[ "$(docker images -q ${DOCKER_IMAGE} 2> /dev/null)" == "" ]]; then
  echo "Creating container image"
  docker build . -t ${DOCKER_IMAGE}
fi

docker run -w /work -v $(pwd):/work ${DOCKER_IMAGE} $@
