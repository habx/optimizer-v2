#!/bin/sh -ex

BUILD_REF=$(cat version.txt)
SERVICE_OPTIMIZER_URL="https://www.habx-dev.fr/api/optimizer-v2/job"

curl -H "Content-Type: application/json" -H "x-habx-token: ${HABX_TOKEN}" -XPOST $SERVICE_OPTIMIZER_URL -d '{"query":"mutation {upsertVersion(version: {name: "'${BUILD_REF}'" family: 2 meta: {author: "'${CIRCLE_USERNAME}'"} }) { id name family release createdAt }}"}'
