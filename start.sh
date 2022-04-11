#!/bin/bash

source config.sh
echo LIMIT_CPU: $LIMIT_CPU
echo LIMIT_MEM: $LIMIT_MEM

#warm-up
# bash -c "sleep 5; curl --connect-timeout 5 http://127.0.0.1:8088/logo_rec_binary --data-binary \"@test/imgs/xtr.jpg\" -X POST" &

rm -rf multiproc-tmp
mkdir multiproc-tmp
export PROMETHEUS_MULTIPROC_DIR=multiproc-tmp

workers=${WORKERS:-1}
gunicorn -c gunicorn.conf.py -b 0.0.0.0:8088 -w ${workers} -t 5000 --reload ServiceLogoV2:app
