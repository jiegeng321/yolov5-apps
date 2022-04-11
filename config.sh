#!/bin/bash

export APPNAME=ai-brand-logo
export USER_HOME=/home/admin
export APP_HOME=${USER_HOME}/${APPNAME}
export APP_HOME=.
export MODEL_HOME=./model
export MODEL_LOCAL_PATH=${MODEL_HOME}
export MODEL_REMOTE_PATH=${APPNAME}/0.16/20220307_yolov5m_brand776_style1376_P40.trt
export MODEL_REMOTE_PATH_T4=${APPNAME}/0.16/20220307_yolov5m_brand776_style1376_T4.trt
export CONFIG_REMOTE_PATH=${APPNAME}/0.16/config.py
export MODEL_BUCKET=ai_vision
export S3_ACCESS_KEY=8KNQVOU27LTFQ32LD3DT
export S3_SECRET_KEY=VZM9KfGqFn6Q6DeYKY8wsR10l1K1DPWu8W4YA0JM
export S3_ENDPOINTS=http://s3.td:8080
