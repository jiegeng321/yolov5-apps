#!/bin/bash

APP_HOME=
MODEL_LOCAL_PATH=
MODEL_BUCKET=
MODEL_REMOTE_PATH=

source ./config.sh

if [[ "$APPNAME" = "" ]]
then
    APPNAME=ai-sphinx-style
fi

die() {
    if [ $# != 2 ] ; then
        echo " The first is return code,the second error message!"
        echo " e.g.: die 1 'error message'"
        exit 1;
    fi
    code=$1
    msg=$2
    echo ${msg} && exit ${code}
}

download_model() {
    echo "downloading model..."
    python updown_model.py -l $MODEL_LOCAL_PATH -b $MODEL_BUCKET -r $MODEL_REMOTE_PATH
    if [ ! "$(ls ${MODEL_LOCAL_PATH})" ]; then
        die 1 "download model failed..."
    fi
}
download_model_t4() {
    echo "downloading model..."
    python updown_model.py -l $MODEL_LOCAL_PATH -b $MODEL_BUCKET -r $MODEL_REMOTE_PATH_T4
    if [ ! "$(ls ${MODEL_LOCAL_PATH})" ]; then
        die 1 "download model failed..."
    fi
}

download_config() {
    echo "downloading config file..."
    python updown_model.py -l $MODEL_LOCAL_PATH -b $MODEL_BUCKET -r $CONFIG_REMOTE_PATH
    if [ ! "$(ls ${MODEL_LOCAL_PATH})" ]; then
        die 1 "download model failed..."
    fi
}

# 编译打包
build() {
    echo "packaging..."
    echo "APP_HOME: ${APP_HOME}"
    echo "APPNAME: ${APPNAME}"
    mkdir -p ${APP_HOME}/target
    tar czf ${APP_HOME}/target/$APPNAME-dist.tar.gz Dockerfile *.sh *.py model/* src/* --exclude updown_model.py
}

check() {
    cd ${APP_HOME}
    if [ ! -f ./target/${APPNAME}-dist.tar.gz ]; then
        die 1 "package not generated and build fail..."
    fi

    die 0 "package generated and build success..."
}

main() {
    download_model
    download_model_t4
    download_config
    build
    check
}

main
