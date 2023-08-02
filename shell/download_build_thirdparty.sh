#!/bin/bash
###
 # @Author: laihuihang laihuihang@foxmail.com
 # @Date: 2023-07-25 17:14:56
 # @LastEditors: laihuihang laihuihang@foxmail.com
 # @LastEditTime: 2023-07-27 17:29:49
 # @FilePath: /pgdl/shell/download_build_thirdparty.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 


function log()
{
    local level=$1
    shift
    local message=$*
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    local log_file="./log.txt"

    # 打印执行结果、时间戳和日志等级
    echo "[${timestamp}] [${level}] ${message}"

    # 保存日志到文件
    echo "[${timestamp}] [${level}] ${message}" >> "${log_file}"
}

function loginfo()
{
    log "INFO" $*
}

function logerror()
{
    log "ERROR" $*
}

cd "$(dirname "$0")"
script_path=$(pwd)

# download sentencepiece
git clone https://github.com/google/sentencepiece.git /tmp/pgdl/third_party/sentencepiece
if [[ $? -ne 0 ]]; then
    logerror "git clone sentencepiece error!"
    exit 1
else
    loginfo "git clone sentencepiece success!"
fi

echo "add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)" >> ${script_path}/../third_party/sentencepiece/src/CMakeLists.txt
cd ${script_path}/../third_party/sentencepiece
mkdir build && cd build
cmake -DSPM_ENABLE_TCMALLOC=OFF ..
if [[ $? -ne 0 ]]; then
    logerror "generate sentencepiece makefile error!"
    exit 1
else
    loginfo "generate sentencepiece makefile success!"
fi

make -j$(nproc)
if [[ $? -ne 0 ]]; then
    logerror "build sentencepiece error!"
    exit 1
else
    loginfo "build sentencepiece success!"
fi

make install
if [[ $? -ne 0 ]]; then
    logerror "install sentencepiece error!"
    exit 1
else
    loginfo "install sentencepiece success!"
fi

cd -

# download libtorch
wget -P ${script_path}/../third_party https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip --no-check-certificate
if [[ $? -ne 0 ]]; then
    logerror "download libtorch error!"
    exit 1
else
    loginfo "download libtorch success!"
fi
# unzip libtorch
unzip -d ${script_path}/../third_party ${script_path}/../third_party/*.zip
if [[ $? -ne 0 ]]; then
    logerror "unzip libtorch error!"
    exit 1
else
    loginfo "unzip libtorch success!"
fi

rm -f ${script_path}/../third_party/*.zip

# download opencv
wget -P ${script_path}/../third_party/ https://codeload.github.com/opencv/opencv/zip/refs/tags/3.4.16 --no-check-certificate
if [[ $? -ne 0 ]]; then
    logerror "download opencv error!"
    exit 1
else
    loginfo "download opencv success!"
fi
# unzip opencv
unzip -d ${script_path}/../third_party/ ${script_path}/../third_party/3.4.16
if [[ $? -ne 0 ]]; then
    logerror "unzip opencv error!"
    exit 1
else
    loginfo "unzip opencv success!"
fi

rm -f ${script_path}/../third_party/3.4.16
# build opencv
cd ${script_path}/../third_party/opencv-3.4.16
mkdir build && cd build
cmake ..
if [[ $? -ne 0 ]]; then
    logerror "generate opencv makefile error!"
    exit 1
else
    loginfo "generate opencv makefile success!"
fi

make -j$(nproc)
if [[ $? -ne 0 ]]; then
    logerror "build opencv error!"
    exit 1
else
    loginfo "build opencv success!"
fi

make install
if [[ $? -ne 0 ]]; then
    logerror "install opencv error!"
    exit 1
else
    loginfo "install opencv success!"
fi

cd ${script_path}/../
