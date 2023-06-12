#!/bin/bash
set -e -u -x

$BUILD_PYTHON_ROOT_PATH/bin/python -m pip install cython numpy

yum install -y openssl-devel libtiff-devel blas-devel lapack-devel

git clone https://github.com/curl/curl && cd curl && cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF -DCURL_CA_BUNDLE=none -DCURL_CA_PATH=none . && make -j32 && make install -j32 && cd .. && rm -rf curl
git clone https://github.com/JosephP91/curlcpp && cd curlcpp && cmake -DBUILD_SHARED_LIBS=OFF -DCURLCPP_USE_PKGCONFIG=OFF . && make -j32 && make install -j32 && cd .. && rm -rf curlcpp
git clone https://github.com/pybind/pybind11 && cd pybind11 && cmake -DPYBIND11_TEST=OFF -DPYBIND11_PYTHON_VERSION=3.$BUILD_MINOR . && make -j32 && make install -j32 && cd .. && rm -rf pybind11
git clone https://github.com/OSGeo/PROJ && cd PROJ && cmake -DBUILD_TESTING=OFF . && make -j32 && make install -j32 && cd .. && rm -rf PROJ
git clone https://github.com/xtensor-stack/xtl && cd xtl && cmake . && make -j32 && make install -j32 && cd .. && rm -rf xtl
git clone https://github.com/xtensor-stack/xsimd && cd xsimd && cmake . && make -j32 && make install -j32 && cd .. && rm -rf xsimd
git clone https://github.com/xtensor-stack/xtensor && cd xtensor && cmake . && make -j32 && make install -j32 && cd .. && rm -rf xtensor
git clone https://github.com/xtensor-stack/xtensor-python && cd xtensor-python && cmake -DPYBIND11_PYTHON_VERSION=3.$BUILD_MINOR . && make -j32 && make install -j32 && cd .. && rm -rf xtensor-python
git clone https://github.com/xtensor-stack/xtensor-blas && cd xtensor-blas && cmake . && make -j32 && make install -j32 && cd .. && rm -rf xtensor-blas
git clone https://github.com/catchorg/Catch2 && cd Catch2 && mkdir build && cd build && cmake .. && make -j32 && make install -j32 && cd ../.. && rm -rf Catch2
git clone https://github.com/fferflo/xtensor-interfaces && cd xtensor-interfaces && cmake . && make -j32 && make install -j32 && cd .. && rm -rf xtensor-interfaces

git clone https://github.com/fferflo/cosy
cd cosy && mkdir build && cd build
cmake -DPython_ROOT_DIR=$BUILD_PYTHON_ROOT_PATH ..
make -j32
cd python

$BUILD_PYTHON_ROOT_PATH/bin/python setup.py bdist_wheel
rename py3- cp3$BUILD_MINOR- dist/*.whl
auditwheel repair dist/*.whl --plat $BUILD_PLATFORM
cp wheelhouse/* /io