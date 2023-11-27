# PGDL-Extension

postgres extension for supporting ai prediction

# Quick Start

```
sudo docker build -t test_pgdl:0.1 .
```

# Installation

## Install libtorch

```
-- cpu
wget -P ./third_party https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip
-- gpu
wget -P ./third_party https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-2.0.0%2Bcu117.zip
unzip -d ./third_party/libtorch ./third_party/*.zip
rm ./third_party/*.zip
```

## Install Postgres

```
sudo yum install postgresql
```

## Install OpenCV

```
sudo yum install opencv opencv-devel opencv-python
```

## Install SentencePiece

```
cd third_party
git clone https://github.com/google/sentencepiece
cd sentencepiece
mkdir build
cd build
cmake ..
make -j4
sudo make install
```

## Make and install

```
cmake -DCMAKE_PREFIX_PATH="third_party/libtorch" ..
make -j4
make install
```

# Getting Started

start server

```
initdb -D <data_directory>
postgres -D <data_directory> -p 5432
```

connect to server

```
psql -p 5432 -d postgres
```

## Quick Test

```
psql -p 5432 -d postgres -f test/sql/docker_test.sql
```

