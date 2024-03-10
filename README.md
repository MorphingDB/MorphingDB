# MorphingDB-Extension

MorphingDB extension for deep learning Prediction and storage vector type.
MorphingDB currently does not support in-database training, only in-database predictions for libtorch models.

## Quick Docker Start

```sql
# build docker image
sudo docker build -t MorphingDB .
# run docker contanier
sudo docker run --name MorphingDB_test -e POSTGRES_PASSWORD=123456 -d MorphingDB:latest -p 5432:5432
# enter docker 
sudo docker exec -it [contanier id] /bin/bash
# run test
su postgres
psql -p 5432 -d postgres -c 'create extension pgdl;'
psql -p 5432 -d postgres -f /home/pgdl/test/sql/docker_test.sql
psql -p 5432 -d postgres -f /home/pgdl/test/sql/vector_test.sql
```

## Installation

### Install libtorch

```shell
-- cpu
wget -P ./third_party https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip
-- gpu
wget -P ./third_party https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-2.0.0%2Bcu117.zip
unzip -d ./third_party/libtorch ./third_party/*.zip
rm ./third_party/*.zip
```

### Install Postgres

```shell
sudo yum install postgresql
```

### Install OpenCV

```shell
sudo yum install opencv opencv-devel opencv-python
```

### Install SentencePiece

```shell
cd third_party
git clone https://github.com/google/sentencepiece
cd sentencepiece
mkdir build
cd build
cmake ..
make -j4
sudo make install
```

### Make and install

```shell
cmake -DCMAKE_PREFIX_PATH="third_party/libtorch" ..
make -j4
make install
```

## Getting Started

Start server

```shell
initdb -D <data_directory>
postgres -D <data_directory> -p 5432
```

Connect to server

```shell
psql -p 5432 -d postgres
```

Enable the extension
```sql
CREATE EXTENSION pgdl;
```


## Prediction

### Create model
```sql
SELECT create_model(model_name, model_path, model description);
```

### Write pre and post process
You need to write the corresponding input and output handlers for the created model in src/external_process, and rebuild extension, make install.

### Register process
```sql
SELECT register_process();
```

### Run prediction function
```sql
SELECT predict_float('sst2', 'cpu', comment) from npl_test;
```

### Batch predictive acceleration
Use window functions to speed up predictions with variable window sizes.
```sql
SELECT comment,predict_batch_text('sst2', 'cpu', comment) over (rows between current row and 15 following)
AS result 
FROM nlp_test;
```

### TODO
Dozens of basic models will be supported, basic pre-processing post-processing processes will be added.


## Vector store
MorphingDB supports vector storage, including storage of vector dimension information. In morphingdb, the vector type is mvec.
### Create table with mvec
```sql
create table vec_test(id integer, vec mvec);
```

### Insert vector
```sql
insert into vec_test values(1, '[1.0,2.2,3.123,4.2]{4}');
insert into vec_test values(1, '[1.0,2.2,3.123,4.2]{2,2}');
insert into vec_test values(1, ARRAY[1.0,2.0,3.0,1.2345]::float4[]::mvec);
```

### Get vector data and shape
```sql
select get_mvec_shape(vec) from vec_test;
select get_mvec_data(vec) from vec_test;
```

### Vector operation
```sql
update vec_test set vec=vec+vec;
update vec_test set vec=vec-text_to_mvec('[1,2,3,4]');

select * from vec_test where vec=='[1,2.2,3.123,4.2]';
```

### TODO
MorphingDB will support the interconversion of libtorch vectors to mvec.


