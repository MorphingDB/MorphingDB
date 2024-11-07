# MorphingDB-Extension
- MorphingDB is a postgreSQL extension for supporting deep learning model inference within the database and vector storage.

- MorphingDB allows users to import the libtorch model and make inferences using data from within the database.

- MorphingDB allows users to store vector data with dimensional information, allowing libtorch to work directly with vector data in the database, users can store data preprocessed into vectors in the database to speed up inference.

- MorphingDB supports model centric and task centric inference.

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
MorphingDB supports Postgres 12+ in linux

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

### Install Onnxruntime
```shell
wget -P ./third_party https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz
tar -xvzf ./third_party/onnxruntime-linux-x64-1.18.1.tgz -C ./third_party
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



## How to use
### Prediction

#### Create model
```sql
SELECT create_model(model_name, model_path, model description);
```

#### Write pre and post process
You need to write the corresponding input and output handlers for the created model in src/external_process, and rebuild extension, make install.

#### Register process
```sql
SELECT register_process();
```

#### Run prediction function
```sql
SELECT predict_float([model_name], ['cpu'/'gpu'], [variable_input_column]) from [table];
SELECT predict_text([model_name], ['cpu'/'gpu'], [variable_input_column]) from [table];

```

#### Batch predictive acceleration
Use window functions to speed up predictions with variable window sizes.
```sql
SELECT comment,predict_batch_text([model_name], ['cpu'/'gpu'], [variable_input_column]) over (rows between current row and [window_size] following)
AS result 
FROM [table];

SELECT comment,predict_batch_float8([model_name], ['cpu'/'gpu'], [variable_input_column]) over (rows between current row and [window_size] following)
AS result
FROM [table];
```


### Tables
After the model is imported, the user can view the model information through the model_info table.
```
SELECT * from model_info;
```

### Vector store
MorphingDB supports vector storage, including storage of vector dimension information. In morphingdb, the vector type is mvec.
#### Create table with mvec
```sql
create table vec_test(id integer, vec mvec);
```

#### Insert vector
```sql
insert into vec_test values(1, '[1.0,2.2,3.123,4.2]{4}');
insert into vec_test values(1, '[1.0,2.2,3.123,4.2]{2,2}');
insert into vec_test values(1, ARRAY[1.0,2.0,3.0,1.2345]::float4[]::mvec);
```

#### Get vector data and shape
```sql
select get_mvec_shape(vec) from vec_test;
select get_mvec_data(vec) from vec_test;
```

#### Vector operation
```sql
update vec_test set vec=vec+vec;
update vec_test set vec=vec-text_to_mvec('[1,2,3,4]');

select * from vec_test where vec=='[1,2.2,3.123,4.2]';
```

#### TODO
MorphingDB will support the interconversion of libtorch vectors to mvec.


## Reference

[A Comparative Study of in-Database Inference Approaches](https://users.cs.utah.edu/~lifeifei/papers/icde22-indbinference.pdf)

[Learning a Data-Driven Policy Network for Pre-Training Automated Feature Engineering](https://openreview.net/pdf?id=688hNNMigVX)

[Pre-Trained Model Recommendation for Downstream Fine-tuning](https://arxiv.org/pdf/2403.06382.pdf)

[SmartLite: A DBMS-based Serving System for DNN Inference in Resource-constrained Environments](https://www.vldb.org/pvldb/vol17/p278-wu.pdf)