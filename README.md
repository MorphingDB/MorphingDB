# MorphingDB-Extension
- MorphingDB is a postgreSQL extension for supporting deep learning model inference within the database and vector storage.

- MorphingDB allows users to import the libtorch model and make inferences using data from within the database.

- MorphingDB allows users to store vector data with dimensional information, allowing libtorch to work directly with vector data in the database, users can store data preprocessed into vectors in the database to speed up inference.

- MorphingDB supports model centric and task centric inference.

## Quick Start
dowload [zip](https://drive.google.com/file/d/17EhgU-ujGzNP75ytClrivUT4tXblLqnV/view?usp=sharing) to morphingdb project path

unzip model
```shell
unzip models.zip -d ./model/
```

use docker image from docker hub
```shell
sudo docker pull morphingdb/morphingdb:latest
```

or build docker image
```shell
sudo docker build -t morphingdb .
```


create docker contanier
```shell
# abs_morphingdb_test_dir is the absolute path of the morphingdb project directory from https://github.com/MorphingDB/Morphingdb_test/
sudo docker run -d 
--name MorphingDB_test 
-e POSTGRES_PASSWORD=123456 
-v [abs_morphingdb_test_dir]:[same_abs_morphingdb_test_dir]
-v [model_dir]:/home/pgdl/model/
-v [data_dir]:/var/lib/postgresql/data
-p [port]:5432 
morphingdb
```
enter docker 
```shell
sudo docker exec -it [contanier id] /bin/bash
```

Connect to server

```shell
psql -p 5432 -d postgres -U postgres -h localhost
```

Enable the extension
```sql
CREATE EXTENSION pgdl;
```

## How to use

#### Create model
```sql
SELECT create_model(model_name, model_path, base_model, model description);
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

#### Task centric
```sql
SELECT image_classification(<column_name>) as task_result
FROM <table_name>
WHERE <conditions>;
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
insert into vec_test values(1, '[1.0,2.2,3.123,4.2]');
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

## Morphingdb Test

1. download models from [here](https://drive.google.com/file/d/17EhgU-ujGzNP75ytClrivUT4tXblLqnV/view)

```shell
unzip models.zip -d ./test/morphingdb_test/
```

2. dowload data from [here](https://drive.google.com/file/d/1bAg66ifb54ge_J8CRjkSk_yRhyTRZYWe/view?usp=sharing)

```shell
unzip data.zip -d ./test/morphingdb_test/
```

### Run

```shell
cd test
uv run main.py
```

### Result

result will be saved in `result` dir.


## Reference

[A Comparative Study of in-Database Inference Approaches](https://users.cs.utah.edu/~lifeifei/papers/icde22-indbinference.pdf)

[Learning a Data-Driven Policy Network for Pre-Training Automated Feature Engineering](https://openreview.net/pdf?id=688hNNMigVX)

[Pre-Trained Model Recommendation for Downstream Fine-tuning](https://arxiv.org/pdf/2403.06382.pdf)

[SmartLite: A DBMS-based Serving System for DNN Inference in Resource-constrained Environments](https://www.vldb.org/pvldb/vol17/p278-wu.pdf)