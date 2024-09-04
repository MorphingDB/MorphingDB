-- support model parameter extraction and model selection
\echo Use "ALTER EXTENSION pgdl UPDATE to '1.2.0'" to load this file. \quit

CREATE TABLE IF NOT EXISTS model_layer_info(
    model_name text, 
    layer_name text, 
    layer_index integer, 
    parameter mvec, 
    PRIMARY KEY (model_name, layer_name)
);

CREATE TABLE IF NOT EXISTS base_model_info(
    base_model_name text, 
    md5 text, 
    base_model_path text, 
    PRIMARY KEY (base_model_name)
);


ALTER TABLE model_info
ADD COLUMN base_model text;


CREATE OR REPLACE FUNCTION create_model(
    IN model_name cstring,
    IN model_path cstring,
    IN base_model cstring,  
    IN description cstring,
    OUT boolean
)
AS 'MODULE_PATHNAME', 'create_model'
LANGUAGE C STRICT;


CREATE OR REPLACE FUNCTION image_to_vector(  
    width integer,  
    height integer,  
    norm_mean double precision,  
    norm_std double precision,  
    image_url text  
)  
RETURNS mvec
AS 'MODULE_PATHNAME', 'image_pre_process'  
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION text_to_vector(  
    text_a text  
)  
RETURNS mvec
AS '/home/lhh/pg_complie/lib/postgresql/pgdl.so', 'text_pre_process'  
LANGUAGE C STRICT;


CREATE OR REPLACE FUNCTION print_cost()
RETURNS text
AS '/home/lhh/pg_complie/lib/postgresql/pgdl.so', 'print_cost'
LANGUAGE C STRICT;

