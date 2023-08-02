-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pgdl" to load this file. \quit

-- create model table
CREATE TABLE IF NOT EXISTS model_info(model_name text primary key, model_path text, create_time timestamp, update_time timestamp, md5 text, upload_by text, discription text);


-- create function
CREATE OR REPLACE FUNCTION create_model(IN model_name cstring, IN model_path cstring, IN discription cstring, OUT boolean) 
    AS 'MODULE_PATHNAME', 'create_model' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION modify_model(IN model_name cstring, IN model_path cstring, OUT boolean) 
    AS 'MODULE_PATHNAME', 'modify_model' LANGUAGE C;
CREATE OR REPLACE FUNCTION drop_model(IN model_name cstring, OUT boolean)
    AS 'MODULE_PATHNAME', 'drop_model' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION predict_float(IN model_name cstring, IN type cstring, IN url text, OUT float)
    AS 'MODULE_PATHNAME', 'predict_float' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION predict_text(IN model_name cstring, IN type cstring, IN url text, OUT text)
    AS 'MODULE_PATHNAME', 'predict_text' LANGUAGE C STRICT;
CREATE OR REPLACE FUNCTION register_process(OUT void)
    AS 'MODULE_PATHNAME', 'register_process' LANGUAGE C STRICT;
