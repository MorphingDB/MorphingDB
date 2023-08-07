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

-- create batch function
CREATE OR REPLACE FUNCTION dummy_predict_batch_func_internal(IN aggstate internal, IN model_name cstring, IN type cstring, VARIADIC vec "any" , OUT internal)
AS 'MODULE_PATHNAME', 'predict_batch_dummy' LANGUAGE C;
CREATE OR REPLACE FUNCTION dummy_predict_batch_func_float(IN aggstate internal, OUT float8)
AS 'MODULE_PATHNAME', 'predict_batch_dummy' LANGUAGE C;
CREATE OR REPLACE FUNCTION dummy_predict_batch_func_text(IN aggstate internal, OUT text)
AS 'MODULE_PATHNAME', 'predict_batch_dummy' LANGUAGE C;

CREATE OR REPLACE FUNCTION predict_batch_accum(IN aggstate internal, IN model_name cstring, IN type cstring, VARIADIC vec "any" , OUT internal)
    AS 'MODULE_PATHNAME', 'predict_batch_accum' LANGUAGE C;
CREATE OR REPLACE FUNCTION predict_batch_accum_inv(IN aggstate internal, IN model_name cstring, IN type cstring, VARIADIC vec "any" , OUT internal)
    AS 'MODULE_PATHNAME', 'predict_batch_accum_inv' LANGUAGE C;
CREATE OR REPLACE FUNCTION predict_batch_final_float8(IN aggstate internal, OUT float8)
    AS 'MODULE_PATHNAME', 'predict_batch_final_float8' LANGUAGE C;
CREATE OR REPLACE FUNCTION predict_batch_final_text(IN aggstate internal, OUT text)
    AS 'MODULE_PATHNAME', 'predict_batch_final_text' LANGUAGE C;
CREATE OR REPLACE AGGREGATE predict_batch_float8(IN model_name cstring, IN type cstring, VARIADIC vec "any") (
    STYPE=internal,
    SFUNC=dummy_predict_batch_func_internal,
    FINALFUNC=dummy_predict_batch_func_float,
    MSFUNC=predict_batch_accum,
    MINVFUNC=predict_batch_accum_inv,
    MFINALFUNC=predict_batch_final_float8,
    MSTYPE=internal);
CREATE OR REPLACE AGGREGATE predict_batch_text(IN model_name cstring, IN type cstring, VARIADIC vec "any") (
    STYPE=internal,
    SFUNC=dummy_predict_batch_func_internal,
    FINALFUNC=dummy_predict_batch_func_text,
    MSFUNC=predict_batch_accum,
    MINVFUNC=predict_batch_accum_inv,
    MFINALFUNC=predict_batch_final_text,
    MSTYPE=internal);
CREATE OR REPLACE PROCEDURE enable_print_batch_time(IN enable bool)
    AS 'MODULE_PATHNAME', 'enable_print_batch_time' LANGUAGE C;