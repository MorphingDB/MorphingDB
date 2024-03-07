-- support mvec type (row storage and column storage)
\echo Use "ALTER EXTENSION pgdl UPDATE to '1.1.0'" to load this file. \quit


-- mvec type and function
CREATE TYPE mvec;

CREATE FUNCTION mvec_input(cstring) RETURNS mvec 
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec_output(mvec) RETURNS cstring 
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec_receive(internal) RETURNS mvec
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec_send(mvec) RETURNS bytea
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION get_mvec_data(mvec) RETURNS float4[] 
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION get_mvec_shape(mvec) RETURNS float4[] 
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION text_to_mvec(text) RETURNS mvec 
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE mvec (
  INPUT = mvec_input,
  OUTPUT = mvec_output,
  RECEIVE = mvec_receive,
  SEND = mvec_send,
  STORAGE = EXTENDED 
);

-- operator
CREATE FUNCTION mvec_add(mvec, mvec) RETURNS mvec
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec_sub(mvec, mvec) RETURNS mvec
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec_equal(mvec, mvec) RETURNS bool
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OPERATOR + (
	LEFTARG = mvec, RIGHTARG = mvec, PROCEDURE = mvec_add,
	COMMUTATOR = +
);

CREATE OPERATOR - (
	LEFTARG = mvec, RIGHTARG = mvec, PROCEDURE = mvec_sub,
	COMMUTATOR = -
);

CREATE OPERATOR == (
	LEFTARG = mvec, RIGHTARG = mvec, PROCEDURE = mvec_equal,
	COMMUTATOR = '=='
);

-- cast function
CREATE FUNCTION mvec(float4[], integer, boolean) RETURNS mvec
	AS 'MODULE_PATHNAME/pgdl.so', 'array_to_mvec'LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec(float8[], integer, boolean) RETURNS mvec
	AS 'MODULE_PATHNAME/pgdl.so', 'array_to_mvec'LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec(integer[], integer, boolean) RETURNS mvec
	AS 'MODULE_PATHNAME/pgdl.so', 'array_to_mvec' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec(text, integer, boolean) RETURNS mvec
	AS 'MODULE_PATHNAME/pgdl.so', 'text_to_mvec' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mvec_to_float_array(mvec, integer, boolean) RETURNS float4[]
	AS 'MODULE_PATHNAME/pgdl.so' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE CAST (float4[] AS mvec)
	WITH FUNCTION mvec(float4[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (float8[] AS mvec)
	WITH FUNCTION mvec(float8[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (integer[] AS mvec)
	WITH FUNCTION mvec(integer[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (text AS mvec)
	WITH FUNCTION mvec(text, integer, boolean) AS ASSIGNMENT;

CREATE CAST (mvec AS float4[])
	WITH FUNCTION mvec_to_float_array(mvec, integer, boolean) AS IMPLICIT;

-- column store access method
-- CREATE FUNCTION mvec_am_handler(internal) RETURNS table_am_handler
-- 	AS 'MODULE_PATHNAME/mvec_am.so', 'mvec_am_handler' LANGUAGE C;

-- CREATE ACCESS METHOD mvec_am TYPE TABLE HANDLER mvec_am_handler;
