#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <string>

#ifdef __cplusplus
extern "C"
{
#include <stddef.h>
#include <stdint.h>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

	#define MAX_VECTOR_DIM (102400000)
	#define MAX_VECTOR_SHAPE_SIZE (10)

	typedef int64_t RowId;
	typedef unsigned int Oid;

	typedef union
	{
		struct MVecRef
		{
			int32_t header_; /* alwary b100 0000 00 (i64 ## b00) */
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
			char unused[3];
			bool is_ref_tag; /* data is row_id when true or vector_data when false */
#else
		bool is_ref_tag; /* data is row_id when true or vector_data when false */
		char unused[3];
#endif
			RowId row_id;
			float dummy[];
		} ref_d;
		struct MVecEntry
		{
			int32_t header_; /* set by SET_VARSIZE, 30bit for length and 2 for b00 */
			int32_t dim;	 /* number of dimensions, less than 2^24 */
			int32_t shape_size; /* number of shape size, less than 10 */
			int32_t shape[MAX_VECTOR_SHAPE_SIZE]; /* shape array */
			float data[];	 /* float vector */
		} vec_d;

	} MVec;

	static size_t MVEC_HEADER_SIZE = ((size_t) & (((MVec *)0)->vec_d.data));
	static size_t MVEC_POINTER_SIZE = ((size_t) & (((MVec *)0)->ref_d.dummy));

/* getter */
#define GET_MVEC_SIZE(dim) (MVEC_HEADER_SIZE + ((MAX_VECTOR_SHAPE_SIZE) * sizeof(int32_t)) + (((size_t)dim) * sizeof(float)))
#define GET_MVEC_ROWID(mvec) (((MVec *)mvec)->ref_d.row_id)
#define IS_MVEC_REF(mvec) (((MVec *)mvec)->ref_d.is_ref_tag)
#define GET_MVEC_DIM(mvec) (((MVec *)mvec)->vec_d.dim)
#define GET_MVEC_VAL(mvec, dim_i) (((MVec *)mvec)->vec_d.data[dim_i])
#define GET_MVEC_SHAPE_SIZE(mvec) (((MVec *)mvec)->vec_d.shape_size)
#define GET_MVEC_SHAPE_VAL(mvec, shape_i) (((MVec *)mvec)->vec_d.shape[shape_i])

/* setter */
#define SET_MVEC_VAL(mvec, dim, value) ((MVec *)mvec)->vec_d.data[dim] = ((float)value)
#define SET_MVEC_SHAPE_VAL(mvec, shape_index, value) ((MVec *)mvec)->vec_d.shape[shape_index] = ((int)value)
#define SET_MVEC_REF_ROWID(mvec, row_id)         \
	do                                           \
	{                                            \
		SET_VARSIZE(mvec, MVEC_POINTER_SIZE);    \
		((MVec *)mvec)->ref_d.row_id = row_id;   \
		((MVec *)mvec)->ref_d.is_ref_tag = true; \
	} while (0)
#define SET_MVEC_DIM_SHAPESIZE(mvec, dim, shape_size)                   \
	do                                                   \
	{                                                    \
		SET_VARSIZE(mvec, GET_MVEC_SIZE(dim));           \
		((MVec *)mvec)->ref_d.is_ref_tag = false;        \
		((MVec *)mvec)->vec_d.dim = dim;                 \
		((MVec *)mvec)->vec_d.shape_size = shape_size;   \
	} while (0)

/**
 * when form a tuple with mvec, it may be compressed to varattrib_1b
 * when dim is less than 30 (which means size of MVec except header is less than 127 byte)
 * so detoast it to varattrib_4b when read the datum
 */
#define DatumGetMVec(x_) ((MVec *)PG_DETOAST_DATUM(x_))
#define PG_GETARG_MVEC_P(n) DatumGetMVec(PG_GETARG_DATUM(n))

	MVec *new_mvec(int dim, int shape_size);
	MVec *new_mvec_ref(RowId row_id);
    void free_vector(MVec *vector);
	//MVec *parse_mvec_str(char *str);
	//void parse_mvec_shape_str(char *shape_str, int32_t *shape_size, int32_t *shape);
    void parse_vector_shape_str(char* shape_str, unsigned int* shape_size, int* shape);
    void parse_vector_str(char* str, unsigned int* dim, float* x, unsigned int* shape_size, int32_t* shape);
	bool shape_equal(MVec *left, MVec *right);

	extern volatile Oid mvec_type_id;

void mvec_to_str(MVec *mvec, std::string& str);
// tensor 与 Mvec的互相转换
MVec* tensor_to_vector(torch::Tensor& tensor);
torch::Tensor vector_to_tensor(MVec* vector);
#ifdef __cplusplus
}
#endif


