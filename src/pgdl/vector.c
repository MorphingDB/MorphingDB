#include "vector.h"
#include "postgres.h"


#include "catalog/pg_type_d.h"
#include "libpq/pqformat.h"
#include "utils/builtins.h"
#include "utils/array.h"



MVec*
new_mvec(int dim, int shape_size) {
	MVec *new_col = (MVec *) palloc(GET_MVEC_SIZE(dim));
	SET_MVEC_DIM_SHAPESIZE(new_col, dim, shape_size);
	return new_col;
} 

MVec*
new_mvec_ref(RowId row_id) {
	MVec *new_col = (MVec *) palloc(MVEC_POINTER_SIZE);
	SET_MVEC_REF_ROWID(new_col, row_id);
	return new_col;
}

void 
free_vector(MVec *vector)
{
    if(vector == NULL){
        return;
    }
    pfree(vector);
    vector = NULL;
}

static inline bool 
is_space(char ch)
{
    if (ch == ' ' ||
		ch == '\t' ||
		ch == '\n' ||
		ch == '\r' ||
		ch == '\v' ||
		ch == '\f')
		return true;
	return false;
}

static inline void 
skip_space(char** p_str)
{
    char* p_ch = *p_str;
    char ch = *p_ch;
    while (ch== ' ' ||
		ch == '\t' ||
		ch == '\n' ||
		ch == '\r' ||
		ch == '\v' ||
		ch == '\f') {
            p_ch++;
            ch = *p_ch;
        }
	*p_str = p_ch;
}

static const char* 
generate_space(int num) {
    int i = 0;
    StringInfoData  ret;
    initStringInfo(&ret);
    for (i = 0; i < num; i++)
        appendStringInfoChar(&ret, ' ');
    return ret.data;
}

static void 
print_parse_error(const char* err_msg, const char* whole_str, const char* err_pos) {
    const char* p_last_20 = (err_pos - 20);
    int last_words = (int)(p_last_20 - whole_str);
    if (last_words < 0) {
        p_last_20 = whole_str;
        last_words = err_pos - p_last_20;
    }
    const char* p_forward_20 = pnstrdup(err_pos, 20);
    int err_word_pos = (int)(err_pos - whole_str);

    bool need_prefix = p_last_20 != whole_str;
    bool need_suffix = strlen(p_forward_20) != strlen(err_pos);
    const char* hint_prefix = "error occur at pos (";
    int space_num = strlen(hint_prefix) + strlen("HINT:  ") + 5 + strlen("): \"") + last_words;
    if (need_prefix)
        space_num += 3;
    ereport(ERROR,
				(errmsg("invalid input: \"%s\": %s.", whole_str, err_msg), 
                 errhint("error occur at pos (%5d): \"%s%s%s%s\"\n%s^", 
                        err_word_pos, 
                        need_prefix ? "..." : "",
                        pnstrdup(p_last_20, last_words), p_forward_20, 
                        need_suffix ? "..." : "",
                        generate_space(space_num))));
}


/*
    parse vector shape
*/
void 
parse_vector_shape_str(char* shape_str, unsigned int* shape_size, int* shape)
{
    char* str_copy = pstrdup(shape_str);
    char* pt = NULL;
    char* end = NULL;

    while(is_space(*shape_str)){
        shape_str++;
    }

    if (*shape_str != '{'){
        ereport(ERROR,
				 errmsg("Vector shape must start with \"{\"."));
    }

    shape_str++;
    pt = strtok(shape_str, ",");
    end = pt;

    while (pt != NULL && *end != '}') {
        if (*shape_size == MAX_VECTOR_SHAPE_SIZE) {
            ereport(ERROR,
					(errmsg("vector shape cannot have more than %d", MAX_VECTOR_SHAPE_SIZE)));
        }

        while (is_space(*pt)){
            pt++;
        }

        if (*pt == '\0') {
            ereport(ERROR,
					(errmsg("invalid input syntax for type vector shape: \"%s\"", str_copy)));
        }

        shape[*shape_size] = strtof(pt, &end);

        (*shape_size)++;

        if(end == pt){
            ereport(ERROR,
					(errmsg("invalid input syntax for type vector shape: \"%s\"", str_copy)));
        }

        while(is_space(*end)){
            end++;
        }

        if (*end != '\0' && *end != '}'){
            ereport(ERROR,
					(errmsg("invalid input syntax for type vector shape: \"%s\"", str_copy)));
        }

        pt = strtok(NULL, ",");
    }

    if (end == NULL || *end != '}'){
        ereport(ERROR,
					(errmsg("malformed vector literal4: \"%s\"", str_copy)));
    }

    end++;

    while (is_space(*end)){
        end++;
    }

    for(pt = str_copy + 1; *pt != '\0'; pt++){
        if (pt[-1] == ',' && *pt == ','){
            ereport(ERROR,
					(errmsg("malformed vector literal5: \"%s\"", str_copy)));
        }
    }

    if(*shape_size < 1){
        ereport(ERROR,
					(errmsg("vector must have at least 1 dimension")));
    }

    pfree(str_copy);
}

void 
parse_vector_str(char* str, unsigned int* dim, float* x,
                 unsigned int* shape_size, int32* shape)
{
    char* str_copy = pstrdup(str);
    char* index = str_copy;
    char* pt = NULL;
    char* end = NULL;

    while(is_space(*str)){
        str++;
    }

    if (*str != '['){
        ereport(ERROR,
				 errmsg("Vector contents must start with \"[\"."));
    }

    str++;
    pt = strtok(str, ",");
    end = pt;

    while (pt != NULL && *end != ']') {
        if (*dim == MAX_VECTOR_DIM) {
            ereport(ERROR,
					(errmsg("vector cannot have more than %d dimensions", MAX_VECTOR_DIM)));
        }

        while (is_space(*pt)){
            pt++;
        }

        if (*pt == '\0') {
            ereport(ERROR,
					(errmsg("invalid input syntax for type vector: \"%s\"", str_copy)));
        }

        x[*dim] = strtof(pt, &end);

        (*dim)++;

        if(end == pt){
            ereport(ERROR,
					(errmsg("invalid input syntax for type vector: \"%s\"", str_copy)));
        }

        while(is_space(*end)){
            end++;
        }

        if (*end != '\0' && *end != ']'){
            ereport(ERROR,
					(errmsg("invalid input syntax for type vector: \"%s\"", str_copy)));
        }

        pt = strtok(NULL, ",");
    }

    if (end == NULL || *end != ']'){
        ereport(ERROR,
					(errmsg("malformed vector literal3: \"%s\"", str_copy)));
    }

    end++;

    while (is_space(*end)){
        end++;
    }

    switch (*end) {
        case '{':
            while(*str_copy != '{'){
                str_copy++;
            }
            parse_vector_shape_str(str_copy, shape_size, shape);
            str_copy = index;
            break;
        case '\0':
            *shape_size = 1;
            shape[0] = (int32)(*dim);
            break;
        default:
            ereport(ERROR,
					(errmsg("malformed vector literal1: \"%s\"", str_copy)));
    }

    for(pt = str_copy + 1; *pt != '\0'; pt++){
        if (pt[-1] == ',' && *pt == ','){
            ereport(ERROR,
					(errmsg("malformed vector literal2: \"%s\"", str_copy)));
        }
    }

    if(*dim < 1){
        ereport(ERROR,
					(errmsg("vector must have at least 1 dimension")));
    }

    pfree(str_copy);

}

bool
shape_equal(MVec* left, MVec* right)
{
    if(left == NULL || right == NULL){
        return false;
    }

    if(GET_MVEC_SHAPE_SIZE(left) != GET_MVEC_SHAPE_SIZE(right)){
        return false;
    }

    for(int i=0; i<GET_MVEC_SHAPE_SIZE(left); ++i){
        if(GET_MVEC_SHAPE_VAL(left, i) != GET_MVEC_SHAPE_VAL(right, i)){
            return false;
        }
    }
    return true;
}
