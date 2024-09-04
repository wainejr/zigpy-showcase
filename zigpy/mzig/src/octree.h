#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {char x, y; } arr2D_i8;
typedef struct {unsigned char x, y; } arr2D_u8;
typedef struct {int x, y; } arr2D_i32;
typedef struct {float x, y; } arr2D_f32;
typedef struct {size_t x, y; } arr2D_usize;
typedef struct {bool x, y; } arr2D_bool;

typedef struct {char x, y, z; } arr3D_i8;
typedef struct {unsigned char x, y, z; } arr3D_u8;
typedef struct {int x, y, z; } arr3D_i32;
typedef struct {float x, y, z; } arr3D_f32;
typedef struct {size_t x, y, z; } arr3D_usize;
typedef struct {bool x, y, z; } arr3D_bool;


/*******************************************************************/
/****************************** BLOCK ******************************/
/*******************************************************************/

typedef struct {
    void* ptr;
} Block2D;

typedef struct {
    void* ptr;
} Block3D;

int Block2D__lvl(Block2D* self);
float Block2D__size(Block2D* self);
bool Block2D__is_leaf(Block2D* self);
bool Block2D__is_runned(Block2D* self);
Block2D* Block2D__neighbour_diff_lvl(Block2D* self, arr2D_i8 direction);
Block2D* Block2D__neighbour(Block2D* self, arr2D_i8 direction);
arr2D_i8 Block2D__block_quadr(Block2D* self);
size_t Block2D__block_quadr_idx(Block2D* self);
Block2D* Block2D__child_quadr_idx(Block2D* self, size_t idx);
Block2D* Block2D__child_quadr(Block2D* self, arr2D_i8 quadr);

int Block3D__lvl(Block3D* self);
float Block3D__size(Block3D* self);
bool Block3D__is_leaf(Block3D* self);
bool Block3D__is_runned(Block3D* self);
Block3D* Block3D__neighbour_diff_lvl(Block3D* self, arr3D_i8 direction);
Block3D* Block3D__neighbour(Block3D* self, arr3D_i8 direction);
arr3D_i8 Block3D__block_quadr(Block3D* self);
size_t Block3D__block_quadr_idx(Block3D* self);
Block3D* Block3D__child_quadr_idx(Block3D* self, size_t idx);
Block3D* Block3D__child_quadr(Block3D* self, arr3D_i8 quadr);

/********************************************************************/
/****************************** FOREST ******************************/
/********************************************************************/

typedef struct {
    size_t forest_ptr;
    int dim;
} Forest2D;

typedef struct {
    size_t forest_ptr;
    int dim;
} Forest3D;

Forest2D Forest2D__new(arr2D_usize domain_size, arr2D_bool periodic_domain);
void Forest2D__free(Forest2D* self);
Block2D* Forest2D__block_from_idx(Forest2D* self, size_t idx);
Block2D* Forest2D__block_from_pos(Forest2D* self, arr2D_f32 pos, char max_lvl);
Block2D* Forest2D__block_from_quadr(Forest2D* self, arr2D_i32 pos_quadr, int quadr_lvl, char max_lvl);
void Forest2D__divide_block(Forest2D* self, size_t block_idx);

Forest2D Forest3D__new(arr3D_usize domain_size, arr3D_bool periodic_domain);
void Forest3D__free(Forest3D* self);
Block3D* Forest3D__block_from_idx(Forest3D* self, size_t idx);
Block3D* Forest3D__block_from_pos(Forest3D* self, arr3D_f32 pos, char max_lvl);
Block3D* Forest3D__block_from_quadr(Forest3D* self, arr3D_i32 pos_quadr, int quadr_lvl, char max_lvl);
void Forest3D__divide_block(Forest3D* self, size_t block_idx);
