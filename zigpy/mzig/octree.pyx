# import cython as cy
from libcpp cimport bool as bool_t

cdef extern from "octree.h":
    int add(int a, int b)

    ctypedef struct arr2D_i8:
        char x, y; 
    ctypedef struct arr2D_u8:
        unsigned char x, y; 
    ctypedef struct arr2D_i32:
        int x, y; 
    ctypedef struct arr2D_f32:
        float x, y; 
    ctypedef struct arr2D_usize:
        size_t x, y; 
    ctypedef struct arr2D_bool:
        bool_t x, y; 

    ctypedef struct arr3D_i8:
        char x, y, z; 
    ctypedef struct arr2D_u8:
        unsigned char x, y, z; 
    ctypedef struct arr3D_i32:
        int x, y, z; 
    ctypedef struct arr3D_f32:
        float x, y, z; 
    ctypedef struct arr3D_usize:
        size_t x, y, z; 
    ctypedef struct arr3D_bool:
        bool_t x, y, z; 

    # ***********************************************************************
    # ******************************** BLOCK ********************************
    # ***********************************************************************

    ctypedef struct Block2D:
        pass
    ctypedef struct Block3D:
        pass

    int Block2D__lvl(Block2D* self);
    float Block2D__size(Block2D* self);
    bool_t Block2D__is_leaf(Block2D* self);
    bool_t Block2D__is_runned(Block2D* self);
    Block2D* Block2D__neighbour_diff_lvl(Block2D* self, arr2D_i8 direction);
    Block2D* Block2D__neighbour(Block2D* self, arr2D_i8 direction);
    arr2D_i8 Block2D__block_quadr(Block2D* self);
    size_t Block2D__block_quadr_idx(Block2D* self);
    Block2D* Block2D__child_quadr_idx(Block2D* self, size_t idx);
    Block2D* Block2D__child_quadr(Block2D* self, arr2D_i8 quadr);

    int Block3D__lvl(Block3D* self);
    float Block3D__size(Block3D* self);
    bool_t Block3D__is_leaf(Block3D* self);
    bool_t Block3D__is_runned(Block3D* self);
    Block3D* Block3D__neighbour_diff_lvl(Block3D* self, arr3D_i8 direction);
    Block3D* Block3D__neighbour(Block3D* self, arr3D_i8 direction);
    arr3D_i8 Block3D__block_quadr(Block3D* self);
    size_t Block3D__block_quadr_idx(Block3D* self);
    Block3D* Block3D__child_quadr_idx(Block3D* self, size_t idx);
    Block3D* Block3D__child_quadr(Block3D* self, arr3D_i8 quadr); 

    # ************************************************************************
    # ******************************** FOREST ********************************
    # ************************************************************************

    ctypedef struct Forest2D:
        size_t forest_ptr;
        int dim;
    ctypedef struct Forest3D:
        size_t forest_ptr;
        int dim;

    Forest2D Forest2D__new(arr2D_usize domain_size, arr2D_bool periodic_domain);
    void Forest2D__free(Forest2D* self);
    Block2D* Forest2D__block_from_idx(Forest2D* self, size_t idx);
    Block2D* Forest2D__block_from_pos(Forest2D* self, arr2D_f32 pos, char max_lvl);
    Block2D* Forest2D__block_from_quadr(Forest2D* self, arr2D_i32 pos_quadr, int quadr_lvl, char max_lvl);
    void Forest2D__divide_block(Forest2D* self, size_t block_idx);

    Forest3D Forest3D__new(arr3D_usize domain_size, arr3D_bool periodic_domain);
    void Forest3D__free(Forest3D* self);
    Block3D* Forest3D__block_from_idx(Forest3D* self, size_t idx);
    Block3D* Forest3D__block_from_pos(Forest3D* self, arr3D_f32 pos, char max_lvl);
    Block3D* Forest3D__block_from_quadr(Forest3D* self, arr3D_i32 pos_quadr, int quadr_lvl, char max_lvl);
    void Forest3D__divide_block(Forest3D* self, size_t block_idx);


cdef class PyForest2D:
    cdef Forest2D c_forest

    def __init__(self, domain_size: tuple[int, int], periodic_domain: tuple[bool, bool]) -> None:
        cdef arr2D_usize c_ds = arr2D_usize(domain_size[0], domain_size[1])
        cdef arr2D_bool c_periodic = arr2D_bool(periodic_domain[0], periodic_domain[1]) 
        print("In init", "cvt", c_ds, c_periodic)
        self.c_forest = Forest2D__new(c_ds, c_periodic)
        print("Outside call", c_ds, c_periodic, self.c_forest)
        ...

    def __dealloc__(self):
        print("freeing ", self.c_forest)
        Forest2D__free(&self.c_forest);
        print("freed", self.c_forest)


cpdef int py_add(int a, int b):
    """
    This function takes two Python integers (automatically converted to C int)
    and returns their sum (done using Python).
    """
    return a + b

def zig_add(int a, int b) -> int:
    """
    This function uses the C function `add` from "znassu.h" to add two integers.
    """
    return add(a, b)  # Explicitly use C types here
