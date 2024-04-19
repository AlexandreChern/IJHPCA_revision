include("CUDA_kernels_second_new.jl")
include("interpolations.jl")
include("mms.jl")
using Arpack
include("utils.jl")

device!(1)

initialize_mg_struct_CUDA(mg_struct_CUDA, 8192, 8192, 13)
# initialize_mg_struct_CUDA(mg_struct_CUDA, 1024, 1024, 10)
mfA_CUDA(mg_struct_CUDA.b_mg[1], mg_struct_CUDA, 1)
mfA_CUDA(mg_struct_CUDA.b_mg[2], mg_struct_CUDA, 2)
mfA_CUDA(mg_struct_CUDA.b_mg[3], mg_struct_CUDA, 3)
mfA_CUDA(mg_struct_CUDA.b_mg[4], mg_struct_CUDA, 4)
mfA_CUDA(mg_struct_CUDA.b_mg[5], mg_struct_CUDA, 5)
# mfA_CUDA(mg_struct_CUDA.b_mg[6], mg_struct_CUDA, 6)
# mfA_CUDA(mg_struct_CUDA.b_mg[7], mg_solver_CUDA, 7 )
