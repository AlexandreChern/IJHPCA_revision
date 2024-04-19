include("CUDA_kernels_second_new.jl")
include("interpolations.jl")
include("mms.jl")
using Arpack
include("utils.jl")

device!(1)


initialize_mg_struct_CUDA(mg_struct_CUDA, 1024, 1024, 10)
mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.b_mg[1][:]
mg_struct_CUDA.A_mg[2] * mg_struct_CUDA.b_mg[2][:]