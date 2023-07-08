using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Plots
using CUDA
using Random
using Arpack
using BenchmarkTools
using DelimitedFiles
using AlgebraicMultigrid

# CUDA.device!(1)
CUDA.device!(2)



include("CUDA_kernels_second_new.jl")
include("interpolations.jl")
include("mms.jl")
include("utils.jl")


A_DDNN_16, b_DDNN_16, H_DDNN_16, H_DDNN_16_inv, metrics_DDNN_16 = create_ops(16,16)
factorization_benchmark_16 = @benchmark lu(A_DDNN_16)



A_CUDA_CSR = CUDA.CUSPARSE.CuSparseMatrixCSR(A_DDNN)
A_CUDA_CSC = CUDA.CUSPARSE.CuSparseMatrixCSC(A_DDNN)

a = 12


Base.summarysize(A_CUDA_CSR)
Base.summary(A_CUDA_CSR)

length(A_CUDA_CSR.nzVal)


(length(A_DDNN.nzval) * 2 )* 8
Base.summarysize(A_DDNN)



clear_mg_struct_CUDA(mg_struct_CUDA)

initialize_mg_struct_CUDA(mg_struct_CUDA, 512, 512, 8)
mg_solver_CUDA(mg_struct_CUDA, mg_struct_CUDA.b_mg[1]; nx=512, ny=512, n_level=3, v1=5, v2=5, v3=5, tolerence=1e-10, iter_algo_num=3,ω=1, ω_richardson=2/1000, max_mg_iterations=1, use_sbp=true, use_direct_sol=false,dynamic_richardson_ω=false)
mg_solver_CUDA_SpMV(mg_struct_CUDA, mg_struct_CUDA.b_mg[1]; nx=512, ny=512, n_level=3, v1=5, v2=5, v3=5, tolerence=1e-10, iter_algo_num=3,ω=1, ω_richardson=2/1000, max_mg_iterations=1, use_sbp=true, use_direct_sol=false,dynamic_richardson_ω=false)

mg_struct_CUDA.x_CUDA[1][:] .= 0;
mgcg_CUDA(mg_struct_CUDA; nx = 512, ny = 512, n_level=8, dynamic_richardson_ω=true, show_error=true)

mg_struct_CUDA.x_CUDA[1][:] .= 0;
mgcg_CUDA_SpMV(mg_struct_CUDA; nx = 512, ny = 512, n_level=8, dynamic_richardson_ω=true, show_error=true)


initialize_mg_struct_CUDA(mg_struct_CUDA, 64, 64, 5)
mg_struct_CUDA.x_CUDA[1][:] .= 0;
_, _, L2_errors_preconditioned = mgcg_CUDA(mg_struct_CUDA; nx = 64, ny = 64, n_level=5, dynamic_richardson_ω=true, show_error=true)
mg_struct_CUDA.x_CUDA[1][:] .= 0;
_, _, L2_errors_unpreconditioned = mgcg_CUDA(mg_struct_CUDA; nx = 64, ny = 64, n_level=5, dynamic_richardson_ω=true, show_error=true, precond=false)


mg_struct_CUDA.odata_mg[1] .= mg_struct_CUDA.b_mg[1]
@inbounds mg_struct_CUDA.odata_mg[1] .= mg_struct_CUDA.b_mg[1]

copyto!(mg_struct_CUDA.odata_mg[1], view(mg_struct_CUDA.b_mg[1],:,:))
copyto!(mg_struct_CUDA.odata_mg[1], mg_struct_CUDA.b_mg[1])

initialize_mg_struct_CUDA(mg_struct_CUDA, 1024, 1024, 9)
mgcg_CUDA(mg_struct_CUDA; nx = 1024, ny = 1024, n_level=9, dynamic_richardson_ω=true, show_error=true)

@benchmark let 
    mg_struct_CUDA.x_CUDA[1] .= 0; 
    mgcg_CUDA(mg_struct_CUDA; nx = 1024, ny = 1024, n_level=9, max_cg_iter=1024^2, dynamic_richardson_ω=true, precond=false, show_error=true)
end


@btime mgcg_CUDA(mg_struct_CUDA; nx = 1024, ny = 1024, n_level=9, dynamic_richardson_ω=true)


initialize_mg_struct_CUDA(mg_struct_CUDA, 2048, 2048, 10)

@btime mgcg_CUDA(mg_struct_CUDA; nx = 2048, ny = 2048, n_level=10, dynamic_richardson_ω=true)


CUDA.@profile for _ in 1:100
    mgcg_CUDA(mg_struct_CUDA; nx = 2048, ny = 2048, n_level=10, dynamic_richardson_ω=true)
end


@btime for _ in 1:10
    mgcg_CUDA(mg_struct_CUDA; nx = 2048, ny = 2048, n_level=10, max_cg_iter=10,dynamic_richardson_ω=true)
end


benchmark_SpMV = @benchmark CUDA.@sync mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.b_mg[1][:]


benchmark_mfA = @benchmark CUDA.@sync mfA_CUDA(mg_struct_CUDA.b_mg[1], mg_struct_CUDA, 1)


(sizeof(mg_struct_CUDA.b_mg[1]) + sizeof(mg_struct_CUDA.crr_mg[1] + mg_struct_CUDA.crs_mg[1] + mg_struct_CUDA.css_mg[1]) + sizeof(mg_struct_CUDA.odata_mg[1])) / (350 * 1e-6)

(Base.summarysize(mg_struct_CUDA.A_cpu_mg[1]) + sizeof(mg_struct_CUDA.b_mg[1])) / (954*1e-6)

(sizeof(mg_struct_CUDA.b_mg[1]) + sizeof(mg_struct_CUDA.crr_mg[1] + mg_struct_CUDA.crs_mg[1] + mg_struct_CUDA.css_mg[1]) + sizeof(mg_struct_CUDA.odata_mg[1]))


CUDA.@profile for _ in 1:10000
    mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.b_mg[1][:]
end


CUDA.@profile for _ in 1:10000
    mfA_CUDA(mg_struct_CUDA.b_mg[1], mg_struct_CUDA, 1)
end


initialize_mg_struct_CUDA(mg_struct_CUDA, 8192, 8192, 12)

CUDA.@profile for _ in 1:10000
    mfA_CUDA(mg_struct_CUDA.b_mg[1], mg_struct_CUDA, 1)
end


CUDA.@profile for _ in 1:10000
    mg_struct_CUDA.odata_mg[1][:] .= mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.b_mg[1][:]
end



# Testing 4096 by 4096
CUDA.@profile for _ in 1:10000
    mfA_CUDA(mg_struct_CUDA.b_mg[2], mg_struct_CUDA, 2)
end


CUDA.@profile for _ in 1:10000
    mg_struct_CUDA.odata_mg[2][:] .= mg_struct_CUDA.A_mg[2] * mg_struct_CUDA.b_mg[2][:]
end


# Testing 2048 by 2048
CUDA.@profile for _ in 1:10000
    mfA_CUDA(mg_struct_CUDA.b_mg[3], mg_struct_CUDA, 3)
end


CUDA.@profile for _ in 1:10000
    mg_struct_CUDA.odata_mg[3][:] .= mg_struct_CUDA.A_mg[3] * mg_struct_CUDA.b_mg[3][:]
end


# Testing 1024 by 1024
CUDA.@profile for _ in 1:10000
    mfA_CUDA(mg_struct_CUDA.b_mg[4], mg_struct_CUDA, 4)
end


CUDA.@profile for _ in 1:10000
    mg_struct_CUDA.odata_mg[4][:] .= mg_struct_CUDA.A_mg[4] * mg_struct_CUDA.b_mg[4][:]
end


# Testing 512 by 512
CUDA.@profile for _ in 1:10000
    mfA_CUDA(mg_struct_CUDA.b_mg[5], mg_struct_CUDA, 5)
end


CUDA.@profile for _ in 1:10000
    mg_struct_CUDA.odata_mg[5][:] .= mg_struct_CUDA.A_mg[5] * mg_struct_CUDA.b_mg[5][:]
end


# Testing 256 by 256
CUDA.@profile for _ in 1:10000
    mfA_CUDA(mg_struct_CUDA.b_mg[6], mg_struct_CUDA, 6)
end


CUDA.@profile for _ in 1:10000
    mg_struct_CUDA.odata_mg[6][:] .= mg_struct_CUDA.A_mg[6] * mg_struct_CUDA.b_mg[6][:]
end


# Profiling MGCG

# MGCG on 8192 by 8192
mg_struct_CUDA.x_CUDA[1] .= 0
mgcg_CUDA(mg_struct_CUDA; nx = 8192, ny = 8192, n_level=12, v1=v2=v3=1, dynamic_richardson_ω=true, max_cg_iter=20, show_error=true, precond=true)
mgcg_CUDA(mg_struct_CUDA; nx = 8192, ny = 8192, n_level=12, v1=v2=v3=1, dynamic_richardson_ω=true, max_cg_iter=8193^2, show_error=true, precond=false)


CUDA.@profile for _ in 1:100
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA(mg_struct_CUDA; nx = 8192, ny = 8192, n_level=12, dynamic_richardson_ω=true)
end





# MGCG on 4096 by 4096
clear_mg_struct_CUDA(mg_struct_CUDA)
mgcg_CUDA(mg_struct_CUDA; nx = 4096, ny = 4096, n_level=11, max_cg_iter=20, dynamic_richardson_ω=true, show_error=true)


CUDA.@profile for _ in 1:100
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA(mg_struct_CUDA; nx = 4096, ny = 4096, max_cg_iter=20, n_level=11, dynamic_richardson_ω=true)
end

# MGCG on 2048 by 2048

clear_mg_struct_CUDA(mg_struct_CUDA)
mgcg_CUDA(mg_struct_CUDA; nx = 2048, ny = 2048, n_level=10, dynamic_richardson_ω=true)


CUDA.@profile for _ in 1:100
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA(mg_struct_CUDA; nx = 2048, ny = 2048, n_level=10, dynamic_richardson_ω=true)
end


# MGCG on 1024 by 1024

clear_mg_struct_CUDA(mg_struct_CUDA)
mgcg_CUDA(mg_struct_CUDA; nx = 1024, ny = 1024, n_level=9, dynamic_richardson_ω=true)


CUDA.@profile for _ in 1:100
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA(mg_struct_CUDA; nx = 1024, ny = 1024, n_level=9, dynamic_richardson_ω=true)
end

# MGCG on 512 by 512

clear_mg_struct_CUDA(mg_struct_CUDA)
mgcg_CUDA(mg_struct_CUDA; nx = 512, ny = 512, n_level=8, dynamic_richardson_ω=true)


CUDA.@profile for _ in 1:100
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA(mg_struct_CUDA; nx = 512, ny = 512, n_level=8, dynamic_richardson_ω=true)
end


####### auto profiling for MGCG ################################
mgcg_iterations = []
cg_iterations = []

for k in 8:13
    nx = ny = 2^k

    # mg_struct_CUDA.x_CUDA[1] .= 0
    _, mgcg_count =  mgcg_CUDA(mg_struct_CUDA; nx = nx, ny = ny, n_level=k-1, dynamic_richardson_ω=true)

    CUDA.@profile for _ in 1:100
        mg_struct_CUDA.x_CUDA[1] .= 0
        _, mgcg_count =  mgcg_CUDA(mg_struct_CUDA; nx = nx, ny = ny, n_level=k-1, dynamic_richardson_ω=true)
    end
    run(`mv report1.nsys-rep $(nx)_mgcg.nsys-rep`)
    push!(mgcg_iterations, mgcg_count)

    # mg_struct_CUDA.x_CUDA[1] .= 0
    _, cg_count = mgcg_CUDA(mg_struct_CUDA; nx = nx, ny = ny, n_level=k-1, dynamic_richardson_ω=true,precond=false)
    CUDA.@profile for _ in 1:10
        mg_struct_CUDA.x_CUDA[1] .= 0
        _, cg_count = mgcg_CUDA(mg_struct_CUDA; nx = nx, ny = ny, n_level=k-1, dynamic_richardson_ω=true,precond=false)
    end
    run(`mv report1.nsys-rep $(nx)_cg.nsys-rep`)
    push!(cg_iterations, cg_count)

    if k == 13
        # file = "iterations.txt"
        # open(file, "w") do f
        #     for i in eachindex(mgcg_iterations)
        #         write(f,mgcg_iterations[i], "\t", cg_iterations[i])
        #     end
        # end
        open("iterations.txt", "w") do io
            writedlm(io, [mgcg_iterations, cg_iterations])
        end
    end

end


######## now do it for SpMV #######################################
mgcg_SpMV_iterations = []
cg_SpMV_iterations = []

# for k in 8:10
for k = 11:13
    nx = ny = 2^k

    # mg_struct_CUDA.x_CUDA[1] .= 0
    _, mgcg_count =  mgcg_CUDA_SpMV(mg_struct_CUDA; nx = nx, ny = ny, n_level=k-1, dynamic_richardson_ω=true)

    CUDA.@profile for _ in 1:100
        mg_struct_CUDA.x_CUDA[1] .= 0
        _, mgcg_count =  mgcg_CUDA_SpMV(mg_struct_CUDA; nx = nx, ny = ny, n_level=k-1, dynamic_richardson_ω=true)
    end
    run(`mv report1.nsys-rep $(nx)_mgcg_SpMV.nsys-rep`)
    push!(mgcg_SpMV_iterations, mgcg_count)

    # mg_struct_CUDA.x_CUDA[1] .= 0
    _, cg_count = mgcg_CUDA_SpMV(mg_struct_CUDA; nx = nx, ny = ny, n_level=k-1, dynamic_richardson_ω=true,precond=false)
    CUDA.@profile for _ in 1:2
        mg_struct_CUDA.x_CUDA[1] .= 0
        _, cg_count = mgcg_CUDA_SpMV(mg_struct_CUDA; nx = nx, ny = ny, n_level=k-1, dynamic_richardson_ω=true,precond=false)
    end
    run(`mv report1.nsys-rep $(nx)_cg_SpMV.nsys-rep`)
    push!(cg_SpMV_iterations, cg_count)

    if k == 13
        # file = "iterations.txt"
        # open(file, "w") do f
        #     for i in eachindex(mgcg_iterations)
        #         write(f,mgcg_iterations[i], "\t", cg_iterations[i])
        #     end
        # end
        open("iterations_SpMV.txt", "w") do io
            writedlm(io, [mgcg_SpMV_iterations, cg_SpMV_iterations])
        end
    end

end

##### end auto profiling #########################################


############

@benchmark for _ in 1:1
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA(mg_struct_CUDA; nx = 8192, ny = 8192, v1=v2=v3=5, max_cg_iter=20, n_level=12, dynamic_richardson_ω=true)
end

level = 1


Nr = mg_struct_CUDA.lnx_mg[level]
Ns = mg_struct_CUDA.lny_mg[level]

hr = 2/Nr
hs = 2/Ns

Nr1 = Nr + 1
Ns1 = Ns + 1

idata_crr = mg_struct_CUDA.crr_mg[level]
idata_css = mg_struct_CUDA.css_mg[level]
idata_crs = mg_struct_CUDA.crs_mg[level]

idata_ψ1 = mg_struct_CUDA.ψ1_mg[level]
idata_ψ2 = mg_struct_CUDA.ψ2_mg[level]

threads_1D = 256
blocks_1D = div(Ns1+threads_1D-1,threads_1D)

threads_2D = (16, 16)
blocks_2D = (div(Nr1 + threads_2D[1] - 1, threads_2D[1]), div(Ns1 + threads_2D[2] - 1, threads_2D[2]))


@cuda threads=threads_2D blocks=blocks_2D cuda_knl_2_x(hr, hs, mg_struct_CUDA.b_mg[1], Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, mg_struct_CUDA.odata_mg[level])


@benchmark CUDA.@sync @cuda threads=threads_2D blocks=blocks_2D cuda_knl_2_x(hr, hs, mg_struct_CUDA.b_mg[1], Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, mg_struct_CUDA.odata_mg[level])

@benchmark CUDA.@sync idata_crr .+ idata_css


CUDA.@profile for _ in 1:1000
        @cuda threads=threads_2D blocks=blocks_2D cuda_knl_2_x(hr, hs, mg_struct_CUDA.b_mg[1], Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, mg_struct_CUDA.odata_mg[level])
    end


let
    level = 1
    total_size = Base.sizeof(mg_struct_CUDA.b_mg[level]) + Base.sizeof(idata_crr) + Base.sizeof(idata_css) + Base.sizeof(idata_crs) + Base.sizeof(idata_ψ1) + Base.sizeof(idata_ψ2) + Base.sizeof(mg_struct_CUDA.odata_mg[1])
    total_size / (3.3*1e-3 * 1024^3) 
end



for k in 1:6
    CUDA.@profile for _ in 1:10000
        # mg_struct_CUDA.odata_mg[k] .= mg_struct_CUDA.b_mg[k]
        copyto!(mg_struct_CUDA.odata_mg[k], mg_struct_CUDA.b_mg[k])
    end
end

for k in 1:6
    CUDA.@profile for _ in 1:10000
        mg_struct_CUDA.odata_mg[k][:] .= mg_struct_CUDA.A_mg[k] * mg_struct_CUDA.b_mg[k][:]
    end
end



###################### Discretization error ################################

LU_factorizations = []
for k = 1:12
    A_lu = lu(mg_struct_CUDA.A_cpu_mg[end+1-k])
    push!(LU_factorizations, A_lu)
    b_array = Array(mg_struct_CUDA.b_mg[end+1-k][:])
    u_direct = A_lu \ b_array
    u_exact = Array(mg_struct_CUDA.u_exact[end+1-k])[:]
    discretization_error = sqrt((u_direct[:] - u_exact[:])' * mg_struct_CUDA.H_mg[end+1-k] * (u_direct[:] - u_exact[:]))
    @show mg_struct_CUDA.lnx_mg[end+1-k], discretization_error
end

direct_solve_benchmark_results = []
factorization_memory_allocs = []
for k = 1:12
    benchmark_factorization = @benchmark lu(mg_struct_CUDA.A_cpu_mg[end+1-k])
    push!(factorization_memory_allocs, benchmark_factorization)
    benchmark_direct = @benchmark LU_factorizations[$k] \ Array(mg_struct_CUDA.b_mg[end+1-$k][:])
    push!(direct_solve_benchmark_results, benchmark_direct)
end



#################### AMG as preconditioner ##################################

A = mg_struct_CUDA.A_cpu_mg[1]
b = Array(mg_struct_CUDA.b_mg[1])[:]
ml = ruge_stuben(A)

p = aspreconditioner(ml)
x_amg_cg, amg_cg_history = cg(A, b, Pl=p, log=true, reltol=1e-6)
x_cg, cg_history = cg(A, b, log=true, reltol=1e-6)

plot(amg_cg_history.data[:resnorm],xscale=:log10, yscale=:log10, label="residual AMGCG (Ruge-Stuben)", marker=(:circle,5))
plot!(cg_history.data[:resnorm], xscale=:log10, yscale=:log10, label="residual CG", marker=(:square,5))
xaxis!("Iterations (log10)")
yaxis!("Residuals (log10)")

savefig("cg_vs_amgcg_log10.png")



plot(amg_cg_history.data[:resnorm], yscale=:log10, label="residual AMGCG (Ruge-Stuben)", marker=(:circle,5))
plot!(cg_history.data[:resnorm], yscale=:log10, label="residual CG", marker=(:square,5))
xaxis!("Iterations (log10)")
yaxis!("Residuals (log10)")

savefig("cg_vs_amgcg.png")