using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Plots
using CUDA
using Random
using Arpack
using BenchmarkTools

include("CUDA_kernels_second_new.jl")
# include("../MF_kernels/second_order_kernels/second_mms_MF.jl")
include("mms.jl")
include("interpolations.jl")
# include("../GPU_kernels/second_order_kernels/CUDA_kernels_1D.jl")


mutable struct MG_CUDA
    A_mg
    b_mg
    A_cpu_mg
    λ_mins
    λ_maxs
    odata_mg
    odata_interior_mg
    odata_f1_mg 
    odata_f2_mg 
    odata_f3_mg
    odata_f4_mg   
    crr_mg
    css_mg
    crs_mg
    ψ1_mg
    ψ2_mg
    H_mg
    f_mg
    u_mg
    r_mg
    prol_fine_mg
    rest_mg
    prol_mg
    lnx_mg
    lny_mg
    u_exact

    # structs for mgcg_CUDA
    x_CUDA
    r_CUDA
    r_new_CUDA
    z_CUDA
    z_new_CUDA
    p_CUDA
end

mg_struct_CUDA = MG_CUDA([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])

function clear_mg_struct_CUDA(mg_struct_CUDA)
    println("Clearing mg_struct")
    mg_struct_CUDA.A_mg = []
    mg_struct_CUDA.A_cpu_mg = []
    mg_struct_CUDA.λ_mins = []
    mg_struct_CUDA.λ_maxs = []
    mg_struct_CUDA.b_mg = []
    mg_struct_CUDA.odata_mg = []
    mg_struct_CUDA.odata_interior_mg = []
    mg_struct_CUDA.odata_f1_mg = []
    mg_struct_CUDA.odata_f2_mg = []
    mg_struct_CUDA.odata_f3_mg = []
    mg_struct_CUDA.odata_f4_mg = []
    mg_struct_CUDA.crr_mg = []
    mg_struct_CUDA.css_mg = []
    mg_struct_CUDA.crs_mg = []
    mg_struct_CUDA.ψ1_mg = []
    mg_struct_CUDA.ψ2_mg = []
    mg_struct_CUDA.H_mg = []
    mg_struct_CUDA.f_mg = []
    mg_struct_CUDA.u_mg = []
    mg_struct_CUDA.r_mg = []
    mg_struct_CUDA.prol_fine_mg = []
    mg_struct_CUDA.rest_mg = []
    mg_struct_CUDA.prol_mg = []
    mg_struct_CUDA.lnx_mg = []
    mg_struct_CUDA.lny_mg = []
    mg_struct_CUDA.u_exact = []
    # clearing MGCG data
    mg_struct_CUDA.x_CUDA = []
    mg_struct_CUDA.r_CUDA = []
    mg_struct_CUDA.r_new_CUDA = []
    mg_struct_CUDA.z_CUDA = []
    mg_struct_CUDA.z_new_CUDA = []
    mg_struct_CUDA.p_CUDA = []
    println("mg_struct cleared")
end

function clear_urf_CUDA(mg_struct_CUDA)
    for i in 1:length(mg_struct_CUDA.u_mg)
        mg_struct_CUDA.u_mg[i] .= 0
        mg_struct_CUDA.r_mg[i] .= 0
        mg_struct_CUDA.f_mg[i] .= 0
        mg_struct_CUDA.prol_fine_mg[i] .= 0
    end
end


function initialize_mg_struct_CUDA(mg_struct_CUDA,nx,ny,n_level)
    println("clearing matrices")
    clear_mg_struct_CUDA(mg_struct_CUDA) # comment out if initialize_mg_struct works well
    println("Starting assembling matrices")

    A_mg = mg_struct_CUDA.A_mg
    A_cpu_mg = mg_struct_CUDA.A_cpu_mg
    b_mg = mg_struct_CUDA.b_mg
    odata_mg = mg_struct_CUDA.odata_mg
    odata_interior_mg = mg_struct_CUDA.odata_interior_mg
    odata_f1_mg = mg_struct_CUDA.odata_f1_mg
    odata_f2_mg = mg_struct_CUDA.odata_f2_mg
    odata_f3_mg = mg_struct_CUDA.odata_f3_mg
    odata_f4_mg = mg_struct_CUDA.odata_f4_mg
    crr_mg = mg_struct_CUDA.crr_mg
    css_mg = mg_struct_CUDA.css_mg
    crs_mg = mg_struct_CUDA.crs_mg
    ψ1_mg = mg_struct_CUDA.ψ1_mg
    ψ2_mg = mg_struct_CUDA.ψ2_mg
    H_mg = mg_struct_CUDA.H_mg
    f_mg = mg_struct_CUDA.f_mg
    u_mg = mg_struct_CUDA.u_mg
    r_mg = mg_struct_CUDA.r_mg
    prol_fine_mg = mg_struct_CUDA.prol_fine_mg
    rest_mg = mg_struct_CUDA.rest_mg
    prol_mg = mg_struct_CUDA.prol_mg
    lnx_mg = mg_struct_CUDA.lnx_mg
    lny_mg = mg_struct_CUDA.lny_mg

    if isempty(A_mg)
        for k in 1:n_level
            nx,ny = nx,ny
            hx,hy = 1/nx, 1/ny
            if k == 1
                A_DDNN, b_DDNN, H_DDNN, H_DDNN_inv, metrics_DDNN = create_ops(nx,ny)
                push!(A_cpu_mg, A_DDNN)
                push!(A_mg, CUDA.CUSPARSE.CuSparseMatrixCSR(A_DDNN))
                push!(b_mg, CuArray(reshape(b_DDNN,nx+1,ny+1)))
                push!(H_mg,H_DDNN)
                push!(f_mg, CuArray(reshape(b_DDNN,nx+1,ny+1)))
                dx = 1.0 ./nx
                dy = 1.0 ./ny
                xs = 0:dx:1
                ys = 0:dy:1
                x = metrics_DDNN.coord[1]
                y = metrics_DDNN.coord[2]
                u_exact = ue(x,y)
                push!(mg_struct_CUDA.u_exact, CuArray(u_exact))
            else
                A_DDNN, b_DDNN, H_DDNN, H_DDNN_inv, metrics_DDNN = create_ops(nx, ny)
                push!(A_cpu_mg, A_DDNN)
                push!(A_mg,  CUDA.CUSPARSE.CuSparseMatrixCSR(A_DDNN))
                push!(b_mg, CuArray(reshape(b_DDNN,nx+1,ny+1)))
                push!(H_mg, H_DDNN)
                push!(f_mg, CuArray(zeros(nx+1, ny+1)))
                x = metrics_DDNN.coord[1]
                y = metrics_DDNN.coord[2]
                u_exact = ue(x,y)
                push!(mg_struct_CUDA.u_exact, CuArray(u_exact))
            end
            push!(u_mg, CuArray(zeros(nx+1, ny+1)))
            push!(odata_mg, CuArray(zeros(nx+1, ny+1)))
            push!(odata_interior_mg, CuArray(zeros(nx+1, ny+1)))

            push!(odata_f1_mg, CuArray(zeros(nx+1, ny+1)))
            push!(odata_f2_mg, CuArray(zeros(nx+1, ny+1)))
            push!(odata_f3_mg, CuArray(zeros(nx+1, ny+1)))
            push!(odata_f4_mg, CuArray(zeros(nx+1, ny+1)))


            # push!(odata_f1_mg, CuArray(zeros(3, ny+1)))
            # push!(odata_f2_mg, CuArray(zeros(3, ny+1)))
            # push!(odata_f3_mg, CuArray(zeros(nx+1, 3)))
            # push!(odata_f4_mg, CuArray(zeros(nx+1, 3)))

            push!(crr_mg, CuArray(metrics_DDNN.crr))
            push!(css_mg, CuArray(metrics_DDNN.css))
            push!(crs_mg, CuArray(metrics_DDNN.crs))

            l = 2
            ψmin_r = reshape(metrics_DDNN.crr,nx+1,ny+1)
            ψ1 = ψmin_r[  1, :]

            l = 2
            for k = 2:l
                ψ1 = min.(ψ1, ψmin_r[k, :])
            end

            idata_ψ1 = CuArray(ψ1)
            push!(ψ1_mg,idata_ψ1)

            ψ2 = ψmin_r[ny+1, :]
            
            for k = 2:l
                ψ2 = min.(ψ2, ψmin_r[nx+1+1-k, :])
            end
            idata_ψ2 = CuArray(ψ2)
            push!(ψ2_mg,idata_ψ2)



            push!(r_mg, CuArray(zeros(nx+1, ny+1)))
            push!(prol_fine_mg, CuArray(zeros(nx+1, ny+1)))
            push!(rest_mg, CUDA.CUSPARSE.CuSparseMatrixCSR(restriction_matrix_v2(nx,ny,div(nx,2),div(ny,2))))
            push!(prol_mg, CUDA.CUSPARSE.CuSparseMatrixCSR(prolongation_matrix_v2(nx,ny,div(nx,2),div(ny,2))))
            push!(lnx_mg, nx)
            push!(lny_mg, ny)
            nx, ny = div(nx,2), div(ny,2)
            hx, hy = 2*hx, 2*hy
        end
    end
    get_lams(mg_struct_CUDA)

    push!(mg_struct_CUDA.x_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.r_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.r_new_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.z_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.z_new_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.p_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
end


function get_lams(mg_struct_CUDA)
    # TO DO, get 
    empty!(mg_struct_CUDA.λ_mins)
    empty!(mg_struct_CUDA.λ_maxs)
    # empty!(mg_struct.αs)
    # empty!(mg_struct.δs)
    reverse_Amg = reverse(mg_struct_CUDA.A_cpu_mg)
    if size(reverse_Amg[1])[1] > 289
        println("The minimal A matrix is too large for λ_min calculation")
        return 0
    end
    for k in eachindex(reverse_Amg)
        # lam_max, v_max = eigs(reverse_Amg[k], nev=1, which=:LR)
        # lam_max = real(lam_max[1]) # try different formulations
        if size(reverse_Amg[k])[1] <= 1089 # nx <= 32
        # if size(reverse_Amg[k])[1] <= 4225 # nx <= 64 not consistent, sometimes work
            lam_min, v_min = eigs(reverse_Amg[k], nev=1, which=:SR)
            lam_min = real(lam_min[1])
            # @show lam_min

            lam_max, v_max = eigs(reverse_Amg[k], nev=1, which=:LR)
            lam_max = real(lam_max[1]) # try different formulations
        else
            lam_min = mg_struct_CUDA.λ_mins[1] / 4
            # @show mg_struct_CUDA.λ_mins[1]
            # @show lam_min
            if size(reverse_Amg[k])[1] <= 16641
                lam_max, v_max = eigs(reverse_Amg[k], nev=1, which=:LR)
                lam_max = real(lam_max[1]) # try different formulations
            else
                lam_max = mg_struct_CUDA.λ_maxs[1] + (mg_struct_CUDA.λ_maxs[1] - mg_struct_CUDA.λ_maxs[2]) * 0.6
            end
            # @show mg_struct_CUDA.λ_maxs[1]
        end
        pushfirst!(mg_struct_CUDA.λ_mins, lam_min)
        pushfirst!(mg_struct_CUDA.λ_maxs, lam_max)
        # α = (lam_min + lam_max) / 2
        # δ = α - 0.99 * lam_min
        # pushfirst!(mg_struct.αs, α)
        # pushfirst!(mg_struct.δs, δ)
    end 
    # return lam_mins, lam_maxs, αs, δs
end




function mfA_CUDA(idata_x,mg_struct_CUDA,level)

    odata_interior = mg_struct_CUDA.odata_interior_mg[level]
    odata_f1 = mg_struct_CUDA.odata_f1_mg[level]
    odata_f2 = mg_struct_CUDA.odata_f2_mg[level]
    odata_f3 = mg_struct_CUDA.odata_f3_mg[level]
    odata_f4 = mg_struct_CUDA.odata_f4_mg[level]
    



    
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

    # odata_interior .= 0

   
    # @cuda threads=threads_2D blocks=blocks_2D cuda_knl_2_x_interior(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, odata_interior)

    # @cuda threads=threads_2D blocks=blocks_2D cuda_knl_2_x(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata_interior)

    @cuda threads=threads_2D blocks=blocks_2D cuda_knl_2_x(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, mg_struct_CUDA.odata_mg[level])



    # using smaller matrices for odata_f1, odata_f2, odata_f3 odata_f3
   

    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f1_1D(hr, hs, view(idata_x,1:3,:), Nr1, Ns1, view(idata_crr,1:3,:), view(idata_css,1:3,:), view(idata_crs,1:3,:), idata_ψ1, idata_ψ2, view(odata_f1,1:3,:))

    # partial idata and odata values
    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f1_1D(hr, hs, idata_f1, Nr1, Ns1, idata_crr_f1, idata_css_f1, idata_crs_f1, idata_ψ1, idata_ψ2, odata_f1_f1)

    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f2_1D(hr, hs, view(idata_x,Nr1-2:Nr1,:), Nr1, Ns1, view(idata_crr,Nr1-2:Nr1,:), view(idata_css,Nr1-2:Nr1,:), view(idata_crs,Nr1-2:Nr1,:), idata_ψ1, idata_ψ2, view(odata_f2,1:3,:))
    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f3_1D(hr, hs, view(idata_x,:,1:3), Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, view(odata_f3,:,1:3))
    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f4_1D(hr, hs, view(idata_x,:,Ns1-2:Ns1), Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata_f4)
    # copyto!(view(odata_interior, 1:3, :), view(odata_interior, 1:3,:) .+ odata_f1)
    # copyto!(view(odata_interior, Nr1-2:Nr1,:), view(odata_interior,Nr1-2:Nr1,:) .+ odata_f2)
    # # synchronize()
    # copyto!(view(odata_interior, :, 1:3), view(odata_interior, :, 1:3) .+ odata_f3)
    # copyto!(view(odata_interior, :, Ns1-2:Ns1), view(odata_interior, :, Ns1-2:Ns1) .+ odata_f4)
    # # synchronize()
    # mg_struct_CUDA.odata_mg[level] .= odata_interior 


    # when odata_f1 odata_f2 odata_f3 odata_f4 have the same size of odata_mg
    
    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f1(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata_f1)
    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f2(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata_f2)
    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f3(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata_f3)
    # @cuda blocks=blocks_1D threads=threads_1D cuda_knl_2_x_f4(hr, hs, idata_x, Nr1, Ns1, idata_crr, idata_css, idata_crs, idata_ψ1, idata_ψ2, odata_f4)
    # mg_struct_CUDA.odata_mg[level] .= odata_interior .+ odata_f1 .+ odata_f2 .+ odata_f3 .+ odata_f4

    # mg_struct_CUDA.odata_mg[level] .= odata_interior
end



# initialize_mg_struct_CUDA(mg_struct_CUDA,48,48,3)

# let 
#     Random.seed!(0)
#     x = rand((49)*(49))
#     idata_x = CuArray(reshape(x,49,49))

#     mfA_CUDA(idata_x, mg_struct_CUDA, 1)

#     idata_x2 = CuArray(randn(25,25))
#     mfA_CUDA(idata_x, mg_struct_CUDA, 2)
# end

function mfA_H_inverse(idata_x, mg_struct_CUDA, level)
    Nr = mg_struct_CUDA.lnx_mg[level]
    Ns = mg_struct_CUDA.lny_mg[level]

    hr = 2/Nr
    hs = 2/Ns

    Nr1 = Nr + 1
    Ns1 = Ns + 1

    threads_1D = 256
    blocks_1D = div(Ns1+threads_1D-1,threads_1D)

    threads_2D = (16, 16)
    blocks_2D = (div(Nr1 + threads_2D[1] - 1, threads_2D[1]), div(Ns1 + threads_2D[2] - 1, threads_2D[2]))

    @cuda threads=threads_2D blocks=blocks_2D cuda_knl_2_H_inverse(hr, hs, idata_x, Nr1, Ns1, mg_struct_CUDA.odata_mg[level])
end

function mfA_H(idata_x, mg_struct_CUDA, level)
    Nr = mg_struct_CUDA.lnx_mg[level]
    Ns = mg_struct_CUDA.lny_mg[level]

    hr = 2/Nr
    hs = 2/Ns

    Nr1 = Nr + 1
    Ns1 = Ns + 1

    threads_1D = 256
    blocks_1D = div(Ns1+threads_1D-1,threads_1D)

    threads_2D = (16, 16)
    blocks_2D = (div(Nr1 + threads_2D[1] - 1, threads_2D[1]), div(Ns1 + threads_2D[2] - 1, threads_2D[2]))

    @cuda threads=threads_2D blocks=blocks_2D cuda_knl_2_H(hr, hs, idata_x, Nr1, Ns1, mg_struct_CUDA.odata_mg[level])
end



function mg_solver_CUDA(mg_struct_CUDA, f_in; nx=64, ny=64, n_level=3, v1=5, v2=5, v3=5, tolerence=1e-10, iter_algo_num=3,ω=1, ω_richardson=2/1000, max_mg_iterations=1, use_sbp=true, use_direct_sol=false,dynamic_richardson_ω=false)
    if isempty(mg_struct_CUDA.A_mg)
        initialize_mg_struct_CUDA(mg_struct_CUDA,nx,ny,n_level)
    end
    clear_urf_CUDA(mg_struct_CUDA)

    # @assert nx == mg_struct_CUDA.lnx_mg[1]
    # @assert ny == mg_struct_CUDA.lny_mg[1]

    mg_struct_CUDA.f_mg[1][:] .= copy(f_in)[:]
    # mg_struct_CUDA.u_mg[1][:] .= CuArray(zeros(nx+1, ny+1))[:]
    # mg_struct_CUDA.r_mg[1][:] .= CuArray(zeros(nx+1, ny+1))[:]

    iter_algos = ["SOR","jacobi","richardson"]
    iter_algo = iter_algos[iter_algo_num]

    mfA_CUDA(mg_struct_CUDA.u_mg[1], mg_struct_CUDA, 1)
    mg_struct_CUDA.r_mg[1][:] = mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.odata_mg[1][:]

    # init_rms = compute_l2norm(nx,ny,mg_struct_CUDA.r_mg[1])

    mg_iter_count = 0

    if nx < (2^n_level)
        println("Number of levels exceeds the possible number.")
    end

    # Allocate matrix for storage at fine level
    # residual at fine level is already defined at global level
    # prol_fine = CuArray(zeros(Float64, mg_struct_CUDA.lnx_mg[1]+1, mg_struct_CUDA.lny_mg[1]+1))
    
    # temporary residual which is restricted to coarse mesh error
    # the size keeps on changing

    # temp_residual = CuArray(zeros(Float64, mg_struct_CUDA.lnx_mg[1]+1, mg_struct_CUDA.lny_mg[1]+1))

    for iteration_count = 1:max_mg_iterations
        mg_iter_count += 1

        if iter_algo == "richardson"
            if dynamic_richardson_ω == true
                ω_richardson = 2 / (mg_struct_CUDA.λ_mins[1] + mg_struct_CUDA.λ_maxs[1])
            end
            for i in 1:v1
                mfA_CUDA(mg_struct_CUDA.u_mg[1],mg_struct_CUDA,1)
                mg_struct_CUDA.u_mg[1][:] .+= ω_richardson * (mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.odata_mg[1][:])
            end
        elseif iter_algo == "SOR" #very slow for GPU arrays
            mg_struct_CUDA.u_mg[1][:] .= sor!(mg_struct_CUDA.u_mg[1][:],mg_struct_CUDA.A_mg[1],mg_struct_CUDA.f_mg[1][:],ω;maxiter=1)
        end

        # mg_struct_CUDA.r_mg[1][:] .= mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.u_mg[1][:]
        mfA_CUDA(mg_struct_CUDA.u_mg[1], mg_struct_CUDA, 1)
        mg_struct_CUDA.r_mg[1][:] .= mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.odata_mg[1][:]


        for k = 2:n_level
            if k == 2
                mg_struct_CUDA.r_mg[k-1] = mg_struct_CUDA.r_mg[1]
            else
                mfA_CUDA(mg_struct_CUDA.u_mg[k-1],mg_struct_CUDA,k-1)
                mg_struct_CUDA.r_mg[k-1][:] .= mg_struct_CUDA.f_mg[k-1][:] .- mg_struct_CUDA.odata_mg[k-1][:]
            end
            # mg_struct_CUDA.f_mg[k][:] .= mg_struct_CUDA.rest_mg[k-1] * CuArray((mg_struct_CUDA.H_mg[k-1] \ Array(mg_struct_CUDA.r_mg[k-1][:])))

            mfA_H_inverse(mg_struct_CUDA.r_mg[k-1],mg_struct_CUDA,k-1)
            # mg_struct_CUDA.f_mg[k][:] .= mg_struct_CUDA.rest_mg[k-1] * mg_struct_CUDA.odata_mg[k-1][:]

            ## matrix-free restriction # needs to fix matrix_free_restriction_2d_GPU code
            matrix_free_restriction_2d_GPU(mg_struct_CUDA.odata_mg[k-1], mg_struct_CUDA.f_mg[k])
            
            # mg_struct_CUDA.f_mg[k][:] = CUDA.CUSPARSE.CuSparseMatrixCSR(mg_struct_CUDA.H_mg[k]) * mg_struct_CUDA.f_mg[k][:]
            mfA_H(mg_struct_CUDA.f_mg[k],mg_struct_CUDA,k)
            mg_struct_CUDA.f_mg[k][:] .= mg_struct_CUDA.odata_mg[k][:]

            if k < n_level
               
                if iter_algo == "richardson"
                    if dynamic_richardson_ω == true
                        ω_richardson = 2 / (mg_struct_CUDA.λ_mins[k] + mg_struct_CUDA.λ_maxs[k])
                    end
                    for i in 1:v1
                        mfA_CUDA(mg_struct_CUDA.u_mg[k],mg_struct_CUDA,k)
                        mg_struct_CUDA.u_mg[k][:] .+= ω_richardson * (mg_struct_CUDA.f_mg[k][:] .- mg_struct_CUDA.odata_mg[k][:])
                    end
                elseif iter_algo == "SOR"
                    mg_struct_CUDA.u_mg[k][:] .= sor!(mg_struct_CUDA.u_mg[k][:],mg_struct_CUDA.A_mg[k],mg_struct_CUDA.f_mg[k][:],ω;maxiter=1)
                end
            elseif k == n_level
                if use_direct_sol == true
                    continue
                else
                    if iter_algo == "richardson"
                        if dynamic_richardson_ω == true
                            ω_richardson = 2 / (mg_struct_CUDA.λ_mins[k] + mg_struct_CUDA.λ_maxs[k])
                        end
                        for i in 1:v2
                            mfA_CUDA(mg_struct_CUDA.u_mg[k],mg_struct_CUDA,k)
                            mg_struct_CUDA.u_mg[k][:] .+= ω_richardson * (mg_struct_CUDA.f_mg[k][:] .- mg_struct_CUDA.odata_mg[k][:])
                        end
                    elseif iter_algo == "SOR"
                        mg_struct_CUDA.u_mg[k][:] .= sor!(mg_struct_CUDA.u_mg[k][:],mg_struct_CUDA.A_mg[k],mg_struct_CUDA.f_mg[k][:],ω;maxiter=1)
                    end
                    # @show norm(mg_struct_CUDA.u_mg[k])
                end
            end
            # @show k, norm(mg_struct_CUDA.r_mg[k])
        end

        # ascending from the coarsest grid to the finest grid
        for k = n_level:-1:2
            # prol_fine = CuArray(zeros(Float64, mg_struct_CUDA.lnx_mg[k-1]+1, mg_struct_CUDA.lny_mg[k-1]+1))

            # # prol_fine[:] = mg_struct_CUDA.prol_mg[k-1] * mg_struct_CUDA.u_mg[k][:]

            # ## matrix-free prolongation
            # matrix_free_prolongation_2d_GPU(mg_struct_CUDA.u_mg[k], prol_fine)

            # # updating u_mg
            # mg_struct_CUDA.u_mg[k-1] .+= prol_fine


            matrix_free_prolongation_2d_GPU(mg_struct_CUDA.u_mg[k], mg_struct_CUDA.prol_fine_mg[k-1])
            mg_struct_CUDA.u_mg[k-1] .+= mg_struct_CUDA.prol_fine_mg[k-1]

            
            if iter_algo == "richardson"
                if dynamic_richardson_ω == true
                    ω_richardson = 2 / (mg_struct_CUDA.λ_mins[k-1] + mg_struct_CUDA.λ_maxs[k-1])
                end
                for i in 1:v3
                    mfA_CUDA(mg_struct_CUDA.u_mg[k-1],mg_struct_CUDA,k-1)
                    mg_struct_CUDA.u_mg[k-1][:] .+= ω_richardson * (mg_struct_CUDA.f_mg[k-1][:] .- mg_struct_CUDA.odata_mg[k-1][:])
                end
            elseif iter_algo == "SOR"
                mg_struct_CUDA.u_mg[k-1][:] .= sor!(mg_struct_CUDA.u_mg[k-1][:],mg_struct_CUDA.A_mg[k-1],mg_struct_CUDA.f_mg[k-1][:],ω;maxiter=1)
            end
            
        end
        mfA_CUDA(mg_struct_CUDA.u_mg[1], mg_struct_CUDA, 1)
        mg_struct_CUDA.r_mg[1][:] .= mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.odata_mg[1][:]

        # rms = compute_l2norm(mg_struct_CUDA.lnx_mg[1],mg_struct_CUDA.lny_mg[1],mg_struct_CUDA.r_mg[1])
        
        # L2_error = norm((mg_struct_CUDA.u_mg[1][:] - mg_struct_CUDA.u_exact[1][:])' * mg_struct_CUDA.H_mg[1] * (mg_struct_CUDA.u_mg[1][:] - mg_struct_CUDA.u_exact[1][:]))
        # @show iteration_count, rms, error
        # @show iteration_count, L2_error

        # comment out for GPU code
    end
    # return mg_struct_CUDA.u_mg[1]
end




if isdefined(:main, :mg_solver)
    initialize_mg_struct(mg_struct, 48, 48, 3)
    mg_solver(mg_struct, Array(f_in), nx=48, ny=48, n_level=3, v1=v2=v3=5, iter_algo_num=5, ω=2/1000, maximum_iterations=10)
end


function mgcg_CUDA(mg_struct_CUDA;nx=64,ny=64,n_level=3,v1=5,v2=5,v3=5, ω=1.0, ω_richardson=2/1000, max_cg_iter=10, max_mg_iterations=1,iter_algo_num=3, precond=true,dynamic_richardson_ω=false)
    if nx != mg_struct_CUDA.lnx_mg[1]
        clear_mg_struct_CUDA(mg_struct_CUDA)
        initialize_mg_struct_CUDA(mg_struct_CUDA, nx, ny, n_level)
    end
   
    # x_CUDA = CuArray(zeros(nx+1,ny+1))
    # r_CUDA = CuArray(zeros(size(x_CUDA)))

    # A, b = create_ops(nx,ny)
    # A_CUDA = CUDA.CUSPARSE.CuSparseMatrixCSR(A)


    # A_CUDA = mg_struct_CUDA.A_mg[1]
    # b_CUDA = CuArray(mg_struct_CUDA.b_mg[1][:])

    # r_CUDA[:] = b_CUDA - A_CUDA * x_CUDA[:]
    mfA_CUDA(mg_struct_CUDA.x_CUDA[1],mg_struct_CUDA,1)

    # r_CUDA[:] .= b_CUDA[:] - mg_struct_CUDA.odata_mg[1][:]
    mg_struct_CUDA.r_CUDA[1][:] .= mg_struct_CUDA.b_mg[1][:] .- mg_struct_CUDA.odata_mg[1][:]

    init_rms = norm(mg_struct_CUDA.r_CUDA[1])
    # z_CUDA = CuArray(zeros(nx+1,ny+1))

    if precond == true
        mg_solver_CUDA(mg_struct_CUDA, mg_struct_CUDA.r_CUDA[1], n_level = n_level, v1=v1,v2=v2,v3=v3, max_mg_iterations=max_mg_iterations, nx = nx, ny = ny, iter_algo_num=iter_algo_num, dynamic_richardson_ω=dynamic_richardson_ω)
        mg_struct_CUDA.z_CUDA[1] .= mg_struct_CUDA.u_mg[1]
    else
        mg_struct_CUDA.z_CUDA[1][:] .= mg_struct_CUDA.r_CUDA[1][:]
    end

    # p_CUDA = CuArray(zeros(size(r_CUDA)))
    # p_CUDA .= z_CUDA
    mg_struct_CUDA.p_CUDA[1] .= mg_struct_CUDA.z_CUDA[1]

    counter = 0
    for k in 1:max_cg_iter
        mfA_CUDA(mg_struct_CUDA.p_CUDA[1],mg_struct_CUDA,1)

        # α = dot(r_CUDA[:],z_CUDA[:]) / (dot(p_CUDA[:],A_CUDA * p_CUDA[:]))
        α = dot(mg_struct_CUDA.r_CUDA[1][:], mg_struct_CUDA.z_CUDA[1][:]) / (dot(mg_struct_CUDA.p_CUDA[1][:],mg_struct_CUDA.odata_mg[1]))

        mg_struct_CUDA.x_CUDA[1] .+= α .* mg_struct_CUDA.p_CUDA[1]


        # r_new_CUDA = r_CUDA[:] .- α * A_CUDA * p_CUDA[:]
        mg_struct_CUDA.r_new_CUDA[1][:] = mg_struct_CUDA.r_CUDA[1][:] .- α * mg_struct_CUDA.odata_mg[1][:]

        norm_v_initial_norm = norm(mg_struct_CUDA.r_new_CUDA[1]) / init_rms
        # @show k, norm(x_CUDA[:] - mg_struct_CUDA.u_exact[1][:])
        
        # L2_error = sqrt((x_CUDA[:] - mg_struct_CUDA.u_exact[1][:])' * CUDA.CUSPARSE.CuSparseMatrixCSR(mg_struct_CUDA.H_mg[1]) * (x_CUDA[:] - mg_struct_CUDA.u_exact[1][:]) )

        mfA_H((mg_struct_CUDA.x_CUDA[1][:] - mg_struct_CUDA.u_exact[1][:]),mg_struct_CUDA,1)
        L2_error = sqrt(dot((mg_struct_CUDA.x_CUDA[1][:] - mg_struct_CUDA.u_exact[1][:])', mg_struct_CUDA.odata_mg[1]))

        # @show k, norm_v_initial_norm, L2_error

        # println("")

        if norm(mg_struct_CUDA.r_new_CUDA[1]) < 1e-8 * init_rms
            break
        end

        if precond == true
            mg_solver_CUDA(mg_struct_CUDA, mg_struct_CUDA.r_new_CUDA[1], n_level = n_level, v1=v1,v2=v2,v3=v3, max_mg_iterations=max_mg_iterations, nx = nx, ny = ny, iter_algo_num=iter_algo_num, dynamic_richardson_ω=dynamic_richardson_ω)
            mg_struct_CUDA.z_new_CUDA[1] .= mg_struct_CUDA.u_mg[1]
        else
            mg_struct_CUDA.z_new_CUDA[1] .= copy(mg_struct_CUDA.r_new_CUDA[1])
        end

        β = dot(mg_struct_CUDA.r_new_CUDA[1][:], mg_struct_CUDA.z_new_CUDA[1][:]) / (dot(mg_struct_CUDA.r_CUDA[1][:],mg_struct_CUDA.z_CUDA[1][:]))

        mg_struct_CUDA.p_CUDA[1][:] .= mg_struct_CUDA.z_new_CUDA[1][:] .+ β * mg_struct_CUDA.p_CUDA[1][:]
        mg_struct_CUDA.z_CUDA[1][:] .= mg_struct_CUDA.z_new_CUDA[1][:]
        mg_struct_CUDA.r_CUDA[1][:] .= mg_struct_CUDA.r_new_CUDA[1][:]
        counter += 1

        # η_alg = dot((x_CUDA[:]), mg_struct_CUDA.A_mg[1] * (x_CUDA[:]))
        # η_tot = dot((mg_struct_CUDA.u_exact[1])[:], mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.u_exact[1][:])
        # @show k, η_alg, η_tot, η_alg / η_tot
    end
    return mg_struct_CUDA.x_CUDA[1], counter     
end

initialize_mg_struct_CUDA(mg_struct_CUDA,32,32,5)
f_in = copy(mg_struct_CUDA.b_mg[1])
result_mg = mg_solver_CUDA(mg_struct_CUDA,f_in,nx=32,ny=32,n_level=4,iter_algo_num=3,max_mg_iterations=10,ω_richardson=2/1000)



initialize_mg_struct_CUDA(mg_struct_CUDA,64,64,6)
mg_solver_CUDA(mg_struct_CUDA,mg_struct_CUDA.b_mg[1], n_level=6,v1=10,v2=10,v3=10,iter_algo_num=3, max_mg_iterations=10,ω_richardson=2/1500,use_direct_sol=false)

initialize_mg_struct_CUDA(mg_struct_CUDA,128,128,7)
result_mg = mg_solver_CUDA(mg_struct_CUDA,mg_struct_CUDA.b_mg[1],nx=128,ny=128,n_level=6,iter_algo_num=3,max_mg_iterations=10,ω_richardson=2/1500)


#######################
initialize_mg_struct_CUDA(mg_struct_CUDA,512,512,8)
mg_solver_CUDA(mg_struct_CUDA,mg_struct_CUDA.b_mg[1],nx=512,ny=512,n_level=8,iter_algo_num=3,max_mg_iterations=10,ω_richardson=2/1000)


######################
initialize_mg_struct_CUDA(mg_struct_CUDA,1024,1024,10)


#######################
initialize_mg_struct_CUDA(mg_struct_CUDA,512,512,9)

mg_struct_CUDA.x_CUDA[1] .= 0;
x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=512,ny=512,n_level=8,ω_richardson=2/1000,max_cg_iter=20, precond=true, dynamic_richardson_ω=true)


x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=1024,ny=1024,n_level=9,ω_richardson=2/1000,max_cg_iter=10, dynamic_richardson_ω=false)

x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=1024,ny=1024,n_level=9,ω_richardson=2/1000,max_cg_iter=10, dynamic_richardson_ω=true)


# resolving it again
x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=1024,ny=1024,n_level=9,ω_richardson=2/1000,max_cg_iter=10)

@btime for _ in 1:1
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA(mg_struct_CUDA, nx=1024,ny=1024,n_level=9, ω_richardson=2/1000,max_cg_iter=10)
end


# mg_solver_CUDA(mg_struct_CUDA, r_CUDA, n_level = n_level, v1=v1,v2=v2,v3=v3, max_mg_iterations=1, nx = nx, ny = ny, iter_algo_num=3)

# solving it on nx = ny = 2048

initialize_mg_struct_CUDA(mg_struct_CUDA, 2048, 2048, 10)
x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=2048,ny=2048,n_level=10,ω_richardson=2/1000,max_cg_iter=10, dynamic_richardson_ω=false)

x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=2048,ny=2048,n_level=10,ω_richardson=2/1000,max_cg_iter=10, dynamic_richardson_ω=true)

@btime for _ in 1:1
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA(mg_struct_CUDA, nx=2048,ny=2048,n_level=10,ω_richardson=2/1000,max_cg_iter=10, dynamic_richardson_ω=true) 
end



# Solving it on nx = ny = 4096
x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=4096,ny=4096,n_level=11,ω_richardson=2/1000,max_cg_iter=10)

@btime for _ in 1:1
    mg_struct_CUDA.x_CUDA[1] .= 0
    mgcg_CUDA($mg_struct_CUDA, nx=4096,ny=4096,n_level=11,ω_richardson=2/1000,max_cg_iter=10)
end


# Solving it on nx = ny = 8192
x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=8192,ny=8192,n_level=13,ω_richardson=2/1000,max_cg_iter=10)

@btime for _ in 1:1
    # x_CUDA, counter = mgcg_CUDA(mg_struct_CUDA, nx=8192,ny=8192,n_level=13,ω_richardson=2/1000,max_cg_iter=10)
    mgcg_CUDA(mg_struct_CUDA, nx=8192,ny=8192,n_level=13,ω_richardson=2/1000,max_cg_iter=10)
end




mg_struct_CUDA.H_mg[5] \ Array(mg_struct_CUDA.b_mg[5])[:]

mfA_H_inverse(mg_struct_CUDA.b_mg[5],mg_struct_CUDA,5)
mg_struct_CUDA.odata_mg[5]


mg_struct_CUDA.H_mg[5] *  Array(mg_struct_CUDA.b_mg[5][:])
mfA_H(mg_struct_CUDA.b_mg[5],mg_struct_CUDA,5)
mg_struct_CUDA.odata_mg[5]




function discretization_error(mg_struct_CUDA)

    for i in 1:length(mg_struct_CUDA.u_exact)
        u_direct = mg_struct_CUDA.A_cpu_mg[i] \ Array(mg_struct_CUDA.b_mg[i][:])
        u_exact = Array(mg_struct_CUDA.u_exact[i])

        L2_error = sqrt((u_direct .- u_exact[:])' * mg_struct_CUDA.H_mg[i] * (u_direct - u_exact[:]) )

        @show mg_struct_CUDA.lnx_mg[i], L2_error, log2(L2_error)
    end

end