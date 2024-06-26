
using CUDA

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
    H_inv_mg
    f_mg
    u_mg
    r_mg
    prol_fine_mg
    rest_mg
    prol_mg
    lnx_mg
    lny_mg
    u_exact
    discretization_error

    # structs for mgcg_CUDA
    x_CUDA
    r_CUDA
    r_new_CUDA
    z_CUDA
    z_new_CUDA
    p_CUDA
end

mg_struct_CUDA = MG_CUDA([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])

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
    mg_struct_CUDA.H_inv_mg = []
    mg_struct_CUDA.f_mg = []
    mg_struct_CUDA.u_mg = []
    mg_struct_CUDA.r_mg = []
    mg_struct_CUDA.prol_fine_mg = []
    mg_struct_CUDA.rest_mg = []
    mg_struct_CUDA.prol_mg = []
    mg_struct_CUDA.lnx_mg = []
    mg_struct_CUDA.lny_mg = []
    mg_struct_CUDA.u_exact = []
    mg_struct_CUDA.discretization_error = []
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



function initialize_mg_struct_CUDA_Galerkin(mg_struct_CUDA,nx,ny,n_level)
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
    H_inv_mg = mg_struct_CUDA.H_inv_mg
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
                push!(H_mg, CUDA.CUSPARSE.CuSparseMatrixCSR(H_DDNN))
                push!(H_inv_mg, CUDA.CUSPARSE.CuSparseMatrixCSR(H_DDNN_inv))
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
                push!(H_mg, CUDA.CUSPARSE.CuSparseMatrixCSR(H_DDNN))
                push!(H_inv_mg, CUDA.CUSPARSE.CuSparseMatrixCSR(H_DDNN_inv))
                push!(f_mg, CuArray(zeros(nx+1, ny+1)))
                x = metrics_DDNN.coord[1]
                y = metrics_DDNN.coord[2]
                u_exact = ue(x,y)
                push!(mg_struct_CUDA.u_exact, CuArray(u_exact))
            end
            push!(u_mg, CuArray(zeros(nx+1, ny+1)))
            push!(odata_mg, CuArray(zeros(nx+1, ny+1)))
            push!(odata_interior_mg, CuArray(zeros(nx+1, ny+1)))

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

    push!(mg_struct_CUDA.x_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.r_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.r_new_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.z_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.z_new_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    push!(mg_struct_CUDA.p_CUDA, CuArray(zeros(lnx_mg[1]+1,lnx_mg[1]+1)))
    
    get_lams_Galerkin(mg_struct_CUDA)
end

function get_lams_Galerkin(mg_struct_CUDA)
    # TO DO, get 
    empty!(mg_struct_CUDA.λ_mins)
    empty!(mg_struct_CUDA.λ_maxs)
    reverse_Amg = reverse(mg_struct_CUDA.A_cpu_mg)
    if size(reverse_Amg[1])[1] > 289
        println("The minimal A matrix is too large for λ_min calculation")
        return 0
    end
    for k in eachindex(reverse_Amg)
        if size(reverse_Amg[k])[1] <= 1089 # nx <= 32
            lam_min, v_min = eigs(reverse_Amg[k], nev=1, which=:SR)
            lam_min = real(lam_min[1])

            lam_max, v_max = eigs(reverse_Amg[k], nev=1, which=:LR)
            lam_max = real(lam_max[1]) # try different formulations
        else
            lam_min = mg_struct_CUDA.λ_mins[1] / 4
            if size(reverse_Amg[k])[1] <= 16641
                lam_max, v_max = eigs(reverse_Amg[k], nev=1, which=:LR)
                lam_max = real(lam_max[1]) # try different formulations
            else
                lam_max = mg_struct_CUDA.λ_maxs[1] + (mg_struct_CUDA.λ_maxs[1] - mg_struct_CUDA.λ_maxs[2]) * 0.6
            end
        end
        pushfirst!(mg_struct_CUDA.λ_mins, lam_min)
        pushfirst!(mg_struct_CUDA.λ_maxs, lam_max)
    end 
    # return lam_mins, lam_maxs, αs, δs
end



function mg_solver_CUDA_SpMV_Galerkin(mg_struct_CUDA, f_in; nx=64, ny=64, n_level=3, v1=5, v2=5, v3=5, tolerence=1e-10, iter_algo_num=3,ω=1, ω_richardson=2/1000, max_mg_iterations=1, use_sbp=true, use_direct_sol=false,dynamic_richardson_ω=false)
    if isempty(mg_struct_CUDA.A_mg)
        initialize_mg_struct_CUDA(mg_struct_CUDA,nx,ny,n_level)
    end
    clear_urf_CUDA(mg_struct_CUDA)


    mg_struct_CUDA.f_mg[1][:] .= copy(f_in)[:]


    iter_algos = ["SOR","jacobi","richardson"]
    iter_algo = iter_algos[iter_algo_num]

    mg_struct_CUDA.r_mg[1][:] .= mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.u_mg[1][:]


    mg_iter_count = 0

    if nx < (2^n_level)
        println("Number of levels exceeds the possible number.")
    end

    for iteration_count = 1:max_mg_iterations
        mg_iter_count += 1

        if iter_algo == "richardson"
            if dynamic_richardson_ω == true
                ω_richardson = 2 / (mg_struct_CUDA.λ_mins[1] + mg_struct_CUDA.λ_maxs[1])
            end
            for i in 1:v1

                mg_struct_CUDA.u_mg[1][:] .+= ω_richardson .* (mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.u_mg[1][:])
            end
        elseif iter_algo == "SOR" #very slow for GPU arrays
            mg_struct_CUDA.u_mg[1][:] .= sor!(mg_struct_CUDA.u_mg[1][:],mg_struct_CUDA.A_mg[1],mg_struct_CUDA.f_mg[1][:],ω;maxiter=1)
        end

        mg_struct_CUDA.r_mg[1][:] .= mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.u_mg[1][:]


        for k = 2:n_level
            if k == 2
                mg_struct_CUDA.r_mg[k-1] .= mg_struct_CUDA.r_mg[1]
            else
                mg_struct_CUDA.r_mg[k-1][:] .= mg_struct_CUDA.f_mg[k-1][:] .- mg_struct_CUDA.A_mg[k-1] * mg_struct_CUDA.u_mg[k-1][:]
            end
            # mg_struct_CUDA.f_mg[k][:] .= mg_struct_CUDA.rest_mg[k-1] * CuArray((mg_struct_CUDA.H_mg[k-1] \ Array(mg_struct_CUDA.r_mg[k-1][:])))
            mg_struct_CUDA.f_mg[k][:] .= mg_struct_CUDA.rest_mg[k-1] * mg_struct_CUDA.H_inv_mg[k-1] * mg_struct_CUDA.r_mg[k-1][:]
            
            mg_struct_CUDA.f_mg[k][:] = mg_struct_CUDA.H_mg[k] * mg_struct_CUDA.f_mg[k][:]


            if k < n_level
               
                if iter_algo == "richardson"
                    if dynamic_richardson_ω == true
                        ω_richardson = 2 / (mg_struct_CUDA.λ_mins[k] + mg_struct_CUDA.λ_maxs[k])
                    end
                    for i in 1:v1
                        mg_struct_CUDA.u_mg[k][:] .+= ω_richardson .* (mg_struct_CUDA.f_mg[k][:] .- mg_struct_CUDA.A_mg[k] * mg_struct_CUDA.u_mg[k][:])
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
                            mg_struct_CUDA.u_mg[k][:] .+= ω_richardson .* (mg_struct_CUDA.f_mg[k][:] .- mg_struct_CUDA.A_mg[k] * mg_struct_CUDA.u_mg[k][:])
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

            # matrix_free_prolongation_2d_GPU(mg_struct_CUDA.u_mg[k], mg_struct_CUDA.prol_fine_mg[k-1])
            mg_struct_CUDA.prol_fine_mg[k-1][:] .= mg_struct_CUDA.prol_mg[k-1] * mg_struct_CUDA.u_mg[k][:]
            mg_struct_CUDA.u_mg[k-1] .+= mg_struct_CUDA.prol_fine_mg[k-1]

            
            if iter_algo == "richardson"
                if dynamic_richardson_ω == true
                    ω_richardson = 2 / (mg_struct_CUDA.λ_mins[k-1] + mg_struct_CUDA.λ_maxs[k-1])
                end
                for i in 1:v3
                    mg_struct_CUDA.u_mg[k-1][:] .+= ω_richardson .* (mg_struct_CUDA.f_mg[k-1][:] .- mg_struct_CUDA.A_mg[k-1] * mg_struct_CUDA.u_mg[k-1][:])
                end
            elseif iter_algo == "SOR"
                mg_struct_CUDA.u_mg[k-1][:] .= sor!(mg_struct_CUDA.u_mg[k-1][:],mg_struct_CUDA.A_mg[k-1],mg_struct_CUDA.f_mg[k-1][:],ω;maxiter=1)
            end
            
        end
  
        mg_struct_CUDA.r_mg[1][:] .= mg_struct_CUDA.f_mg[1][:] .- mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.u_mg[1][:]
    end
    return mg_struct_CUDA.u_mg[1]
end




function mgcg_CUDA_SpMV_Galerkin(mg_struct_CUDA;nx=64,ny=64,n_level=3,v1=5,v2=5,v3=5, ω=1.0, ω_richardson=2/1000, max_cg_iter=(nx+1)^2, max_mg_iterations=1,iter_algo_num=3, rel_tol=1e-6, precond=true,dynamic_richardson_ω=false, show_error=false)
    if length(mg_struct_CUDA.lnx_mg) == 0 || nx != mg_struct_CUDA.lnx_mg[1]
        clear_mg_struct_CUDA(mg_struct_CUDA)
        initialize_mg_struct_CUDA(mg_struct_CUDA, nx, ny, n_level)
    end


    disc_error = discretization_errors[nx] # maybe there's a better way to do this


    mg_struct_CUDA.r_CUDA[1][:] .= mg_struct_CUDA.b_mg[1][:] .- mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.x_CUDA[1][:]

    init_rms = norm(mg_struct_CUDA.r_CUDA[1])
    # z_CUDA = CuArray(zeros(nx+1,ny+1))

    L2_error = sqrt(dot(mg_struct_CUDA.x_CUDA[1][:] .- mg_struct_CUDA.u_exact[1][:], mg_struct_CUDA.H_mg[1] * (mg_struct_CUDA.x_CUDA[1][:] .- mg_struct_CUDA.u_exact[1][:]) ))
    if show_error == true
        @show 0, init_rms, L2_error
    end

    if precond == true
        @inbounds mg_solver_CUDA_SpMV(mg_struct_CUDA, mg_struct_CUDA.r_CUDA[1], n_level = n_level, v1=v1,v2=v2,v3=v3, max_mg_iterations=max_mg_iterations, nx = nx, ny = ny, iter_algo_num=iter_algo_num, dynamic_richardson_ω=dynamic_richardson_ω)
        mg_struct_CUDA.z_CUDA[1][:] .= mg_struct_CUDA.u_mg[1][:]
    else
        mg_struct_CUDA.z_CUDA[1][:] .= mg_struct_CUDA.r_CUDA[1][:]
    end

    @inbounds mg_struct_CUDA.p_CUDA[1][:] .= mg_struct_CUDA.z_CUDA[1][:]

    counter = 0
    for k in 1:max_cg_iter
        counter += 1

        α = dot(mg_struct_CUDA.r_CUDA[1][:],mg_struct_CUDA.z_CUDA[1][:]) / (dot(mg_struct_CUDA.p_CUDA[1][:],mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.p_CUDA[1][:]))

        @inbounds mg_struct_CUDA.x_CUDA[1][:] .+= α .* mg_struct_CUDA.p_CUDA[1][:]

        @inbounds mg_struct_CUDA.r_new_CUDA[1][:] .= mg_struct_CUDA.r_CUDA[1][:] .- α .* mg_struct_CUDA.A_mg[1] * mg_struct_CUDA.p_CUDA[1][:]


        norm_v_initial_norm = norm(mg_struct_CUDA.r_new_CUDA[1]) / init_rms
        
        L2_error = sqrt(dot(mg_struct_CUDA.x_CUDA[1][:] .- mg_struct_CUDA.u_exact[1][:], mg_struct_CUDA.H_mg[1] * (mg_struct_CUDA.x_CUDA[1][:] .- mg_struct_CUDA.u_exact[1][:]) ))


        if show_error == true
            @show k, norm_v_initial_norm, L2_error
        end

        # println("")

        if norm(mg_struct_CUDA.r_new_CUDA[1]) < rel_tol * init_rms # && L2_error <= 2 * disc_error
            break
        end

        if precond == true
            mg_solver_CUDA_SpMV(mg_struct_CUDA, mg_struct_CUDA.r_new_CUDA[1], n_level = n_level, v1=v1,v2=v2,v3=v3, max_mg_iterations=max_mg_iterations, nx = nx, ny = ny, iter_algo_num=iter_algo_num, dynamic_richardson_ω=dynamic_richardson_ω)
            @inbounds mg_struct_CUDA.z_new_CUDA[1] .= mg_struct_CUDA.u_mg[1]
        else
            mg_struct_CUDA.z_new_CUDA[1][:] .= mg_struct_CUDA.r_new_CUDA[1][:]
        end

        β = dot(mg_struct_CUDA.r_new_CUDA[1][:], mg_struct_CUDA.z_new_CUDA[1][:]) / (dot(mg_struct_CUDA.r_CUDA[1][:],mg_struct_CUDA.z_CUDA[1][:]))

        @inbounds mg_struct_CUDA.p_CUDA[1][:] .= mg_struct_CUDA.z_new_CUDA[1][:] .+ β .* mg_struct_CUDA.p_CUDA[1][:]
        @inbounds mg_struct_CUDA.z_CUDA[1][:] .= mg_struct_CUDA.z_new_CUDA[1][:]
        @inbounds mg_struct_CUDA.r_CUDA[1][:] .= mg_struct_CUDA.r_new_CUDA[1][:]
    end
    return mg_struct_CUDA.x_CUDA[1], counter     
end

