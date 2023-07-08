include("utils_MF.jl")

# Physical domain
Lx = 2
Ly = 2
Lw = 0.05
D = 0.05
B_on = 1 # basin on or off?

B_p = (μ_out = 36.0,
μ_in = 20.0,
c = (Lw/2)/D,
r̄ = (Lw/2)^2,
r_w = 0.01*(1 + (Lw/2)/D),
on = B_on)

# Lx = 2
# Ly = 2
# Lw = 24
# D = 4
# B_on = 0 # basin on or off?
# el_x = 10e12 # set it to be infinity to have even spread
# el_y = 10e12

# B_p = (μ_out = 36.0,
# μ_in = 20.0,
# c = (Lw/2)/D,
# r̄ = (Lw/2)^2,
# r_w = 1 + (Lw/2)/D,
# on = B_on)

function create_ops(Nr, Ns)
    SBPp = 2
    Nr = Nr
    Ns = Ns
    hr = 2 / Nr
    hs = 2 / Ns
    Nr1 = Nr + 1
    Ns1 = Ns + 1


    # Define corners of domain:
    (x1, x2, x3, x4) = [-0.3 0.5 0 1.0]
    (y1, y2, y3, y4) = [0 -.25 1.0 1.5]


    # Initialize the block transformations as transfinite between the corners
    ex = [(α) -> x1 * (1 .- α) / 2 + x3 * (1 .+ α) / 2,
        (α) -> x2 * (1 .- α) / 2 + x4 * (1 .+ α) / 2,
        (α) -> x1 * (1 .- α) / 2 + x2 * (1 .+ α) / 2,
        (α) -> x3 * (1 .- α) / 2 + x4 * (1 .+ α) / 2]
    exα = [(α) -> -x1 / 2 + x3 / 2,
        (α) -> -x2 / 2 + x4 / 2,
        (α) -> -x1 / 2 + x2 / 2,
        (α) -> -x3 / 2 + x4 / 2]
    ey = [(α) -> y1 * (1 .- α) / 2 + y3 * (1 .+ α) / 2,
        (α) -> y2 * (1 .- α) / 2 + y4 * (1 .+ α) / 2,
        (α) -> y1 * (1 .- α) / 2 + y2 * (1 .+ α) / 2,
        (α) -> y3 * (1 .- α) / 2 + y4 * (1 .+ α) / 2]
    eyα = [(α) -> -y1 / 2 + y3 / 2,
        (α) -> -y2 / 2 + y4 / 2,
        (α) -> -y1 / 2 + y2 / 2,
        (α) -> -y3 / 2 + y4 / 2]


    #Put in curvy edge transform:
    β1 = 0.1; β2 = β1; β3 = β1; β4 = β1; 
    ex[1] = (α) -> x1 * (1 .- α) / 2 .+ x3 * (1 .+ α) / 2 .+ β1 .* sin.(π .* (α .+ 1) ./ 2)
    ey[1] = (α) -> y1 .* (1 .- α) / 2 .+ y3 .* (1 .+ α) / 2 
    exα[1] = (α) -> -x1 / 2 .+ x3 / 2 .+ β1 .* (π/2) .* cos.(π .* (α .+ 1) ./ 2)
    eyα[1] = (α) -> -y1 / 2 .+ y3 / 2 

    ex[2] = (α) -> x2 * (1 .- α) / 2 .+ x4 * (1 .+ α) / 2 .- β2 .* sin.(π .* (α .+ 1) ./ 2)
    ey[2] = (α) -> y2 .* (1 .- α) / 2 .+ y4 .* (1 .+ α) / 2 
    exα[2] = (α) -> -x2 / 2 .+ x4 / 2 .- β2 .* (π/2) .* cos.(π .* (α .+ 1) ./ 2)
    eyα[2] = (α) -> -y2 / 2 .+ y4 / 2 


    ex[3] = (α) -> x1 * (1 .- α) / 2 .+ x2 * (1 .+ α) / 2
    ey[3] = (α) -> y1 .* (1 .- α) / 2 .+ y2 .* (1 .+ α) / 2 .+ β3 .* sin.(π .* (α .+ 1) ./ 2)
    exα[3] = (α) -> -x1 / 2 .+ x2 / 2
    eyα[3] = (α) -> -y1 / 2 .+ y2 / 2 .+ β3 .* (π/2) .* cos.(π .* (α .+ 1) ./ 2)

    ex[4] = (α) -> x3 * (1 .- α) / 2 .+ x4 * (1 .+ α) / 2
    ey[4] = (α) -> y3 .* (1 .- α) / 2 .+ y4 .* (1 .+ α) / 2 .+ β4 .* sin.(π .* (α .+ 1))
    exα[4] = (α) -> -x3 / 2 .+ x4 / 2
    eyα[4] = (α) -> -y3 / 2 .+ y4 / 2 .+ β4 .* π .* cos.(π .* (α .+ 1))

    # Create the volume transform as the transfinite blending of the edge
    # transformations
    xt(r,s) = transfinite_blend(ex[1], ex[2], ex[3], ex[4],
                                    exα[1], exα[2], exα[3], exα[4],
                                    r, s)
    yt(r,s) = transfinite_blend(ey[1], ey[2], ey[3], ey[4],
                                    eyα[1], eyα[2], eyα[3], eyα[4],
                                    r, s)


    # EToV defines the element by its vertices
    # EToF defines element by its four faces, in global face number
    # FToB defines whether face is Dirichlet (1), Neumann (2), interior jump (7)
    #      or just an interior interface (0)
    # EToDomain is 1 if element is inside circle; 2 otherwise

    # Physical domain
    Lx = 2
    Ly = 2
    Lw = 0.05
    D = 0.05
    B_on = 1 # basin on or off?

    B_p = (μ_out = 36.0,
    μ_in = 20.0,
    c = (Lw/2)/D,
    r̄ = (Lw/2)^2,
    r_w = 0.01*(1 + (Lw/2)/D),
    on = B_on)

    metrics = create_metrics(Nr, Ns, B_p, μ, xt, yt)

    x = metrics.coord[1]
    y = metrics.coord[2]

    # boundary condtions on faces 1, 2, 3, 4
    ops = MFoperators(SBPp, Nr, Ns, metrics)

    # obtain g that stores boundary data
    g = zeros((Nr[1] + 1) * (Ns[1] + 1))

    vf = zeros((Nr[1] + 1)) # TODO: Modify this because as it assumes equal # of grid points in both directions

    # Assumes zeros boundary data for Neumann conditions on faces 3 and 4
    f1_data = ue(metrics.facecoord[1][1], metrics.facecoord[2][1])
    f2_data = ue(metrics.facecoord[1][2], metrics.facecoord[2][2])


    # Create Neumann data on faces 3 and 4:
    f3_data = metrics.sJ[3] .* metrics.nx[3] .* μ(metrics.facecoord[1][3], metrics.facecoord[2][3], B_p) .* ue_x(metrics.facecoord[1][3], metrics.facecoord[2][3]) .+ 
            metrics.sJ[3] .* metrics.ny[3] .* μ(metrics.facecoord[1][3], metrics.facecoord[2][3], B_p) .* ue_y(metrics.facecoord[1][3], metrics.facecoord[2][3])
    f4_data = metrics.sJ[4] .* metrics.nx[4] .* μ(metrics.facecoord[1][4], metrics.facecoord[2][4], B_p) .* ue_x(metrics.facecoord[1][4], metrics.facecoord[2][4]) .+ 
            metrics.sJ[4] .* metrics.ny[4] .* μ(metrics.facecoord[1][4], metrics.facecoord[2][4], B_p) .* ue_y(metrics.facecoord[1][4], metrics.facecoord[2][4])

    mod_data!(f1_data, f2_data, f3_data, f4_data, g, ops.K, vf)
    g .+= ops.JH * Force(x, y, B_p)[:]

    return ops.M̃, g, ops.H̃, ops.H̃I, metrics

end

function ue(x, y)
    return sin.(2 .* pi .* x ./ Lx) .* cos.(2 .* pi .* y ./ Ly)
end

function ue_x(x, y)
    return (2*pi/Lx) .* cos.(2 .* pi .* x ./ Lx) .* cos.(2 .* pi .* y ./ Ly)
end

function ue_xx(x, y)
    return -(2*pi/Lx)^2 .* sin.(2 .* pi .* x ./ Lx) .* cos.(2 .* pi .* y ./ Ly)
end


function ue_y(x, y)
    return -(2*pi/Ly) .* sin.(2 .* pi .* x ./ Lx) .* sin.(2 .* pi .* y ./ Ly)
end

function ue_yy(x, y)
    return -(2*pi/Ly)^2 .* sin.(2 .* pi .* x ./ Lx) .* cos.(2 .* pi .* y ./ Ly)
end

function Force(x, y, B_p)
        
    F = .-(μ_x(x, y, B_p) .* ue_x(x, y) .+ μ(x, y, B_p) .* ue_xx(x, y) .+
         μ_y(x, y, B_p) .* ue_y(x, y) .+ μ(x, y, B_p) .* ue_yy(x, y))

    return F
end

function compute_l2norm(nx, ny, r)
    rms = 0.0
    # println(residual)
    for j = 2:ny for i = 2:nx
        rms = rms + r[i,j]^2
    end end
    # println(rms)
    rms = sqrt(rms/((nx-1)*(ny-1)))
    return rms
end

discretization_errors = Dict(
    2^13=>5.29E-08,
    2^12=>2.12E-07,
    2^11=>8.46E-07,
    2^10=>3.39E-06,
    2^9=>1.35E-05,
    2^8=>5.41E-05,
    2^7=>0.0002155362349,
    2^6=>0.0008521251106,
    2^5=>0.00329236586,
    2^4=>0.011944002,
    2^3=>0.03734125119,
    2^2=>0.105521651)