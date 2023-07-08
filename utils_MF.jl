include("DiagonalSBP.jl")

using SparseArrays
using LinearAlgebra

⊗(A,B) = kron(A, B)


function μ(x, y, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    on = B_p.on

    if on == false
        if ndims(x) == 2
            return repeat([μ_out], outer=size(x))
        else
            return repeat([μ_out], outer=length(x))
        end
    else
        return (μ_out - μ_in)/2 *
            (tanh.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .+ 1) .+ μ_in
    end
end

function μ_x(x, y, B_p)
    
    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    on = B_p.on

    if on == false
        if ndims(x) == 2
            return zeros(size(x))
        else
            return zeros(length(x))
        end
    else
        return ((μ_out - μ_in) .* x .*
                sech.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w
    end
end

function μ_y(x, y, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    on = B_p.on

    if on == false
        if ndims(x) == 2
            return zeros(size(x))
        else
            return zeros(length(x))
        end

    else    
        return ((μ_out - μ_in) .* (c^2 * y) .*
            sech.((x .^ 2 + c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w
    end
end


function create_metrics(Nr, Ns, B_p, μ,
    xf=(r,s)->(r, ones(size(r)), zeros(size(r))),
    yf=(r,s)->(s, zeros(size(s)), ones(size(s))))

    Nrp = Nr + 1
    Nsp = Ns + 1

    r = range(-1, stop=1, length=Nrp)
    s = range(-1, stop=1, length=Nsp)

    # Create the mesh
    r = ones(1, Nsp) ⊗ r
    s = s' ⊗ ones(Nrp)

    (x, xr, xs) = xf(r, s)
    (y, yr, ys) = yf(r, s)

    J = xr .* ys - xs .* yr

    @assert minimum(J) > 0

    JI = 1 ./ J

    rx =  ys ./ J
    sx = -yr ./ J
    ry = -xs ./ J
    sy =  xr ./ J

    μx = μy = μ(x, y, B_p)


    # variable coefficient matrix components
    crr = J .* (rx .* μx .* rx + ry .* μy .* ry)
    crs = J .* (sx .* μx .* rx + sy .* μy .* ry)
    css = J .* (sx .* μx .* sx + sy .* μy .* sy)

    # surface matrices
    (xf1, yf1) = (view(x, 1, :), view(y, 1, :))
    nx1 = -ys[1, :]
    ny1 =  xs[1, :]
    sJ1 = hypot.(nx1, ny1)
    nx1 = nx1 ./ sJ1
    ny1 = ny1 ./ sJ1


    (xf2, yf2) = (view(x, Nrp, :), view(y, Nrp, :))
    nx2 =  ys[end, :]
    ny2 = -xs[end, :]
    sJ2 = hypot.(nx2, ny2)
    nx2 = nx2 ./ sJ2
    ny2 = ny2 ./ sJ2
  

    (xf3, yf3) = (view(x, :, 1), view(y, :, 1))
    nx3 =  yr[:, 1]
    ny3 = -xr[:, 1]
    sJ3 = hypot.(nx3, ny3)
    nx3 = nx3 ./ sJ3
    ny3 = ny3 ./ sJ3

    (xf4, yf4) = (view(x, :, Nsp), view(y, :, Nsp))
    nx4 = -yr[:, end]
    ny4 =  xr[:, end]
    sJ4 = hypot.(nx4, ny4)
    nx4 = nx4 ./ sJ4
    ny4 = ny4 ./ sJ4

    (coord = (x,y),
    facecoord = ((xf1, xf2, xf3, xf4), (yf1, yf2, yf3, yf4)),
    crr = crr, css = css, crs = crs, 
    J=J,
    JI = JI,
    sJ = (sJ1, sJ2, sJ3, sJ4),
    nx = (nx1, nx2, nx3, nx4),
    ny = (ny1, ny2, ny3, ny4),
    rx = rx, ry = ry, sx = sx, sy = sy)
end


function MFoperators(p, Nr, Ns, metrics, 
    crr = metrics.crr,
    css = metrics.css,
    crs = metrics.crs)


csr = crs
J = metrics.J

hr = 2/Nr
hs = 2/Ns

hmin = min(hr, hs)

r = -1:hr:1
s = -1:hs:1

Nrp = Nr + 1
Nsp = Ns + 1
Np = Nrp * Nsp

# Derivative operators for the rest of the computation
(Dr, HrI, Hr, r) = D1(p, Nr; xc = (-1,1))
Qr = Hr * Dr
QrT = sparse(transpose(Qr))

(Ds, HsI, Hs, s) = D1(p, Ns; xc = (-1,1))
Qs = Hs * Ds
QsT = sparse(transpose(Qs))

# Identity matrices for the comuptation
Ir = sparse(I, Nrp, Nrp)
Is = sparse(I, Nsp, Nsp)

(_, S0e, SNe, _, _, Ae, _) = variable_D2(p, Nr, rand(Nrp))
IArr = Array{Int64,1}(undef,Nsp * length(Ae.nzval))
JArr = Array{Int64,1}(undef,Nsp * length(Ae.nzval))
VArr = Array{Float64,1}(undef,Nsp * length(Ae.nzval))
stArr = 0
A_t = @elapsed begin
for j = 1:Nsp
rng = (j-1) * Nrp .+ (1:Nrp)
(_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Nr, crr[rng])
(Ie, Je, Ve) = findnz(Ae)
IArr[stArr .+ (1:length(Ve))] = Ie .+ (j-1) * Nrp
JArr[stArr .+ (1:length(Ve))] = Je .+ (j-1) * Nrp
VArr[stArr .+ (1:length(Ve))] = Hs[j,j] * Ve
stArr += length(Ve)
end

Ãrr = sparse(IArr[1:stArr], JArr[1:stArr], VArr[1:stArr], Np, Np)

(_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Ns, rand(Nsp))
IAss = Array{Int64,1}(undef,Nrp * length(Ae.nzval))
JAss = Array{Int64,1}(undef,Nrp * length(Ae.nzval))
VAss = Array{Float64,1}(undef,Nrp * length(Ae.nzval))
stAss = 0

for i = 1:Nrp
rng = i .+ Nrp * (0:Ns)
(_, S0e, SNe, _, _, Ae, _) =  variable_D2(p, Ns, css[rng])

(Ie, Je, Ve) = findnz(Ae)
IAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Ie .- 1)
JAss[stAss .+ (1:length(Ve))] = i .+ Nrp * (Je .- 1)
VAss[stAss .+ (1:length(Ve))] = Hr[i,i] * Ve
stAss += length(Ve)

end

Ãss = sparse(IAss[1:stAss], JAss[1:stAss], VAss[1:stAss], Np, Np)

Ãsr = (QsT ⊗ Ir) * sparse(1:length(crs), 1:length(crs), view(crs, :)) * (Is ⊗ Qr)
Ãrs = (Is ⊗ QrT) * sparse(1:length(csr), 1:length(csr), view(csr, :)) * (Qs ⊗ Ir)

Ã = Ãrr + Ãss + Ãrs + Ãsr
end


# volume quadrature
H̃ = kron(Hr, Hs)
H̃inv = spdiagm(0 => 1 ./ diag(H̃))


JI = spdiagm(0 => reshape(metrics.JI, Nrp*Nsp))

er0 = sparse([1  ], [1], [1], Nrp, 1)
erN = sparse([Nrp], [1], [1], Nrp, 1)
es0 = sparse([1  ], [1], [1], Nsp, 1)
esN = sparse([Nsp], [1], [1], Nsp, 1)

H = (Hs, Hs, Hr, Hr)
HI = (HsI, HsI, HrI, HrI)
# Volume to Face Operators (transpose of these is face to volume)
L = (convert(SparseMatrixCSC{Float64, Int64}, kron(Ir, es0)'),
convert(SparseMatrixCSC{Float64, Int64}, kron(Ir, esN)'),
convert(SparseMatrixCSC{Float64, Int64}, kron(er0, Is)'),
convert(SparseMatrixCSC{Float64, Int64}, kron(erN, Is)'))

# coefficent matrices
Crr1 = spdiagm(0 => crr[1, :])
Crs1 = spdiagm(0 => crs[1, :])
Csr1 = spdiagm(0 => crs[1, :])
Css1 = spdiagm(0 => css[1, :])

Crr2 = spdiagm(0 => crr[Nrp, :])
Crs2 = spdiagm(0 => crs[Nrp, :])
Csr2 = spdiagm(0 => crs[Nrp, :])
Css2 = spdiagm(0 => css[Nrp, :])

Css3 = spdiagm(0 => css[:, 1])
Crs3 = spdiagm(0 => crs[:, 1])
Csr3 = spdiagm(0 => crs[:, 1])
Crr3 = spdiagm(0 => crr[:, 1])

Css4 = spdiagm(0 => css[:, Nsp])
Crs4 = spdiagm(0 => crs[:, Nsp])
Csr4 = spdiagm(0 => crs[:, Nsp])
Crr4 = spdiagm(0 => crr[:, Nsp])

(_, S0, SN, _, _) = D2(p, Nr, xc=(-1,1))[1:5]
S0 = sparse(Array(S0[1,:])')
SN = sparse(Array(SN[end, :])')


# BoundarY Derivatives
B1r =  Crr1 * kron(Is, S0)
B1s = Crs1 * L[1] * kron(Ds, Ir)
B2r = Crr2 * kron(Is, SN)
B2s = Crs2 * L[2] * kron(Ds, Ir)
B3s = Css3 * kron(S0, Ir)
B3r = Csr3 * L[3] * kron(Is, Dr)
B4s = Css4 * kron(SN, Ir)
B4r = Csr4 * L[4] * kron(Is, Dr)

# Penalty terms

if p == 2
l = 2
α = 1.0
θ_R = 1 / 2
elseif p == 4
l = 4
β = 0.2505765857
α = 0.5776
θ_R = 17 / 48
elseif p == 6
l = 7
β = 0.1878687080
α = 0.3697
θ_R = 13649 / 43200
else
error("unknown order")
end

ψmin_r = reshape(crr, Nrp, Nsp)
ψmin_s = reshape(css, Nrp, Nsp)
@assert minimum(ψmin_r) > 0
@assert minimum(ψmin_s) > 0

hr = 2 / Nr
hs = 2 / Ns

ψ1 = ψmin_r[  1, :]
ψ2 = ψmin_r[Nrp, :]
ψ3 = ψmin_s[:,   1]
ψ4 = ψmin_s[:, Nsp]

for k = 2:l
ψ1 = min.(ψ1, ψmin_r[k, :])
ψ2 = min.(ψ2, ψmin_r[Nrp+1-k, :])
ψ3 = min.(ψ3, ψmin_s[:, k])
ψ4 = min.(ψ4, ψmin_s[:, Nsp+1-k])
end

τR1 = (1/(α*hr))*Is
τR2 = (1/(α*hr))*Is
τR3 = (1/(α*hs))*Ir
τR4 = (1/(α*hs))*Ir

p1 = ((crr[  1, :]) ./ ψ1)
p2 = ((crr[Nrp, :]) ./ ψ2)
p3 = ((css[:,   1]) ./ ψ3)
p4 = ((css[:, Nsp]) ./ ψ4)

P1 = sparse(1:Nsp, 1:Nsp, p1)
P2 = sparse(1:Nsp, 1:Nsp, p2)
P3 = sparse(1:Nrp, 1:Nrp, p3)
P4 = sparse(1:Nrp, 1:Nrp, p4)


# penalty matrices
Γ = ((2/(θ_R*hr))*Is + τR1 * P1,
(2/(θ_R*hr))*Is + τR2 * P2,
(2/(θ_R*hs))*Ir + τR3 * P3,
(2/(θ_R*hs))*Ir + τR4 * P4)

JH = sparse(1:Np, 1:Np, view(J, :)) * (Hs ⊗ Hr)



Cf = ((Crr1, Crs1), (Crr2, Crs2), (Css3, Csr3), (Css4, Csr4))
B = ((B1r, B1s), (B2r, B2s), (B3s, B3r), (B4s, B4r))
nl = (-1, 1, -1, 1)
G = (-H[1] * (B[1][1] + B[1][2]), H[2] * (B[2][1] + B[2][2]),
-H[3] * (B[3][1] + B[3][2]),  H[4] * (B[4][1] + B[4][2]))

# boundary data operators for quasi-static displacement conditions
K1 = L[1]' * H[1] * Cf[1][1] * Γ[1] - G[1]'
K2 = L[2]' * H[2] * Cf[2][1] * Γ[2] - G[2]'
#K3 = L[3]' * H[3] * Cf[3][1] * Γ[3] - G[3]'
#K4 = L[4]' * H[4] * Cf[4][1] * Γ[4] - G[4]'

# boundary data operator for quasi-static traction-free conditions
#K1 = L[1]' * H2]
#K2 = L[2]' * H[2]
K3 = L[3]' * H[3]
K4 = L[4]' * H[4]


# modification of second derivative operator for displacement conditions
# needs no modification for Neumann conditions.
M̃ = copy(Ã)
for f in 1:2
M̃ -= L[f]' * G[f]
M̃ += L[f]' * H[f] * Cf[f][1] * Γ[f] * L[f]
M̃ -= G[f]' * L[f]
end

# (M̃ = cholesky(Symmetric(M̃)),
(M̃ = M̃,
K = (K1, K2, K3, K4),
G = G,
Crr = Crr1,
Γ = Γ,
HI = HI,
H̃ = H̃,
H̃I = H̃inv,
JI = JI,
JH = JH,
sJ = metrics.sJ,
nx = metrics.nx,
ny = metrics.ny,
L = L,
H = H,
Arr = Ãrr,
Ass = Ãss,
Ars = Ãrs,
Asr = Ãsr,
hr,
hs,
oo = -L[2]' * G[2] + L[2]' * H[2] * Cf[2][1] * Γ[2] * L[2] -G[2]' * L[2], 
#oo2 = -L[1]' * G[1] + L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1] -G[1]' * L[1],
oo2 = -L[1]' * G[1] + L[1]' * H[1] * Cf[1][1] * Γ[1] * L[1]-G[1]' * L[1],
#oo2 = -L[1]' * G[1],
oo3 =  (Qs ⊗ Ir),
oo4 = (Is ⊗ QrT),
oo5 = (QsT ⊗ Ir))

end


function mod_data!(f1_data, f2_data, f3_data, f4_data, ge, K, vf)

ge .= 0

for i in 1:4
if i == 1
# face 1 data (Dirichlet)
vf .=  f1_data
elseif i == 2
# face 2 data (Dirichlet)
vf .= f2_data
elseif i == 3
# face 3 data (Neumann)
vf .= f3_data
elseif i == 4
# face 4 data (Neumann)
vf .= f4_data
end
ge .+= K[i] * vf
end

end


# {{{ Constructor for inp files
function read_inp_2d(T, S, filename::String; bc_map=1:10000)
# {{{ Read in the file
f = try
open(filename)
catch
error("InpRead cannot open \"$filename\" ")
end
lines = readlines(f)
close(f)
# }}}

# {{{ Read in nodes
str = "NSET=ALLNODES"
linenum = SeekToSubstring(lines, str);
linenum > 0 || error("did not find: $str")
num_nodes = 0
for l = linenum+1:length(lines)
occursin(r"^\s*[0-9]*\s*,.*", lines[l]) ? num_nodes+=1 : break
end
Vx = fill(S(NaN), num_nodes)
Vy = fill(S(NaN), num_nodes)
Vz = fill(S(NaN), num_nodes)
for l = linenum .+ (1:num_nodes)
node_data = split(lines[l], r"\s|,", keepempty=false)
(node_num, node_x, node_y, node_z) = try
(parse(T, node_data[1]),
parse(S, node_data[2]),
parse(S, node_data[3]),
parse(S, node_data[4]))
catch
error("cannot parse line $l: \"$(lines[l])\" ")
end

Vx[node_num] = node_x
Vy[node_num] = node_y
Vz[node_num] = node_z
end
# }}}

# {{{ Read in Elements
str = "ELEMENT"
linenum = SeekToSubstring(lines, str);
num_elm = 0
while linenum > 0
for l = linenum .+ (1:length(lines))
occursin(r"^\s*[0-9]*\s*,.*", lines[l]) ? num_elm+=1 : break
end
linenum = SeekToSubstring(lines, str; first=linenum+1)
end
num_elm > 0 || error("did not find any element")

EToV = fill(T(0), 4, num_elm)
EToBlock = fill(T(0), num_elm)
linenum = SeekToSubstring(lines, str);
while linenum > 0
foo = split(lines[linenum], r"[^0-9]", keepempty=false)
B = parse(T, foo[end])
for l = linenum .+ (1:num_elm)
elm_data = split(lines[l], r"\s|,", keepempty=false)
# read into z-order
(elm_num, elm_v1, elm_v2, elm_v4, elm_v3) = try
(parse(T, elm_data[1]),
parse(T, elm_data[2]),
parse(T, elm_data[3]),
parse(T, elm_data[4]),
parse(T, elm_data[5]))
catch
break
end
EToV[:, elm_num] = [elm_v1, elm_v2, elm_v3, elm_v4]
EToBlock[elm_num] = B
end
linenum = SeekToSubstring(lines, str; first=linenum+1)
end
# }}}

# {{{ Determine connectivity
EToF = fill(T(0), 4, num_elm)

VsToF = Dict{Tuple{Int64, Int64}, Int64}()
numfaces = 0
for e = 1:num_elm
for lf = 1:4
if lf == 1
Vs = (EToV[1, e], EToV[3, e])
elseif lf == 2
Vs = (EToV[2, e], EToV[4, e])
elseif lf == 3
Vs = (EToV[1, e], EToV[2, e])
elseif lf == 4
Vs = (EToV[3, e], EToV[4, e])
end
if Vs[1] > Vs[2]
Vs = (Vs[2], Vs[1])
end
if haskey(VsToF, Vs)
EToF[lf, e] = VsToF[Vs]
else
numfaces = numfaces + 1
EToF[lf, e] = VsToF[Vs] = numfaces
end
end
end
#}}}

# {{{ Read in side set info
FToB = Array{T, 1}(undef, numfaces)
fill!(FToB, BC_LOCKED_INTERFACE)
linenum = SeekToSubstring(lines, "\\*ELSET")
inp_to_zorder = [3,  2, 4, 1]
while linenum > 0
foo = split(lines[linenum], r"[^0-9]", keepempty=false)
(bc, face) = try
(parse(T, foo[1]),
parse(T, foo[2]))
catch
error("cannot parse line $linenum: \"$(lines[linenum])\" ")
end
bc = bc_map[bc]
face = inp_to_zorder[face]
for l = linenum+1:length(lines)
if !occursin(r"^\s*[0-9]+", lines[l])
break
end
elms = split(lines[l], r"\s|,", keepempty=false)
for elm in elms
elm = try
parse(T, elm)
catch
error("cannot parse line $linenum: \"$(lines[l])\" ")
end
if bc == 3
bc = BC_LOCKED_INTERFACE
end
FToB[EToF[face, elm]] = bc
@assert (bc == BC_DIRICHLET || bc == BC_NEUMANN ||
   bc == BC_LOCKED_INTERFACE || bc >= BC_JUMP_INTERFACE)
end
end
linenum = SeekToSubstring(lines, "\\*ELSET"; first=linenum+1)
end
# }}}

([Vx Vy]', EToV, EToF, FToB, EToBlock)
end
read_inp_2d(filename;kw...) = read_inp_2d(Int64, Float64, filename;kw...)

function SeekToSubstring(lines, substring; first=1)
for l = first:length(lines)
if occursin(Regex(".*$(substring).*"), lines[l])
return l
end
end
return -1
end

# }}}


#{{{ Transfinite Blend
function transfinite_blend(α1, α2, α3, α4, α1s, α2s, α3r, α4r, r, s)
# +---4---+
# |       |
# 1       2
# |       |
# +---3---+
@assert [α1(-1) α2(-1) α1( 1) α2( 1)] ≈ [α3(-1) α3( 1) α4(-1) α4( 1)]


x = (1 .+ r) .* α2(s)/2 + (1 .- r) .* α1(s)/2 +
(1 .+ s) .* α4(r)/2 + (1 .- s) .* α3(r)/2 -
((1 .+ r) .* (1 .+ s) .* α2( 1) +
(1 .- r) .* (1 .+ s) .* α1( 1) +
(1 .+ r) .* (1 .- s) .* α2(-1) +
(1 .- r) .* (1 .- s) .* α1(-1)) / 4

xr =  α2(s)/2 - α1(s)/2 +
(1 .+ s) .* α4r(r)/2 + (1 .- s) .* α3r(r)/2 -
(+(1 .+ s) .* α2( 1) +
-(1 .+ s) .* α1( 1) +
+(1 .- s) .* α2(-1) +
-(1 .- s) .* α1(-1)) / 4


xs = (1 .+ r) .* α2s(s)/2 + (1 .- r) .* α1s(s)/2 +
α4(r)/2 - α3(r)/2 -
(+(1 .+ r) .* α2( 1) +
+(1 .- r) .* α1( 1) +
-(1 .+ r) .* α2(-1) +
-(1 .- r) .* α1(-1)) / 4

return (x, xr, xs)
end

function transfinite_blend(α1, α2, α3, α4, r, s, p)
(Nrp, Nsp) = size(r)
(Dr, _, _, _) = diagonal_sbp_D1(p, Nrp-1; xc = (-1,1))
(Ds, _, _, _) = diagonal_sbp_D1(p, Nsp-1; xc = (-1,1))

α2s(s) = α2(s) * Ds'
α1s(s) = α1(s) * Ds'
α4r(s) = Dr * α4(r)
α3r(s) = Dr * α3(r)

transfinite_blend(α1, α2, α3, α4, α1s, α2s, α3r, α4r, r, s)
end

function transfinite_blend(v1::T1, v2::T2, v3::T3, v4::T4, r, s
            ) where {T1 <: Number, T2 <: Number,
                     T3 <: Number, T4 <: Number}
e1(α) = v1 * (1 .- α) / 2 + v3 * (1 .+ α) / 2
e2(α) = v2 * (1 .- α) / 2 + v4 * (1 .+ α) / 2
e3(α) = v1 * (1 .- α) / 2 + v2 * (1 .+ α) / 2
e4(α) = v3 * (1 .- α) / 2 + v4 * (1 .+ α) / 2
e1α(α) = -v1 / 2 + v3 / 2
e2α(α) = -v2 / 2 + v4 / 2
e3α(α) = -v1 / 2 + v2 / 2
e4α(α) = -v3 / 2 + v4 / 2
transfinite_blend(e1, e2, e3, e4, e1α, e2α, e3α, e4α, r, s)
end
#}}}