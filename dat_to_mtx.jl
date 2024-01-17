using MatrixMarket
using PETScBinaryIO
using DelimitedFiles

A = readpetsc("A_32.dat")[1]
b = readpetsc("b_32.dat")[1]
MatrixMarket.mmwrite("mat_32.mtx", A);
open("mat_32.mtx", "a") do io
    writedlm(io, b)
end


A_1024 = readpetsc("A_1024.dat")[1]
b_1024 = readpetsc("b_1024.dat")[1]
MatrixMarket.mmwrite("mat_1024.mtx", A_1024);
open("mat_1024.mtx", "a") do io
    writedlm(io, b_1024)
end
# { cat header.mtx; tail -n +2 mat_1024.mtx; } > mat_1024_combined.mtx

A_2048 = readpetsc("A_2048.dat")[1]
b_2048 = readpetsc("b_2048.dat")[1]
MatrixMarket.mmwrite("mat_2048.mtx", A_2048);
open("mat_2048.mtx", "a") do io
    writedlm(io, b_2048)
end



A_4096 = readpetsc("A_4096.dat")[1]
b_4096 = readpetsc("b_4096.dat")[1]
MatrixMarket.mmwrite("mat_4096.mtx", A_4096);
open("mat_4096.mtx", "a") do io
    writedlm(io, b_4096)
end

# { sed -n '1,$p' header.mtx; sed -n '2,$p' mat_4096.mtx; } > mat_4096_combined.mtx



A_8192 = readpetsc("A_8192.dat")[1]
b_8192 = readpetsc("b_8192.dat")[1]
MatrixMarket.mmwrite("mat_8192.mtx", A_8192);
open("mat_8192.mtx", "a") do io
    writedlm(io, b_8192)
end
# { sed -n '1,$p' header.mtx; sed -n '2,$p' mat_8192.mtx; } > mat_8192_combined.mtx
