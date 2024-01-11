using SparseArrays
using PETScBinaryIO

# manually loading A and b and merge them into a single file
# to be compatible with code in tutorials

my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []

matrix = readpetsc("A_32.dat");
vector = readpetsc("b_32.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_32.dat", my_vector);


my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []
matrix = readpetsc("A_64.dat");
vector = readpetsc("b_64.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_64.dat", my_vector);


my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []
matrix = readpetsc("A_128.dat");
vector = readpetsc("b_128.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_128.dat", my_vector);


my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []
matrix = readpetsc("A_256.dat");
vector = readpetsc("b_256.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_256.dat", my_vector);

my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []
matrix = readpetsc("A_512.dat");
vector = readpetsc("b_512.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_512.dat", my_vector);


my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []
matrix = readpetsc("A_1024.dat");
vector = readpetsc("b_1024.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_1024.dat", my_vector);


my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []
matrix = readpetsc("A_2048.dat");
vector = readpetsc("b_2048.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_2048.dat", my_vector);

my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []
matrix = readpetsc("A_4096.dat");
vector = readpetsc("b_4096.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_4096.dat", my_vector);


my_vector::Vector{Union{SparseArrays.SparseMatrixCSC, Vector}} = []
matrix = readpetsc("A_8192.dat");
vector = readpetsc("b_8192.dat");

push!(my_vector, matrix[1]);
push!(my_vector, vector[1]);

writepetsc("test_8192.dat", my_vector);
