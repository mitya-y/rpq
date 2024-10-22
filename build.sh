cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -B build -S .
cmake --build build -j10
