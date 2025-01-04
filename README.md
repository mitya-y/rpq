## GPU-based matrices implementation of Regular Path Query algorithm

# Configure and build
git submodule update --init --recursive  \
cmake -B build -S . -DRPQ_RUN_ON_CPU=OFF \
cmake --build build

# Run tests
./build/rpq test

