# GPU computing with CUDA (Nvidia GPU)
-------
It is good to understand some basic ideas of GPU parallelism, and getting familiar with some parallelism algorithms. And I have coded some simple CUDA C examples. However, from my experience, it is quite difficult to master most CUDA parallelism algorithms because of the complexity. For example, sorting in CUDA is way more complicated than sorting in serial. To write large scale code, using CUDA libraries is a must. 

Some useful libraries in CUDA (either by Nvidia or third party):
1. cuBLAS -- BLAS
2. cuFFT -- 1D, 2D, 3D FFT
3. cuSPARSE -- BLAS-like routines for sparse matrix
4. cuRAND -- Pseudo- and quasi-random generation routines
5. NPP  -- Low-level image processing primitives
6. Magma -- GPU + multicore CPU LAPACK routines
7. CULA -- Eigensolvers, matrix factorizations and solvers
8. ArrayFire -- Framework for data-parallel array manipulation
9. cuDNN -- Deep Neural Networks

Lower-level libraries:
1. thrust -- like C++ STL  -- host-side interface, no kernels, cannot set thread parameters (e.g. number of blocks, number of threads, shared memory)
2. CUB -- more control
3. CUDPP -- CUDA Data Parallel Primitives Library


-------
Simple Code Examples:
1. Hello world
2. Vector Add v1 -- one block, v2 -- several blocks
3. Matrix Multiply, global mem and shared mem
4. Reduce, 1 block vs arbitrary block
5. Scan, 1 block vs arbitrary block
6. Histogram, atomic add
7. Unified memory
8. Stencil 1D
9. Radix Sort

Udacity GPU class projects:
1. Map
2. Histogram
3. Histogram, Reduce, small Scan
4. Histogram, Compact, large Scan, Radix sort


### Hello World Example
```c
int main() {
    int blocksize=2; int N=3;
    hello<<<blocksize,N>>>();
}
```

output:
```
Hello world! block ID 1, thread ID 0
Hello world! block ID 1, thread ID 1
Hello world! block ID 1, thread ID 2
Hello world! block ID 0, thread ID 0
Hello world! block ID 0, thread ID 1
Hello world! block ID 0, thread ID 2
```

### Vector Add 
Memory initialization in Host, Device and memory copy between Host and Device is really verbose...
- `vectorAdd.cu`

```c
// Kernal, call on Host, run on Device
__global__ 
void vectoradd(int* a, int *b, int *c){
    c[threadIdx.x]=a[threadIdx.x]+b[threadIdx.x];
}
```

- `vectorAdd2.cu`

```c
__global__ 
void vectoradd2(int* a, int *b, int *c, int N){
    int idx=threadIdx.x + blockIdx.x*blockDim.x;
    if(idx<N)
    c[idx]=a[idx]+b[idx];
}
```
 