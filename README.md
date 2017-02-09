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

> I also worked on the homework assignments of Udacity GPU class cs344. Check my Solutions [here](https://github.com/lijiyao111/Udacity_CUDA_GPU_cs344).
> Udacity GPU class projects:
> 
> 1. Map
> 2. 2D Stencil
> 3. Histogram, Reduce, small Scan
> 4. Histogram, Compact, large Scan, Radix sort



### Hello World Example
> Code: `hello.cu`

First program in CUDA. Say hello. This is to understand the block-thread structure.

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
Do some simple calculation with CUDA. The key idea is to let each thread do one job if possible. Spread the work to all the thread. Note: Memory initialization in Host, Device and memory copy between Host and Device is really verbose...

> Code: `vectorAdd.cu`   Allows 1 block

```c
// Kernal, call on Host, run on Device
__global__ 
void vectoradd(int* a, int *b, int *c){
    c[threadIdx.x]=a[threadIdx.x]+b[threadIdx.x];
}
```

> Code: `vectorAdd2.cu`   Allows many blocks

```c
__global__ 
void vectoradd2(int* a, int *b, int *c, int N){
    int idx=threadIdx.x + blockIdx.x*blockDim.x;
    if(idx<N)
    c[idx]=a[idx]+b[idx];
}
```
 
### Matrix Multiply
My code allows arbitrary size of the Matrix. Code is modified from the Example in [Nvidia CUDA guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).

- Using global memory
> Code: Matrix_multiply_gmem.cu

- Using shared memory
> Code: Matrix_multiply_shmem.cu

### Reduce, 1 block vs arbitrary block
Number of threads in a block need to be power of 2 (automatically taken care of in the code). Otherwise result is not correct. Check Udacity GPU class note.

Use shared memory for performance. Shared memory assigned when calling the kernal.`kernal<<<blockDim, threadDim, sharedMemSpace>>>()`

- Reduce with small input (1 block)

> Code: reduce_small.cu

- Reduce with large input (many blocks)

> Code: reduce_large.cu

Use a temporary array to store the reduced result from each block.

### Scan, 1 block vs arbitrary block
Interestingly, the scan code I studied from Nvidia website (http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html) has bug and does not work correctly. How could you do this ```temp[pout*n+thid] += temp[pin*n+thid - offset]; ```? It should be ```temp[pout*n+thid] = temp[pout*n+thid] + temp[pin*n+thid - offset]; ```.

- Scan with small input (1 block)

> Code: scan_small.cu

Scan with 1 block due to small number of elements to scan. 


- Scan with large input (many blocks)

> Code: scan_large.cu

Use a temporary array to store the exclusively-scanned histogram from each block.

### Histogram, atomic add
> Code: atomic_histogram.cu

AtomicAdd() is making the parallel code serial, significantly slow the code. Here is some algorithm to make it fast (https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/). 

### Unified memory
> Code: unified_memory.cu

This is a pain-killer to significantly simply the memory allocation-copy process!

It is instructive to compare zero-copy and unified memory. For the former, the memory is allocated in
page-locked fashion on the host. A device thread has to reach out to get the data. No guarantee of
coherence is provided as, for instance, the host could change the content of the pinned memory while the
device reads its content.

### Stencil 1D
> Code: Stencil1d_sum.cu

Do a 1D stencil sum. Use shared memory for performance.

### Radix Sort
> Code: radix_sort.cu

Not easy to do compared with the CPU version... Use histogram, compact, scan, then move. Check [here](http://stackoverflow.com/questions/26206544/parallel-radix-sort-how-would-this-implementation-actually-work-are-there-some) for explanation of the radix sort. But my method is slightly different. 