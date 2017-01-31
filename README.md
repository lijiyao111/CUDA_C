# GPU computing with CUDA (Nvidia GPU)

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
 