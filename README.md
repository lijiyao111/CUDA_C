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

 