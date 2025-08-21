struct Tensor {
    RWStructuredBuffer<float> data;
    StructuredBuffer<int> shape;
    StructuredBuffer<int> strides;
}

[shader("compute")]
[numthreads(32)]
void add(
    uint3 threadId: SV_DispatchThreadID,
    uniform Tensor a,
    uniform Tensor b,
    uniform Tensor c
) {
    uint index = threadId.x;
    c.data[index] = a.data[index] + b.data[index];
}