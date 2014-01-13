__kernel void vva(
	const __global float* V1,
	const __global float* V2,
	__global float* P,
	unsigned int size
	) {
	int read_idx = get_global_id(0) + get_global_id(1) * size;
	int write_idx = get_global_id(1) + get_global_id(0) * size;
	P[write_idx] = V1[read_idx] + V2[read_idx];
}