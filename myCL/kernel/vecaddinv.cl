__kernel void vecaddinv(
	const __global float* V1,
	const __global float* V2,
	__global float* P
	) 
{
	unsigned int read_idx = get_global_id(0) + get_global_id(1) * get_global_size(0);
	unsigned int write_idx = get_global_id(1) + get_global_id(0) * get_global_size(1);

	P[write_idx] = V1[read_idx] + V2[read_idx];
}
