__kernel void opt_vecadd(
	const __global float* V1,
	const __global float* V2,
	__global float* P,
	const uint SIZE
	)
{
	uint start = 4 * get_global_id(0);
	uint stop = SIZE/4;
	int stride = get_local_size(0) * get_num_groups(0) * 4;

	for(uint i = start; i < stop; i += stride)
		P[i] = V1[i] + V2[i];
}
