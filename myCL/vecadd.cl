__kernel void vecadd(
	const __global float* V1,
	const __global float* V2,
	__global float* sum
	) 
{
	unsigned int idx = get_global_id(0);
	sum[idx] = V1[idx] + V2[idx];
}

