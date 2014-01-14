__kernel void matinv(
	const __global float* M,
	__global float* invM
	) 
{

	unsigned int read_idx = get_global_id(0) + get_global_id(1) * get_global_size(0);
	unsigned int write_idx = get_global_id(1) + get_global_id(0) * get_global_size(1);

	invM[write_idx] = M[read_idx];

}