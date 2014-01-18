__kernel void mattrans(
	const __global float* M,
	__global float* transM
	) 
{

	unsigned int read_idx = get_global_id(0) + get_global_id(1) * get_global_size(0);
	unsigned int write_idx = get_global_id(1) + get_global_id(0) * get_global_size(1);

	transM[write_idx] = M[read_idx];

}
