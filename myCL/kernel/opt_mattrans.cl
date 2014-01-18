__kernel void opt_mattrans(
	const __global float* M,
	__global float* transM
	) 
{
	uint read_idx = get_global_id(0) + get_global_id(1) * get_global_size(0);
	uint write_idx = get_global_id(1) + get_global_id(0) * get_global_size(1);
	
	transM[write_idx] = M[read_idx];
}
