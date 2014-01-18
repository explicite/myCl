__kernel void matvecmul(
	const __global float* M,  
	const __global float* V, 
	__global float* P
			       ) 
{ 
	P[get_global_id(0)] = M[get_global_id(0) + get_global_size(1) * get_global_id(1)] * V[get_global_id(1)];
}
