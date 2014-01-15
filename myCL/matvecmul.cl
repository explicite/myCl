__kernel void matvecmul(
 const __global float* M,  
  const __global float* V, 
  __global float* P
			       ) 
{ 
	uint height = get_global_size(0);
	uint width = get_global_size(1);

	uint i = get_global_id(0);
	uint j = get_global_id(1);

	P[i] = M[i + width * j] * V[j];
}

