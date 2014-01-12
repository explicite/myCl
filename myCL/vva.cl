__kernel void vva(
				   const __global float* a,
				   const __global float* b,
				   __global float* c
				  )
{
	int idx = get_global_id(0);
	c[idx] = a[idx] + b[idx];
}