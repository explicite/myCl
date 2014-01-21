__kernel void opt_mat_mul (
   const __global float* A, 
   const __global float* B, 
   __global float* C,
   const uint N			       
   ) 
{ 
  uint i;
  uint row = get_global_id(1); 
  uint col = get_global_id(0); 
  float temp = 0.0;
  for (i = 0; i < N; i++) {
    temp += A[row * N + i] * B[i * N + col]; 
  }

  C[row * N + col] = temp; 
}
