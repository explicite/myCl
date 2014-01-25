#define BLOCK_SIZE 16 

__kernel void opt_mat_mul (
   const __global float* A, 
   const __global float* B, 
   __global float* C,
   const uint N			       
   ) 
{ 
  uint i, iblock;
  uint row = get_global_id(1); 
  uint col = get_global_id(0); 
  uint local_row = get_local_id(1); 
  uint local_col = get_local_id(0); 
  uint row_block_id = get_group_id(1);
  uint col_block_id = get_group_id(0);

  __local float A_local[BLOCK_SIZE*BLOCK_SIZE]; 
  __local float B_local[BLOCK_SIZE*BLOCK_SIZE]; 
  __local float C_local[BLOCK_SIZE*BLOCK_SIZE]; 

  C_local[local_row * BLOCK_SIZE + local_col] = 0.0f;

  float temp = 0.0;
  int nr_blocks = N/BLOCK_SIZE;
  for(iblock = 0; iblock < nr_blocks; iblock++){
    A_local[local_row * BLOCK_SIZE + local_col] = 
      A[row * N + iblock*BLOCK_SIZE + local_col]; 
    B_local[local_row * BLOCK_SIZE + local_col] = 
      B[(local_row+iblock*BLOCK_SIZE) * N + col];

    barrier(CLK_LOCAL_MEM_FENCE);

    for(i=0; i< BLOCK_SIZE; i++){
      temp += A_local[local_row*BLOCK_SIZE+i]*B_local[i*BLOCK_SIZE+local_col];
    }
    C_local[local_row * BLOCK_SIZE + local_col] += temp; 
    
	barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[row * N + col] = C_local[local_row * BLOCK_SIZE + local_col]; 
}
