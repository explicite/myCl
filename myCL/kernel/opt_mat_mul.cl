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

    C_local[local_row * BLOCK_SIZE + local_col] = 
		A_local[local_row*BLOCK_SIZE+i]*B_local[i*BLOCK_SIZE+local_col] +
		A_local[local_row*(BLOCK_SIZE+1)+i]*B_local[(i*BLOCK_SIZE+1)+local_col] +
		A_local[local_row*(BLOCK_SIZE+2)+i]*B_local[(i*BLOCK_SIZE+2)+local_col] +
		A_local[local_row*(BLOCK_SIZE+3)+i]*B_local[(i*BLOCK_SIZE+3)+local_col] +
		A_local[local_row*(BLOCK_SIZE+4)+i]*B_local[i*(BLOCK_SIZE+4)+local_col] +
		A_local[local_row*(BLOCK_SIZE+5)+i]*B_local[(i*BLOCK_SIZE+5)+local_col] +
		A_local[local_row*(BLOCK_SIZE+6)+i]*B_local[(i*BLOCK_SIZE+6)+local_col] +
		A_local[local_row*(BLOCK_SIZE+7)+i]*B_local[(i*BLOCK_SIZE+7)+local_col] +
		A_local[local_row*(BLOCK_SIZE+8)+i]*B_local[i*(BLOCK_SIZE+8)+local_col] +
		A_local[local_row*(BLOCK_SIZE+9)+i]*B_local[(i*BLOCK_SIZE+9)+local_col] +
		A_local[local_row*(BLOCK_SIZE+10)+i]*B_local[(i*BLOCK_SIZE+10)+local_col] +
		A_local[local_row*(BLOCK_SIZE+11)+i]*B_local[(i*BLOCK_SIZE+11)+local_col] +
		A_local[local_row*(BLOCK_SIZE+12)+i]*B_local[i*(BLOCK_SIZE+12)+local_col] +
		A_local[local_row*(BLOCK_SIZE+13)+i]*B_local[(i*BLOCK_SIZE+13)+local_col] +
		A_local[local_row*(BLOCK_SIZE+14)+i]*B_local[(i*BLOCK_SIZE+14)+local_col] +
		A_local[local_row*(BLOCK_SIZE+15)+i]*B_local[(i*BLOCK_SIZE+15)+local_col];
    
	barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[row * N + col] = C_local[local_row * BLOCK_SIZE + local_col]; 
}
