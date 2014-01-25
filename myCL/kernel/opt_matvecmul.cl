__kernel void opt_matvecmul (
  const __global float* M, 
  const uint WIDTH, 
  const uint HEIGHT, 
  const __global float* V, 
  __global float* P, 
  __local float* pdt
  ) 
{ 
  for (uint y = get_group_id(0); y < HEIGHT; y += get_num_groups(0)) { 
    const __global float* row = M + y * WIDTH; 

    float sum = 0; 
    for (uint x = get_local_id(0); x < WIDTH; x += get_local_size(0)) 
      sum += row[x] * V[x]; 

    pdt[get_local_id(0)] = sum; 

    for (uint stride = get_local_size(0)/2; stride > 0; stride /= 2) { 

      barrier(CLK_LOCAL_MEM_FENCE); 

      if (get_local_id(0) < stride) { 
                pdt[get_local_id(0)] += pdt[get_local_id(0) + stride]; 
      } 

    } 

    if (get_local_id(0) == 0) P[y] = pdt[0]; 

    barrier(CLK_LOCAL_MEM_FENCE); 

  } 
}