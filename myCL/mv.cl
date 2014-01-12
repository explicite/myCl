__kernel void mv(
 const __global float* M, 
  unsigned int width, 
  unsigned int height, 
  const __global float* V, 
  __global float* W
			       ) 
{ 
  // Each work-item computes multiple elements of W 
  for (unsigned int y = get_global_id(0); y < height; y += get_global_size(0)) { 
    const __global float* row = M + y * width; 
    float dotProduct = 0.0; 
    for (unsigned int x = 0; x < width; ++x) 
      dotProduct += row[x] * V[x]; 
    W[y] = dotProduct; 
  } 
}