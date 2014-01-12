__kernel void mv(	
				  const __global float* a, 
				  const __global float* b, 
				  __global float* c,
				  unsigned int width, 
				  unsigned int height
				 ) 
{ 
  unsigned int y = get_global_id(0); 
  const __global float* row = a + y * width; 
  float dotProduct = 0; 
  for (unsigned int x = 0; x < width; ++x) 
    dotProduct += row[x] * b[x]; 
  c[y] = dotProduct; 
}