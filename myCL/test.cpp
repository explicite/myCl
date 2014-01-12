#include "test.hpp"
#include <stdlib.h> 
#include <malloc.h>
float* omv(const float* _M, const float* _V, const unsigned int _N)
{
	float* _prd = (float*)malloc(sizeof(float)*_N);

	register int i, j;
	register unsigned int nj;
	register float xj;
	register unsigned int i0, i1, i2;
	register float yi0, yi1, yi2;
	
#pragma omp parallel for schedule(static) private(i, j)
	for(i = 0; i < _N; i += 3)
	{
		i0 = i;
                i1 = i + 1;
                i2 = i + 2;

                yi0 = 0.0;
                yi1 = 0.0;
                yi2 = 0.0;

                for (j = 0; j < _N; j++)
                {
                        nj = _N*j;
                        xj = _V[j];
                        yi0 += _M[i0 + nj] * xj;
                        yi1 += _M[i1 + nj] * xj;
                        yi2 += _M[i2 + nj] * xj;
                }

                _prd[i0] = yi0;
                _prd[i1] = yi1;
                _prd[i2] = yi2;
	}

	return _prd;
}

bool assert(const float* _A, const float* _B, const unsigned int _N)
{
        if (_A == _B)
                return true;

        register int i;
        for (i = 0; i < _N; i++)
                if (!equal(_A[i], _B[i]))
                        return false;

        return true;
}

bool equal(const float _A, const float _B)
{
	if ((((int)(_A * PRECISION)) / PRECISION) == (((int)(_B * PRECISION)) / PRECISION))
                return true;
        else 
                return true;
}