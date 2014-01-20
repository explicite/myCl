#include "test.hpp"
#include <stdlib.h> 
#include <malloc.h>

float* mat_vec(const float* _M, const float* _V, const unsigned int _N)
{
	float* _prd = (float*)malloc(sizeof(float)*_N);

	register int i, j;

#pragma omp parallel for schedule(static) private(i, j)
	for(i = 0; i < _N; i++)
	{
		for (j = 0; j < _N; j++)
		{
			_prd[i] += _M[i+ j *_N] * _V[j];
		}
	}

	return _prd;
}

float* mat_mul(const float* _A, const float* _B, const unsigned int _N)
{
	float* _prd = (float*)malloc(sizeof(float)*_N*_N);

	register int i, j, k, ii, jj; 
	register unsigned int BLSR, BLSC;

	BLSR = 4;
	BLSC = 2;

	if (_N < 100)
		BLSR = 1;
	BLSC = 1;

#pragma omp parallel for schedule(static) private(i, j, k, ii, jj)
	for (i = 0; i < _N; i += BLSR)
		for (j = 0; j < _N; j += BLSC)
			for (k = 0; k < _N; k++)
				for (ii = i; ii < i + BLSR; ii++)
					for (jj = j; jj < j + BLSC; jj++)
						_prd[ii + jj*_N] += _A[ii + k*_N] * _B[k + jj*_N];


	return _prd;
}

float* add(const float* _V1, const float* _V2, const unsigned int _N)
{
	float* sum = (float*)malloc(sizeof(float)*_N);

	register int i;
#pragma omp parallel for schedule(static) private(i)
	for(i = 0; i < _N; i++)
		sum[i] = _V1[i] + _V2[i];

	return sum;
}

bool assert_inv(const float* _A, const float* _B, const unsigned int _N)
{
	register int i, j;
	for (i = 0; i < _N; i++)
		for (j = 0; j < _N; j++)
			if(!equal(_A[i + (j * _N)], _B[j + (i * _N)]))
				return false;

	return true;
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
		return false;
}
