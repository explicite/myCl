#pragma once
#define PRECISION 10000.0

float* omv(const float* _M, const float* _V, const unsigned int _N);

bool assert(const float* _A, const float* _B, const unsigned int _N);

bool equal(const float _A, const float _B);