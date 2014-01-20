#pragma once
#define PRECISION 10000.0

float* mat_vec(const float* _M, const float* _V, const unsigned int _N);

float* mat_mul(const float* _A, const float* _B, const unsigned int _N);

float* add(const float* _V1, const float* _V2, const unsigned int _N);

bool assert_inv(const float* _A, const float* _B, const unsigned int _N);

bool assert(const float* _A, const float* _B, const unsigned int _N);

bool equal(const float _A, const float _B);