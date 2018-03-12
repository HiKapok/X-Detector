// MIT License

// Copyright (c) 2018 Changan Wang

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef COMMON_H_
#define COMMON_H_

#include <cstdlib>
#include <cassert>
#include <cstdint>

// atomic addition for float using gcc built-in functions for atomic memory access
// this code snippet borrowed from https://codereview.stackexchange.com/questions/135852/atomic-floating-point-addition
template <typename Target, typename Source>
__attribute__((always_inline)) Target binary_cast(Source s)
{
    static_assert(sizeof(Target) == sizeof(Source), "binary_cast: 'Target' must has the same size as 'Source'");
    union
    {
        Source  m_source;
        Target  m_target;
    } u;

    u.m_source = s;
    return u.m_target;
}

template <typename T>
__attribute__((always_inline)) bool is_pow2(const T x)
{
    return (x & (x - 1)) == 0;
}

template <typename T>
__attribute__((always_inline)) bool is_aligned(const T ptr, const size_t alignment)
{
    assert(alignment > 0);
    assert(is_pow2(alignment));

    const uintptr_t p = (uintptr_t)ptr;
    return (p & (alignment - 1)) == 0;
}

extern void atomic_float_add(volatile float* ptr, const float operand);

#endif // COMMON_H_
