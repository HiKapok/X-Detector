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
#include "common.h"

//__attribute__((always_inline))
// template<typename T, typename std::is_same<float, typename std::remove_cv<T>::type>::value>
void atomic_float_add(volatile float* ptr, const float operand)
{
    assert(is_aligned(ptr, 4));

    volatile int32_t* iptr = reinterpret_cast<volatile int32_t*>(ptr);
    int32_t expected = *iptr;

    while (true)
    {
        const float value = binary_cast<float>(expected);
        const int32_t new_value = binary_cast<int32_t>(value + operand);
        const int32_t actual = __sync_val_compare_and_swap(iptr, expected, new_value);
        if (actual == expected)
            return;
        expected = actual;
    }
}
