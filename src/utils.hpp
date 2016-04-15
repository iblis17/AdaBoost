#ifndef _ADABOOST_UTILS_HPP
#define _ADABOOST_UTILS_HPP

#include <cmath>

template <typename T>
void assert_float(const T &f1, const T &f2)
{
    T epsilon = std::numeric_limits<T>::epsilon();

    if (std::fabs(f1 - f2) >= epsilon)
        throw "AssertionError";
}

template <typename T, const size_t row, const size_t col>
void assert_2d_arr(T (&arr)[row][col],
                   T (&exp_arr)[row][col],
                   std::string msg)
{
    std::cout << "[Assert] checking " << msg << " ...";
    for (auto i=0; i<row; ++i)
        for (auto j=0; j<col; ++j)
            if (arr[i][j] != exp_arr[i][j])
            {
                printf("\n\tAssertionError: arr[%d][%d] != expect[%d][%d]\n\t",
                       i, j, i, j);
                std::cout << arr[i][j] << " != " << exp_arr[i][j] << std::endl;
                throw;
            }
    std::cout << "done" << std::endl;
}

#endif  // _ADABOOST_UTILS_HPP
