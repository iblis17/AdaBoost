#include <cstdio>
#include <iostream>

#include "compute/compute.hpp"
#include "utils.hpp"

#ifdef GPU
    #define DEV_TYPE CL_DEVICE_TYPE_GPU
#else
    #define DEV_TYPE CL_DEVICE_TYPE_CPU
#endif


int main()
{
    float pf[2][4] = {
        {50, 20, 84, 30},
        {30, 50, 24, 11}
    };
    float pw[2][4] = {
        {0.1, 0.1, 0.1, 0.1},
        {0.1, 0.1, 0.1, 0.1}
    };
    float nf[2][6] = {
        {42, 80, 104, 5, 70, 65},
        {31, 25, 34, 91, 22, 15}
    };
    float nw[2][6] = {
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
    };
    int pf_shape[2] = { 2, 4 };
    int nf_shape[2] = { 2, 6 };
    int group_size = 2;
    float ret[2][3];

    Compute c("WeakLearn", DEV_TYPE);

    c.set_buffer((float *)pf, 2 * 4 * sizeof(float));
    c.set_buffer((float *)nf, 2 * 6 * sizeof(float));

    c.set_buffer((float *)pw, 2 * 4 * sizeof(float));
    c.set_buffer((float *)nw, 2 * 6 * sizeof(float));

    c.set_buffer((int *)pf_shape, 2 * sizeof(int));
    c.set_buffer((int *)nf_shape, 2 * sizeof(int));

    c.set_buffer(group_size);

    c.set_ret_buffer((float *)ret, 2 * 3 * sizeof(float));

    c.run(1);

    std::cout << ret[0][0] << std::endl;
    std::cout << ret[0][1] << std::endl;
    std::cout << ret[0][2] << std::endl;
    std::cout << ret[1][0] << std::endl;
    std::cout << ret[1][1] << std::endl;
    std::cout << ret[1][2] << std::endl;

    assert_float(ret[0][0], (float)0.3);
    assert_float(ret[0][1], (float)-1.0);
    assert_float(ret[0][2], (float)38.0);
    assert_float(ret[1][0], (float)0.4);
    assert_float(ret[1][1], (float)-1.0);
    assert_float(ret[1][2], (float)19.88888889);

    c.reset_buffer();  // obj `c` can reuse for next `set_buffer` and `run`
}
