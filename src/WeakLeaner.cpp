#include <cstdio>
#include <iostream>

#include "compute/compute.hpp"


int main()
{
    float pf[1][4] = {
        {50, 20, 84, 30}
    };
    float pw[1][4] = {
        {0.1, 0.1, 0.1, 0.1}
    };
    float nf[1][6] = {
        {42, 80, 104, 5, 70, 65}
    };
    float nw[1][6] = {
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
    };
    int pf_shape[2] = { 1, 4 };
    int nf_shape[2] = { 1, 6 };
    float ret[1][3];

    Compute c("WeakLearn", CL_DEVICE_TYPE_CPU);

    c.set_buffer((float *)pf, 1 * 4 * sizeof(float));
    c.set_buffer((float *)nf, 1 * 6 * sizeof(float));

    c.set_buffer((float *)pw, 1 * 4 * sizeof(float));
    c.set_buffer((float *)nw, 1 * 6 * sizeof(float));

    c.set_buffer((int *)pf_shape, 2 * sizeof(float));
    c.set_buffer((int *)nf_shape, 2 * sizeof(float));

    c.set_ret_buffer((float *)ret, 1 * 3 * sizeof(float));

    c.run(1, 4);

    std::cout << ret[0][0] << std::endl;
    std::cout << ret[0][1] << std::endl;
    std::cout << ret[0][2] << std::endl;
}
