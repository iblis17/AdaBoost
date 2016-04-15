#include <cstdio>
#include <iostream>
#include <cassert>


#include "compute/compute.hpp"

#ifdef GPU
    #define DEV_TYPE CL_DEVICE_TYPE_GPU
#else
    #define DEV_TYPE CL_DEVICE_TYPE_CPU
#endif


int main()
{
	float test[4] = {1, 2, 3, 4};
	float test2[4] = {5, 6, 7, 8};
	float ret[4];
	Compute c("add", DEV_TYPE);

	c.set_buffer(test, 4 * sizeof(float));
	c.set_buffer(test2, 4 * sizeof(float));
	c.set_ret_buffer(ret, 4 * sizeof(float));
	c.run(4);

	for(auto &i: ret)
        std::cout << i << std::endl;
    assert(ret[0] == 6);
    assert(ret[1] == 8);
    assert(ret[2] == 10);
    assert(ret[3] == 12);

    // test for reset_buffer
    test[0] = 100;
    c.reset_buffer(0, test);
	c.run(4);

    for(auto &i: ret)
        std::cout << i << std::endl;
    assert(ret[0] == 105);
    assert(ret[1] == 8);
    assert(ret[2] == 10);
    assert(ret[3] == 12);

}
