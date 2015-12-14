#include <cstdio>
#include <iostream>

#include "compute/compute.hpp"


int main()
{
	float test[4] = {1, 2, 3, 4};
	float test2[4] = {5, 6, 7, 8};
	float ret[4];
	Compute c("add", CL_DEVICE_TYPE_CPU);

	c.set_buffer(test, 4 * sizeof(float));
	c.set_buffer(test2, 4 * sizeof(float));
	c.set_ret_buffer(ret, 4 * sizeof(float));
	c.run(4);

	for(auto &i: ret)
		std::cout << i << std::endl;

    c.reset_buffer();

    test[0] = 100;
    c.set_buffer(test, 4 * sizeof(float));
	c.set_buffer(test2, 4 * sizeof(float));
	c.set_ret_buffer(ret, 4 * sizeof(float));
	c.run(4);
	for(auto &i: ret)
		std::cout << i << std::endl;
}
