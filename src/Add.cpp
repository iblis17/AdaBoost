#include <cstdio>
#include <iostream>

#include "compute/compute.hpp"
#include "compute/kernel.hpp"


int main()
{
	float test[4] = {1, 2, 3, 4};
	float test2[4] = {5, 6, 7, 8};
	float ret[4];
	Compute c(4);

	c.set_buffer(test);
	c.set_buffer(test2);
	c.set_ret_buffer(ret);
	c.add();

	for(auto &i: ret)
		std::cout << i << std::endl;
}
