#include <cstdio>
#include <iostream>

#include "compute/compute.hpp"
#include "compute/kernel.hpp"


float* WeakLearn(float pf1[], float nf1[], float pw[], float nw[], int pf1_sn, int nf1_sn);


int main()
{
	float pf[4][1] = { { 50 }, { 20 }, { 84 }, { 30 } };
	float pw[4][1] = { { 0.1 }, { 0.1 }, { 0.1}, { 0.1 } };
	float nf[6][1] = { { 42 }, { 80 }, { 104 }, { 5 }, { 70 }, {65} };
	float nw[6][1] = { { 0.1 }, { 0.1 }, { 0.1 }, { 0.1 }, { 0.1 }, { 0.1 } };
	float *output = WeakLearn(pf[0], nf[0], pw[0], nw[0] ,4, 6);
	printf("error:%f polarity:%f theta:%f", output[0], output[1], output[2]);
}


float* WeakLearn(float pf1[], float nf1[], float pw[], float nw[], int pf1_sn, int nf1_sn)
{
	float max = 0, min = 20000000000000, error = 1, theta=0, polarity=1;
	//找最大 最小值
	for (int i = 0; i < pf1_sn; i++)
	{
		if (pf1[i] > max)
		{
			max = pf1[i];
		}
		if (pf1[i] < min)
		{
			min = pf1[i];
		}
	}
	for (int i = 0; i < nf1_sn; i++)
	{
		if (nf1[i] > max)
		{
			max = nf1[i];
		}
		if (nf1[i] < min)
		{
			min = nf1[i];
		}
	}
	//找最好的一刀
	for (int j = 1; j < 10; j++)
	{
		float theta1 = (max - min) / 10 * j;
		float error1 = 0;
		float polarity1 = 1;
		for (int i = 0; i < pf1_sn; i++)
		{
			if (pf1[i] < theta1)
			{
				error1 =error1+ pw[i];
			}
		}
		for (int i = 0; i < nf1_sn; i++)
		{
			if (nf1[i] > theta1)
			{
				error1 = error1+nw[i];
			}
		}
		if (error1 > 0.5)
		{
			polarity1 = -1;
			error1 = 1 - error1;
		}
		if (error1 < error)
		{
			error = error1;
			polarity = polarity1;
			theta = theta1;
		}
	}
	float output[] = {error, polarity, theta};
	return output;
}
