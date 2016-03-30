#include <cstdio>
#include <iostream>
#include <algorithm>

#include "compute/compute.hpp"

#ifdef GPU
    #define DEV_TYPE CL_DEVICE_TYPE_GPU
#else
    #define DEV_TYPE CL_DEVICE_TYPE_CPU
#endif

int main()
{
	const int fn = 3;		// feature number
	const int pf_sn = 15;	// positive sample size
	const int nf_sn = 14;	// negative sample size
    const int total_sn = pf_sn + nf_sn;
    const int cn2 = fn * (fn - 1) / 2;

	float pf[fn][pf_sn] = {
        {1, 2, 7, 7, 7, 4, 5, 8, 6, 5, 9, 10, 3, 6, 8},
        {7, 9, 3, 4, 6, 2, 2, 0, 1, 3, 4, 5, 7, 10, 2},
        {3, 6, 6, 6, 2, 4, 3, 1, 5, 8, 9, 5, 7, 9, 3}
    };
	float nf[fn][nf_sn] = {
        {9, 1, 3, 4, 6, 2, 2, 2, 2, 7, 0, 8, 1, 3},
        {7, 7, 4, 6, 3, 5, 6, 3, 2, 8, 9, 3, 1, 2},
        {8, 5, 0, 10, 10, 2, 4, 1, 3, 5, 7, 2, 0, 9}
    };

    float pw[pf_sn];
    float nw[nf_sn];

    int clist[cn2][2] = { /* combination table*/
        {0, 1},
        {0, 2},
        {1, 2},
    };
    float q_map[fn][3] = {};
    float ret[cn2] = {};

    /* init pw and nw */
    float init_weight = 1.0 / (total_sn);
    for (auto i=0; i<pf_sn; ++i)
        pw[i] = init_weight;
    for (auto i=0; i<nf_sn; ++i)
        nw[i] = init_weight;

    /* calculate quartile map */
	for (int i=0; i<fn; i++)
	{
		float arr[total_sn];

		for (int z=0; z<pf_sn;z++)
			arr[z] = pf[i][z];
		for (int z=0; z<nf_sn; z++)
			arr[z + pf_sn] = nf[i][z];

        std::sort(arr, arr + total_sn);

		float q1_Index = (total_sn + 1) / 4.0;
		float q2_Index = (total_sn + 1) / 2.0;
		float q3_Index = (total_sn + 1) / 4.0 * 3.0;

		if ((total_sn + 1) % 4 == 0)
		{
			q_map[i][0] = arr[(int)q1_Index-1];
			q_map[i][2] = arr[(int)q3_Index-1];
		}
		else
		{
			q_map[i][0] = (arr[(int)q1_Index-1] + arr[(int)q1_Index])/2;
			q_map[i][2] = (arr[(int)q3_Index-1] + arr[(int)q3_Index])/2;
		}

		if ((total_sn + 1) % 2 == 0)
			q_map[i][1] = arr[(int)q2_Index-1];
		else
			q_map[i][1] = (arr[(int)q2_Index-1] + arr[(int)q2_Index]) / 2;

        for (auto j=0; j<pf_sn; ++j)
        {
           if (pf[i][j] < q_map[i][0])
               pf[i][j] = 0;
           else if (pf[i][j] < q_map[i][1])
               pf[i][j] = 1;
           else if (pf[i][j] < q_map[i][2])
               pf[i][j] = 2;
           else
               pf[i][j] = 3;
        }
        for (auto j=0; j<nf_sn; ++j)
        {
           if (nf[i][j] < q_map[i][0])
               nf[i][j] = 0;
           else if (nf[i][j] < q_map[i][1])
               nf[i][j] = 1;
           else if (nf[i][j] < q_map[i][2])
               nf[i][j] = 2;
           else
               nf[i][j] = 3;
        }
	}


    /* opencl routine */
    Compute c("JointLearn", DEV_TYPE);

    c.set_buffer((float *)pf, fn * pf_sn * sizeof(float));
    c.set_buffer((float *)nf, fn * pf_sn * sizeof(float));

    c.set_buffer(pw, pf_sn * sizeof(float));
    c.set_buffer(nw, nf_sn * sizeof(float));

    c.set_buffer(fn);
    c.set_buffer(pf_sn);
    c.set_buffer(nf_sn);

    c.set_buffer(clist, cn2 * 2 * sizeof(int));
    c.set_buffer(q_map, fn * 3 * sizeof(float));

    c.set_ret_buffer((float *)ret, cn2 * sizeof(float));

    c.run(3);

    std::cout << ret[0] << std::endl;
    std::cout << ret[1] << std::endl;
    std::cout << ret[2] << std::endl;

    c.reset_buffer();  // obj `c` can reuse for next `set_buffer` and `run`

    return 0;
}
