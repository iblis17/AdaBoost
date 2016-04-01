#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <iostream>

#include "compute/compute.hpp"

#ifdef GPU
    #define DEV_TYPE CL_DEVICE_TYPE_GPU
#else
    #define DEV_TYPE CL_DEVICE_TYPE_CPU
#endif

template <typename T, const size_t row, const size_t col>
void assert_2d_arr(T (&arr)[row][col],
                   T (&exp_arr)[row][col],
                   std::string msg);

template <typename T>
void assert_float(const T &f1, const T &f2);

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

    /* assertion */
    float exp_q_map[fn][3] = {
        {2, 5, 7},
        {2, 4, 7},
        {2.5, 5, 7.5}
    };
    assert_2d_arr(q_map, exp_q_map, "q_map");

    float expect_pf[fn][pf_sn] = {
        {0, 1, 3, 3, 3, 1, 2, 3, 2, 2, 3, 3, 1, 2, 3},
        {3, 3, 1, 2, 2, 1, 1, 0, 0, 1, 2, 2, 3, 3, 1},
        {1, 2, 2, 2, 0, 1, 1, 0, 2, 3, 3, 2, 2, 3, 1}};
    assert_2d_arr(pf, expect_pf, "pf");

    float expect_nf[fn][nf_sn] = {
        {3, 0, 1, 1, 2, 1, 1, 1, 1, 3, 0, 3, 0, 1},
        {3, 3, 2, 2, 1, 2, 2, 1, 1, 3, 3, 1, 0, 1},
        {3, 2, 0, 3, 3, 0, 1, 0, 1, 2, 2, 0, 0, 3}};
    assert_2d_arr(nf, expect_nf, "nf");

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
    /* end of opencl routine */

    std::cout.precision(std::numeric_limits<float>::max_digits10);

    std::cout << ret[0] << std::endl;
    assert_float(ret[0], (float)0.137931034);

    std::cout << ret[1] << std::endl;
    assert_float(ret[1], (float)0.172413796);

    std::cout << ret[2] << std::endl;
    assert_float(ret[2], (float)0.275862068);

    c.reset_buffer();  // obj `c` can reuse for next `set_buffer` and `run`

    return 0;
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

template <typename T>
void assert_float(const T &f1, const T &f2)
{
    T epsilon = std::numeric_limits<T>::epsilon();

    if (std::fabs(f1 - f2) >= epsilon)
        throw "AssertionError";
}
