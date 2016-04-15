#ifndef KERNEL_HPP
#define KERNEL_HPP


const std::string Compute::kernel_src = R"(

kernel void add(global float *a, global float *b, global float *ret)
{
    const int row = get_global_id(0);

    ret[row] = a[row] + b[row];
}


kernel void WeakLearn(
    global read_only float *pf_matrix, global read_only float *nf_matrix,
    global read_only float *pf_weight, global read_only float *nf_weight,
    global read_only int *pf_shape,    global read_only int *nf_shape,
    global write_only float *ret_matrix)
{
    /*
     * :param pf_matrix:
     *         float matrix, feature_size x sample_size
     * :param ret_matrix:
     *      feature_size x 3 (error, polarity, theta)
     * */
    const int row = get_global_id(0);

    const int pf_col = *(pf_shape + 1);
    const int nf_col = *(nf_shape + 1);

    global float *pf = pf_matrix + row * pf_col;
    global float *nf = nf_matrix + row * nf_col;
    global float *ret = ret_matrix + row * 3;

    float max_ = *pf;
    float min_ = max_;
    const int slice = 10;

    float error = 1;
    float theta = 0;
    float polarity = 1;

    // find the max/min from (pf + nf)
    #pragma unroll
    for (private int i=1; i<pf_col; ++i)
    {
        max_ = max(max_, pf[i]);
        min_ = min(min_, pf[i]);
    }

    #pragma unroll
    for (private int i=0; i<nf_col; ++i)
    {
        max_ = max(max_, nf[i]);
        min_ = min(min_, nf[i]);
    }

    #pragma unroll
    for (private int i=1; i<slice; ++i)
    {
        float theta1 = (max_ - min_) * i / (slice - 1) + min_;
        float error1 = 0;
        float polarity1 = 1;
        /*
            *   negative data  |  positive data
            *                  |
            *                theta
            *                  |
            *  polarity = -1 <-|-> polarity = 1
            */

        #pragma unroll
        for (private int j=0; j<pf_col; ++j)
                error1 += pf_weight[j] * (pf[j] < theta1);

        #pragma unroll
        for (private int j=0; j<nf_col; ++j)
                error1 += nf_weight[j] * (nf[j] > theta1);

        /* if (error1 > 0.5) */
            polarity1 = select(1, -1, (error1 > 0.5));
            error1 = select(error1, (1 - error1), (error1 > 0.5));

        /* if (error1 < error) */
            private int cmp = isless(error1, error);
            error = select(error, error1, cmp);
            polarity = select(polarity, polarity1, cmp);
            theta = select(theta, theta1, cmp);
    }

    // return
    ret[0] = error;
    ret[1] = polarity;
    ret[2] = theta;
}


kernel void JointLearn(
    global read_only float *pf_matrix, global read_only float *nf_matrix,
    global read_only float *pw,        global read_only float *nw,
           read_only int fn,                  read_only int pf_sn,
           read_only int nf_sn,        global read_only int *clist,
    global read_only float *q_map,     global write_only float *ret_matrix)
{
    /*
     * :param pf_matrix:
     *      float matrix, feature_size x sample_size
     * :param ret_matrix:
     *      feature_size x 3 (error, polarity, theta)
     * :param pw: positive data weight
     * :param nw: negative data weight
     * :param fn: feature size
     * :param pf_sn: positive smaple size
     * :param nf_sn: negative sample size
     * :param clist: combination list
     *      matrix; shape = C(n, 2) x 2
     *      e.g.
     *      {
     *          {1, 2},
     *          {1, 3},
     *          ...
     *      }
     * :param ret_matrix:
     *      1-D array of errors, len = C(n, 2)
     * :param q_map:
     *      2-D array of quartile
     *      e.g.
     *         Q1, Q2, Q3
     *      f1
     *      f2
     *      ..
     *      fn
     * */
    const int idx = get_global_id(0);
    const int cn2 = (fn * (fn - 1)) / 2;
    const int total_sn = pf_sn + nf_sn;

    const int ftr_x_idx = clist[idx * 2 + 0];  // index of feature x
    const int ftr_y_idx = clist[idx * 2 + 1];  // index of feature y

    global read_only float *pf_x = pf_matrix + (ftr_x_idx * pf_sn);
    global read_only float *pf_y = pf_matrix + (ftr_y_idx * pf_sn);
    global read_only float *nf_x = nf_matrix + (ftr_x_idx * nf_sn);
    global read_only float *nf_y = nf_matrix + (ftr_y_idx * nf_sn);

    private float vote_map[4][4] = {};  // init with all zeros

    global read_only float *ret = ret_matrix + idx;

    /* voting */
    for (size_t i=0; i<pf_sn; i++)
        vote_map[(int)pf_x[i]][(int)pf_y[i]] += pw[i];
    for (size_t i=0; i<nf_sn; ++i)
        vote_map[(int)nf_x[i]][(int)nf_y[i]] -= nw[i];

    /* calculate error, place it into ret_matrix */
    *ret = 0;
    for (size_t i=0; i<pf_sn; ++i)
        *ret += pw[i] * (vote_map[(int)pf_x[i]][(int)pf_y[i]] < 0);

    for (size_t i=0; i<nf_sn; ++i)
        *ret += nw[i] * (vote_map[(int)nf_x[i]][(int)nf_y[i]] >= 0);
}
)";


#endif /* end of include guard: KERNEL_HPP */
