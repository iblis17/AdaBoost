#ifndef KERNEL_HPP
#define KERNEL_HPP


const std::string Compute::kernel_src = R"(

kernel void add(global float *a, global float *b, global float *ret)
{
    const int row = get_global_id(0);

    ret[row] = a[row] + b[row];
}


kernel void WeakLearn(
    global read_only float *pf_maxtrix, global read_only float *nf_maxtrix,
    global read_only float *pf_weight,  global read_only float *nf_weight,
    global read_only int *pf_shape,     global read_only int *nf_shape,
           read_only int group_size,    global write_only float *ret_matrix)
{
    /*
     * :param pf_maxtrix:
     *         float matrix, feature_size x sample_size
     * :param ret_matrix:
     *      feature_size x 3 (error, polarity, theta)
     * */
    const int row_start = get_global_id(0);

    const int pf_col = *(pf_shape + 1);
    const int nf_col = *(nf_shape + 1);

    const int row_end = row_start + group_size;

    for(int row=row_start; row<row_end; row++)
    {
        global float *pf = pf_maxtrix + row * pf_col;
        global float *nf = nf_maxtrix + row * nf_col;
        global float *ret = ret_matrix + row * 3;

        float max_ = *pf;
        float min_ = max_;
        const int slice = 10;

        float error = 1;
        float theta = 0;
        float polarity = 1;

        // find the max/min from (pf + nf)
        for (int i=1; i<pf_col; ++i)
        {
            max_ = max(max_, pf[i]);
            min_ = min(min_, pf[i]);
        }
        for (int i=0; i<nf_col; ++i)
        {
            max_ = max(max_, nf[i]);
            min_ = min(min_, nf[i]);
        }

        for (int i=1; i<slice; ++i)
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

            for (int j=0; j<pf_col; ++j)
                if (*(pf + j) < theta1)
                    error1 += pf_weight[j];

            for (int j=0; j<nf_col; ++j)
                if (*(nf + j) > theta1)
                    error1 += nf_weight[j];

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

        // return
        ret[0] = error;
        ret[1] = polarity;
        ret[2] = theta;
    }
}
)";


#endif /* end of include guard: KERNEL_HPP */
