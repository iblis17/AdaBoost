import logging
import numpy as np

from itertools import combinations

log = logging.getLogger(__name__)


def gen_data(size=(3, 10)):
    return np.random.randint(0, 10, size)


def get_quartile(p_data: np.array, n_data: np.array, feature_size):
    ret = np.zeros((feature_size, 3))
    serise_size = len(p_data[0]) + len(n_data[0]) + 1
    # quartile index: Q1, Q2, Q3
    q_idx = np.array((serise_size / 4,
                      serise_size / 2,
                      serise_size * 3 / 4)) - 1

    print('quartile index:', q_idx)

    for f_n in range(feature_size):
        log.debug('get_quartile: f_n = {}'.format(f_n))
        log.debug('get_quartile: p_data[f_n] = {}'.format(p_data[f_n]))

        series = np.concatenate((p_data[f_n], n_data[f_n]))
        series.sort()

        print('feature {} series:\n'.format(f_n), series)

        ret[f_n] = tuple(series[int(i)] if i.is_integer()
                         else (series[int(i)] + series[int(i)+1]) / 2
                         for i in q_idx)

    print('quartile matrix:\n', ret)
    return ret


def vote(px_series, py_series,
         nx_series, ny_series,
         p_weight,  n_weight,
         qx_series, qy_series):
    '''
    :param px_series: 1 dimension array of positive data
    :param py_series: 1 dimension array of positive data
    :param nx_series: 1 dimension array of negative data
    :param ny_series: 1 dimension array of negative data
    :param p_weight: 1 dimension array of positive weight
    :param n_weight: 1 dimension array of negative weight
    :param qx_series: 1 dimension array of x axis quartile
    :param qy_series: 1 dimension array of y axis quartile
    '''
    ret_map = np.zeros((4, 4))

    def get_idx(series, q_series):
        series = series.copy()
        for i in range(len(series)):
            if series[i] < q_series[0]:
                series[i] = 0
            elif series[i] < q_series[1]:
                series[i] = 1
            elif series[i] < q_series[2]:
                series[i] = 2
            else:
                series[i] = 3
        return series

    px = get_idx(px_series, qx_series)
    py = get_idx(py_series, qy_series)
    nx = get_idx(nx_series, qx_series)
    ny = get_idx(ny_series, qy_series)
    print(px)
    print(py)
    print(nx)
    print(ny)

    for idx, _ in enumerate(px):
        ret_map[px[idx], py[idx]] += p_weight[idx]
    for idx, _ in enumerate(nx):
        ret_map[nx[idx], ny[idx]] -= n_weight[idx]

    print('voting map:\n', ret_map)
    print(ret_map >= 0)

    error = 0
    error += ((ret_map < 0)[px, py] * p_weight).sum()
    error += ((ret_map >= 0)[nx, ny] * n_weight).sum()
    print('error:', error)

    return ret_map >= 0, error


def main():
    feature_size = 3

    p_sample_size = 15
    p_data = np.array(
        [[1, 2, 7, 7, 7, 4, 5, 8, 6, 5, 9, 10, 3, 6, 8],
         [7, 9, 3, 4, 6, 2, 2, 0, 1, 3, 4, 5, 7, 10, 2],
         [3, 6, 6, 6, 2, 4, 3, 1, 5, 8, 9, 5, 7, 9, 3]])

    n_sample_size = 14
    n_data = np.array(
        [[9, 1, 3, 4, 6, 2, 2, 2, 2, 7, 0, 8, 1, 3],
         [7, 7, 4, 6, 3, 5, 6, 3, 2, 8, 9, 3, 1, 2],
         [8, 5, 0, 10, 10, 2, 4, 1, 3, 5, 7, 2, 0, 9]]
    )

    t = np.concatenate((p_data[1], n_data[1]))
    t.sort()
    print(t)
    print(np.percentile(t, 75))

    total_sn = p_sample_size + n_sample_size

    pw = np.zeros(p_sample_size) + (1 / total_sn)
    nw = np.zeros(n_sample_size) + (1 / total_sn)

    q_map = get_quartile(p_data, n_data, feature_size)

    print('positive data:\n', p_data)
    print('negative data:\n', n_data)

    for i, j in combinations(range(feature_size), 2):
        print(i, j)
        vote_map, err = vote(
            p_data[i], p_data[j],
            n_data[i], n_data[j],
            pw, nw,
            q_map[i], q_map[j])


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    main()
