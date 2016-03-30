import logging
import numpy as np


log = logging.getLogger(__name__)


def gen_data(size=(3, 10)):
    return np.random.randint(0, 10, size)


def get_quartile(p_data: np.array, n_data: np.array, feature_size):
    ret = np.zeros((feature_size, 3))
    serise_size = len(p_data[0]) + len(n_data[0])
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

        print('feature {} series:'.format(f_n), series)

        ret[f_n] = tuple(series[int(i)] if i.is_integer()
                         else (series[int(i)] + series[int(i)+1]) / 2
                         for i in q_idx)

    print('quartile matrix:\n', ret)
    return ret


def vote(p_series, n_series, p_weight, n_weight, q_series):
    '''
    :param p_series: 1 dimension array of positive data
    :param n_series: 1 dimension array of negative data
    :param p_weight: 1 dimension array of positive weight
    :param n_weight: 1 dimension array of negative weight
    :param q_series: 1 dimension array of quartile
    '''
    ret_map = np.zeros((4, 4))
    a


def main():
    feature_size = 3

    p_sample_size = 10
    p_data = gen_data((feature_size, p_sample_size))

    n_sample_size = 13
    n_data = gen_data((feature_size, n_sample_size))

    print('positive data:\n', p_data)
    print('negative data:\n', n_data)

    q_map = get_quartile(p_data, n_data, feature_size)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    main()
