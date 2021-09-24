import numpy as np
from ail.common.math import pure_discount_cumsum, discount_cumsum


x1 = [1, 1, 1, 1, 1, 1, 1, 1]
discount1 = 0.99
res1 = [
    7.72553055720799,
    6.793465209301,
    5.8519850599,
    4.90099501,
    3.9403989999999998,
    2.9701,
    1.99,
    1,
]

x2 = [1, 2, 3, 4, 5, 6, 7, 8]
discount2 = 0.99
res2 = [
    34.35730018,
    33.6942426,
    32.01438647,
    29.30746108,
    25.563092,
    20.7708,
    14.92,
    8.0,
]

x3 = [1, 2, 3, 4, 5, 6, 7, 8]
discount3 = 0.0
res3 = [1, 2, 3, 4, 5, 6, 7, 8]

x4 = [1, 2, 3, 4, 5, 6, 7, 8]
discount4 = 0.9
res4 = [22.5159022, 23.906558, 24.34062, 23.7118, 21.902, 18.78, 14.2, 8.0]

x5 = [8, 7, 6, 5, 4, 3, 2, 1]
discount5 = 0.99
res5 = [35.17247484, 27.44694428, 20.65347907, 14.80149401, 9.900499, 5.9601, 2.99, 1.0]


x6 = [1, 1, 1, 1, 1, 1, 1, 1]
discount6 = 0.99
res6 = np.array(
    [
        7.72553055720799,
        6.793465209301,
        5.8519850599,
        4.90099501,
        3.9403989999999998,
        2.9701,
        1.99,
        1,
    ]
)

x7 = [1, 1, 1, 1, 1, 1, 1, 1]
discount7 = 0.99
res7 = np.array(
    [
        [
            7.72553055720799,
            6.793465209301,
            5.8519850599,
            4.90099501,
            3.9403989999999998,
            2.9701,
            1.99,
            1,
        ]
    ]
)


def test_pure_discount_cumsum_calculation():

    assert np.allclose(pure_discount_cumsum(x1, discount1), res1)
    assert np.allclose(pure_discount_cumsum(x2, discount2), res2)


def test_pure_discount_cumsum_discount():
    assert np.allclose(pure_discount_cumsum(x3, discount3), res3)
    assert np.allclose(pure_discount_cumsum(x4, discount4), res4)


def test_pure_discount_cumsum_reverse():
    assert np.allclose(pure_discount_cumsum(x5, discount5), res5)


def test_pure_discount_cumsum_output_type():
    assert isinstance(pure_discount_cumsum(x1, discount1), list)


def teste_pure_discount_cumsum_vector():
    assert np.allclose(pure_discount_cumsum(x6, discount6), res6)


################################################################
def test_discount_cumsum_calculation():

    assert np.allclose(discount_cumsum(x1, discount1), res1)
    assert np.allclose(discount_cumsum(x2, discount2), res2)


def test_discount_cumsum_discount():
    assert np.allclose(discount_cumsum(x3, discount3), res3)
    assert np.allclose(discount_cumsum(x4, discount4), res4)


def test_discount_cumsum_reverse():
    assert np.allclose(discount_cumsum(x5, discount5), res5)


def test_discount_cumsum_output_type():
    assert isinstance(discount_cumsum(x1, discount1), np.ndarray)


def teste_discount_cumsum_array():
    assert np.allclose(discount_cumsum(x7, discount7), res7)
