from __future__ import print_function

import numpy
import pandas

from arimo.df.spark import ADF


adf = ADF.create(
    data=pandas.DataFrame(
        data=dict(
            x=[numpy.nan, 0., -1., 1.]))) \
    ('IF(x < 0, NULL, x) AS x')


adf.show(
    'x',
    'x <= 0',
    'IF(x <= 0, TRUE, FALSE)',
    'x > 0',
    'IF(x > 0, TRUE, FALSE)')

print('x <= 0:')
adf.filter('x <= 0').show()

print('x > 0:')
adf.filter('x > 0').show()


adf.show(
    'x',
    'x < 1',
    'IF(x < 1, TRUE, FALSE)',
    'x >= 1',
    'IF(x >= 1, TRUE, FALSE)')

print('x < 1:')
adf.filter('x < 1').show()

print('x >= 1:')
adf.filter('x >= 1').show()


print('x between 0 and 1:')
adf.filter('x BETWEEN 0 AND 1').show()
