#!/usr/bin/python

import sys

cnt = 0
s = 0.0
for line in sys.stdin:
    f_line = float(line)
    print('%.6f' % f_line)
    s += f_line
    cnt +=1

print('avr: %.6f' % (s / cnt))
