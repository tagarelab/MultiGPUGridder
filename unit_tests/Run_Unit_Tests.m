clc
clear
close all

import matlab.unittest.TestSuite

addpath('../src')
addpath('../utils')

reset(gpuDevice())

run(TestSuite.fromFile('ForwardProjectTests.m'))
run(TestSuite.fromFile('BackProjectTests.m'))

reset(gpuDevice())