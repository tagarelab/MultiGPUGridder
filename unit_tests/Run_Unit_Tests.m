clc
clear
close all

import matlab.unittest.TestSuite

addpath('../src')
addpath('../utils')

reset(gpuDevice())

run(TestSuite.fromFile('ForwardProjectTests.m'))


reset(gpuDevice())