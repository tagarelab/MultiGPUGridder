clc
clear
close all

import matlab.unittest.TestSuite

dir = ['/home/brent/Documents/MATLAB/simple_gpu_gridder_Obj/'];

run(TestSuite.fromFile([dir 'unit_tests/ForwardProjectTests.m']))