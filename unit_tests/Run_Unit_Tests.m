clc
clear
close all

import matlab.unittest.TestSuite

addpath('../src')
addpath('../utils')


run(TestSuite.fromFile('ForwardProjectTests.m'))