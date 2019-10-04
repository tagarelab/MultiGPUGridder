clc
clear
close all

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.TestRunProgressPlugin

addpath(genpath('../Tests'))

runner = TestRunner.withNoPlugins;
p = TestRunProgressPlugin.withVerbosity(4);
runner.addPlugin(p);

% Run the test suite
results = runner.run(TestSuite.fromFile('FilterTests.m', 'ParameterProperty', 'GPU_Device'));

% Display the results as a table
table(results)