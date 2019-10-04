clc
clear
close all

import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
import matlab.unittest.plugins.TestRunProgressPlugin

addpath(genpath('../Tests'))

% Create the Matlab test suite object
runner = TestRunner.withNoPlugins;
p = TestRunProgressPlugin.withVerbosity(3);
runner.addPlugin(p);

% Run the test suite on the CUDA filters
FilterResults = runner.run(TestSuite.fromFile('Filter_Unit_Tests.m', 'ParameterProperty', 'GPU_Device'));

% Run the test suite on the forward projection
ForwardProjectionResults = runner.run(TestSuite.fromFile('ForwardProjectTests.m', 'ParameterProperty', 'GPU_Device'));

% Run the test suite on the back projection
BackProjectionResults = runner.run(TestSuite.fromFile('BackProjectTests.m', 'ParameterProperty', 'GPU_Device'));

% Display the results of the CUDA filters as a table
table(FilterResults)

% Display the results of the forward projection as a table
table(ForwardProjectionResults)

% Display the results of the back projection as a table
table(BackProjectionResults)


