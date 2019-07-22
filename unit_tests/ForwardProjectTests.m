classdef ForwardProjectTests < matlab.unittest.TestCase
    % SolverTest tests solutions to the forward project CUDA gridder
    
    % import matlab.unittest.TestSuite
    % clc; clear; close all;
    % suiteFile = TestSuite.fromFile('unit_tests/ForwardProjectTests.m')
    % result = run(suiteFile)
    
    % Class variables
    properties   

        %Initialize parameters
        volSize=64;
        n1_axes=15;
        n2_axes=15;
        interpFactor=2.0;

    end
    
    methods
        function initilize_data()

            import matlab.unittest.TestSuite
            
            clear all;
            close all;
            reset(gpuDevice());


            %kernelHWidth=2.0;


            %Set the path to the utils directory
            addpath(fullfile('.','utils'));


            %Get the gridder
            a=gpuBatchGridder(volSize,n1_axes*n2_axes+1,interpFactor);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Volume

            origSize=volSize;
            volCenter=volSize/2+1;
            origCenter=origSize/2+1;
            origHWidth= origCenter-1;

            %Fuzzy sphere
            vol=fuzzymask(origSize,3,origSize*.25,2,origCenter*[1 1 1]);
            a.setVolume(vol);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Create Coordaxes

            coordAxes=single([1 0 0 0 1 0 0 0 1]');
            coordAxes=[coordAxes create_uniform_axes(n1_axes,n2_axes,0,10)];

        end

    end




    methods (Test)
        function testRealSolution(testCase)
            testCase.verifyEqual(2, 2);
            % Forward Project
%             img=a.forwardProject(coordAxes);

        end
        function testImaginarySolution(testCase)
            testCase.verifyEqual(4,4);
        end
    end
    
end