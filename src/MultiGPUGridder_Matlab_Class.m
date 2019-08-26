%MultiGPUGridder_Matlab_Class Example MATLAB class wrapper to an underlying C++ class
classdef MultiGPUGridder_Matlab_Class < handle
    properties (SetAccess = public, Hidden = false)
        objectHandle; % Handle to the underlying C++ class instance

        interpFactor = 2;
        kernelHWidth = 2;        
        extraPadding = 3;
        
        NumAxes;        
        VolumeSize;
        Volume;
        ImageSize;
        Images;
        
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = MultiGPUGridder_Matlab_Class(varargin)
            this.objectHandle = mexCreateGridder(varargin{:});
        end        
        %% Deconstructor - Delete the C++ class instance 
        function Delete(this)
            mexDeleteGridder(this.objectHandle);
        end  
        %% SetVariables - Set the variables of the C++ class instance 
        function Set(this)
            [varargout{1:nargout}] = mexSetVariables('SetVolumeSize', this.objectHandle, this.VolumeSize);
            [varargout{1:nargout}] = mexSetVariables('SetVolume', this.objectHandle, this.Volume);
            [varargout{1:nargout}] = mexSetVariables('SetImageSize', this.objectHandle, this.ImageSize);
            [varargout{1:nargout}] = mexSetVariables('SetImages', this.objectHandle, this.Images);
        end 
        %% GetVariables - Get the variables of the C++ class instance 
        function varargout = Get(this, variableName)                 
            switch variableName
                case 'Volume'
                    [varargout{1:nargout}] = mexGetVariables('GetVolume', this.objectHandle);
                case 'Images'
                    [varargout{1:nargout}] = mexGetVariables('GetImages', this.objectHandle);
            end
        end 
        %% ForwardProject - Run the forward projection function
        function ForwardProject(varargin)
            mexForwardProject(varargin{:});
        end      
    end
end