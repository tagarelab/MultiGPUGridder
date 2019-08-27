%MultiGPUGridder_Matlab_Class Example MATLAB class wrapper to an underlying C++ class
classdef MultiGPUGridder_Matlab_Class < handle
    properties (SetAccess = public, Hidden = false)
        objectHandle; % Handle to the underlying C++ class instance

        interpFactor = 2;
        kernelHWidth = 2;        
        extraPadding = 3;
        
        coordAxes;
        NumAxes;        
        VolumeSize;
        CASVolumeSize;
        CASVolume;
        CASImages;
        Volume;
        ImageSize;
        Images;
        GPUs = int32([0, 1, 2, 3]);
        
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
            [varargout{1:nargout}] = mexSetVariables('SetCoordAxes', this.objectHandle, this.coordAxes, this.NumAxes);
            [varargout{1:nargout}] = mexSetVariables('SetVolumeSize', this.objectHandle, this.VolumeSize);
            [varargout{1:nargout}] = mexSetVariables('SetVolume', this.objectHandle, this.Volume);
            [varargout{1:nargout}] = mexSetVariables('SetCASVolume', this.objectHandle, this.CASVolume);              
            [varargout{1:nargout}] = mexSetVariables('SetCASImages', this.objectHandle, this.CASImages, length(this.CASImages(:)));  
            [varargout{1:nargout}] = mexSetVariables('SetImageSize', this.objectHandle, this.ImageSize);
            [varargout{1:nargout}] = mexSetVariables('SetImages', this.objectHandle, this.Images);
            [varargout{1:nargout}] = mexSetVariables('SetGPUs', this.objectHandle, this.GPUs, length(this.GPUs));
        end 
        %% GetVariables - Get the variables of the C++ class instance 
        function varargout = Get(this, variableName)                 
            switch variableName
                case 'Volume'
                    [varargout{1:nargout}] = mexGetVariables('Volume', this.objectHandle);
                case 'CASVolume'
                    [varargout{1:nargout}] = mexGetVariables('CASVolume', this.objectHandle);
%                 case 'Images'
%                     [varargout{1:nargout}] = mexGetVariables('GetImages', this.objectHandle);
                case 'CASImages'
                    [varargout{1:nargout}] = mexGetVariables('CASImages', this.objectHandle);
                otherwise
                    disp('Failed to locate variable')
            end                       
            
        end 
        %% ForwardProject - Run the forward projection function
        function ForwardProject(this, varargin)
            mexForwardProject(this.objectHandle);
        end      
    end
end