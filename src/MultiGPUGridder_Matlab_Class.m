%MultiGPUGridder_Matlab_Class Example MATLAB class wrapper to an underlying C++ class
classdef MultiGPUGridder_Matlab_Class < handle
    properties (SetAccess = public, Hidden = false)
        objectHandle; % Handle to the underlying C++ class instance

        interpFactor = 2;
        kernelHWidth = 2;        
        extraPadding = 3;
        
        KBTable;
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
        MaxGPUAxesToAllocate = 4000;
        
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
            [varargout{1:nargout}] = mexSetVariables('SetCoordAxes', this.objectHandle, single(this.coordAxes), int32(size(this.coordAxes)));
            [varargout{1:nargout}] = mexSetVariables('SetVolume', this.objectHandle, single(this.Volume), int32(size(this.Volume)));
            [varargout{1:nargout}] = mexSetVariables('SetCASVolume', this.objectHandle, single(this.CASVolume), int32(size(this.CASVolume)));              
            [varargout{1:nargout}] = mexSetVariables('SetCASImages', this.objectHandle, single(this.CASImages), int32(size(this.CASImages)));
            [varargout{1:nargout}] = mexSetVariables('SetImages', this.objectHandle, single(this.Images), int32(size(this.Images)));
            [varargout{1:nargout}] = mexSetVariables('SetGPUs', this.objectHandle, int32(this.GPUs), int32(length(this.GPUs)));
            [varargout{1:nargout}] = mexSetVariables('SetKBTable', this.objectHandle, single(this.KBTable), int32(size(this.KBTable)));
            [varargout{1:nargout}] = mexSetVariables('SetMaxGPUAxesToAllocate', this.objectHandle, int32(this.MaxGPUAxesToAllocate)); 
        end 
        %% GetVariables - Get the variables of the C++ class instance 
        function varargout = Get(this, variableName)                 
            switch variableName
                case 'Volume'
                    [varargout{1:nargout}] = mexGetVariables('Volume', this.objectHandle);
                case 'CASVolume'
                    [varargout{1:nargout}] = mexGetVariables('CASVolume', this.objectHandle);
                case 'Images'
                    [varargout{1:nargout}] = mexGetVariables('Images', this.objectHandle);
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