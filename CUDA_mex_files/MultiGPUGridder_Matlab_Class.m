%MultiGPUGridder_Matlab_Class Example MATLAB class wrapper to an underlying C++ class
classdef MultiGPUGridder_Matlab_Class < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = MultiGPUGridder_Matlab_Class(varargin)
            this.objectHandle = mexFunctionWrapper('new', varargin{:});
        end        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            mexFunctionWrapper('delete', this.objectHandle);
        end
        %% SetAxes - Set coordinate axes
        function varargout = SetAxes(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetAxes', this.objectHandle, varargin{:});
        end
        %% SetVolume - Set GPU volume
        function varargout = SetVolume(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetVolume', this.objectHandle, varargin{:});
        end
        %% GetVolume - Get the summed volume from all of the GPUs
        function varargout = GetVolume(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('GetVolume', this.objectHandle, varargin{:});
        end
        %% SetImages - Set CAS Imgs
        function varargout = SetImages(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetImages', this.objectHandle, varargin{:});
        end
        %% ResetVolume - Reset GPU volume
        function varargout = ResetVolume(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('ResetVolume', this.objectHandle, varargin{:});
        end
        %% SetImgSize - Set output image size
        function varargout = SetImgSize(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetImgSize', this.objectHandle, varargin{:});
        end
        %% SetMaskRadius - Set mask radius
        function varargout = SetMaskRadius(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetMaskRadius', this.objectHandle, varargin{:});
        end
        %% SetNumberGPUs - Set number of GPUs to use with CUDA kernel
        function varargout = SetNumberGPUs(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetNumberGPUs', this.objectHandle, varargin{:});
        end
        %% SetNumberStreams - Set number of GPUs to use with CUDA kernel
        function varargout = SetNumberStreams(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetNumberStreams', this.objectHandle, varargin{:});
        end
        %% SetNumberBatches - Set number of batches to use with CUDA kernel
        function varargout = SetNumberBatches(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetNumberBatches', this.objectHandle, varargin{:});
        end
        %% mem_alloc - Allocate memory
        function varargout = mem_alloc(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('mem_alloc', this.objectHandle, varargin{:});
        end
        %% pin_mem - Pin CPU array to memory
        function varargout = pin_mem(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('pin_mem', this.objectHandle, varargin{:});
        end
        %% disp_mem - Display memory
        function varargout = disp_mem(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('disp_mem', this.objectHandle, varargin{:});
        end
        %% mem_Copy - Copy memory
        function varargout = mem_Copy(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('mem_Copy', this.objectHandle, varargin{:});
        end
        %% mem_Return - Return memory array
        function varargout = mem_Return(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('mem_Return', this.objectHandle, varargin{:});
        end
        %% GetImgs - Return CASImgs array
        function varargout = GetImgs(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('GetImgs', this.objectHandle, varargin{:});
        end      
        %% mem_Free - Free CPU memory
        function varargout = mem_Free(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('mem_Free', this.objectHandle, varargin{:});
        end
        %% CUDA_alloc - Allocate GPU memory 
        function varargout = CUDA_alloc(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('CUDA_alloc', this.objectHandle, varargin{:});
        end
        %% CUDA_Free - Free GPU memory
        function varargout = CUDA_Free(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('CUDA_Free', this.objectHandle, varargin{:});
        end
        %% CUDA_disp_mem - Copy memory
        function varargout = CUDA_disp_mem(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('CUDA_disp_mem', this.objectHandle, varargin{:});
        end
        %% CUDA_Copy - Copy Matlab array to GPU array
        function varargout = CUDA_Copy(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('CUDA_Copy', this.objectHandle, varargin{:});
        end
        %% CUDA_Return - Return CUDA array back to Matlab
        function varargout = CUDA_Return(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('CUDA_Return', this.objectHandle, varargin{:});
        end
        %% Projection_Initilize - Initialize the projection kernels by allocating the rest of needed memory
        function varargout = Projection_Initilize(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('Projection_Initilize', this.objectHandle, varargin{:});
        end
        %% Forward_Project - Run the forward projection kernel
        function varargout = Forward_Project(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('Forward_Project', this.objectHandle, varargin{:});
        end
        %% Back_Project - Run the back projection kernel
        function varargout = Back_Project(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('Back_Project', this.objectHandle, varargin{:});
        end
    end
end