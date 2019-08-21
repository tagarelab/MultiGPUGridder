%MultiGPUGridder_Matlab_Class Example MATLAB class wrapper to an underlying C++ class
classdef MultiGPUGridder_Matlab_Class < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
        interpBox;
        fftinfo;
        interpFactor = 2;
        origBox;
        CASBox;
        kernelHWidth = 2;
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
        %% setCoordAxes - Set coordinate axes
        function varargout = setCoordAxes(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('setCoordAxes', this.objectHandle, varargin{:});
        end
        %% SetVolume - Set GPU volume
        function varargout = setVolume(this, varargin)
            
            [CASVol, CASBox, origBox, interpBox, fftinfo] = Vol_Preprocessing(varargin{1}, this.interpFactor);
            this.origBox = origBox;
            this.interpBox = interpBox;
            this.fftinfo = fftinfo;
            this.CASBox = CASBox;
            
            [varargout{1:nargout}] = mexFunctionWrapper('SetVolume', this.objectHandle, CASVol);
        end
        %% GetVolume - Get the summed volume from all of the GPUs
        function varargout = GetVolume(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('GetVolume', this.objectHandle, varargin{:});
            
%              varargout{1:nargout}=volFromCAS(varargout{1:nargout},this.CASBox, this.interpBox, this.origBox,this.kernelHWidth);
       
        end
        %% SetImages - Set CAS Imgs
        function varargout = SetImages(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('SetImages', this.objectHandle, varargin{:});
        end
        %% resetVolume - Reset GPU volume
        function varargout = resetVolume(this, varargin)
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
        function varargout = forwardProject(this, varargin)
            coordAxes = varargin{1};
            coordAxes = coordAxes(:);
            disp("Forward Project...")
            imgs = mexFunctionWrapper('forwardProject', this.objectHandle, coordAxes);
            disp("imgsFromCASImgs...")
            varargout{1}=imgsFromCASImgs(imgs, this.interpBox, this.fftinfo); 
        end
        %% Forward_Project get the CAS images
        function varargout = forwardProjectCAS(this, varargin)
            coordAxes = varargin{1};
            coordAxes = coordAxes(:);
            disp("Forward Project...")
            varargout{1} = mexFunctionWrapper('forwardProject', this.objectHandle, coordAxes);
        end
        %% Set the kaiser bessel vector
        function varargout = SetKerBesselVector(this, varargin)
            [varargout{1:nargout}]  = mexFunctionWrapper('SetKerBesselVector', this.objectHandle,  varargin{:});
        end        
        
        %% Back_Project - Run the back projection kernel
        function varargout = backProject(this, varargin)
            
            imgs = varargin{1};
            coordAxes = varargin{2};
            
            % Need to convert the images to CAS imgs
            CAS_projection_imgs = CASImgsFromImgs(imgs, this.interpBox, this.fftinfo);
            
            this.SetImages(single(CAS_projection_imgs));
            this.setCoordAxes(single(coordAxes(:)));
            
            this.Print()
            
            [varargout{1:nargout}] = mexFunctionWrapper('Back_Project', this.objectHandle);
        end
        %% Print- Print the current parameters to the console
        function varargout = Print(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('Print', this.objectHandle, varargin{:});
        end
        %% Reconstruct Volume
        function volReconstructed = reconstructVol(this, coordAxes)
            
            volCAS = this.GetVolume();
%    
%             % Get the density of inserted planes by backprojecting CASimages of values equal to one
%             disp("Get Plane Density()...")
%             interpImgs=ones([this.interpBox.size this.interpBox.size size(coordAxes(:),1)/9],'single');
%             
% %             CAS_interpImgs = [];
% %             CAS_interpImgs = CASImgsFromImgs(interpImgs,this.interpBox, this.fftinfo);
%             this.resetVolume();
%             this.SetImages(single(interpImgs))
% %             this.backProject(interpImgs, coordAxes(:))
%             this.setCoordAxes(single(coordAxes(:)));
%             [varargout{1:nargout}] = mexFunctionWrapper('Back_Project', this.objectHandle);
%             
%             volWt = this.GetVolume();

%             imagesc(volWt(:,:,10))
            
            % Normalize the back projection result with the plane density
            % Divide the previous volume with the plane density volume
%             volCAS=volCAS./(volWt+1e-6);
%             volCAS = volWt; % TEST

            % Reconstruct the volume from CASVol
            disp("volFromCAS()...")
            volReconstructed=volFromCAS(volCAS,this.CASBox, this.interpBox, this.origBox,this.kernelHWidth);
       
            
            
        end
        %% CropVolume - Crop a volume
        function varargout = CropVolume(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('CropVolume', this.objectHandle, varargin{:});
        end
        
        %% PadVolume - Zero pad a volume
        function varargout = PadVolume(this, varargin)
            [varargout{1:nargout}] = mexFunctionWrapper('PadVolume', this.objectHandle, varargin{:});
        end
    end
end