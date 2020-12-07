%MultiGPUGridder_Matlab_Class Example MATLAB class wrapper to an underlying C++ class
classdef MultiGPUGridder_Matlab_Class < handle
    properties (SetAccess = public, Hidden = false)
        
        objectHandle; % Handle to the underlying C++ class instance
        
        % Flag to run the forward / inverse FFT on the device (i.e. the GPU)
        RunFFTOnGPU = true;        
        
        % Flag for status output to the console
        verbose = false;
        
        % Int 32 type variables        
        VolumeSize;        
        NumAxes;
        GPUs = int32([]);
        MaxAxesToAllocate;
        nStreamsFP;% For the forward projection
        nStreamsBP; % For the back projection
        
        % Single type variables        
        interpFactor;
        kerHWidth = 2;        
        kerTblSize = 501;
        extraPadding = 3;    
        KBTable;        
        coordAxes;      
        CASVolume;
        CASImages;
        Volume;
        Images;  
        MaskRadius;
        
    end
    
    methods
        %% Constructor - Create a new C++ class instance 
        function this = MultiGPUGridder_Matlab_Class(varargin)  
           
            p = inputParser;
            addOptional(p, 'VolumeSize', 0, @isnumeric);
            addOptional(p, 'NumAxes', 0, @isnumeric);
            addOptional(p, 'interpFactor', 2, @isnumeric);
            addOptional(p, 'GPUs', [], @isnumeric);
            addOptional(p, 'RunFFTOnGPU', 1, @isnumeric); % Offset of the ProjNdx index (if we want to project a subset of the images)
            addOptional(p, 'verbose', 0, @isnumeric);
            addOptional(p, 'nStreamsFP', 10, @isnumeric);
            addOptional(p, 'nStreamsBP', 4, @isnumeric);
            addOptional(p, 'MaxAxesToAllocate', 1000, @isnumeric);
            
            parse(p, varargin{:});
            this.VolumeSize = int32(p.Results.VolumeSize);
            this.NumAxes = int32(p.Results.NumAxes);
            this.interpFactor = single(p.Results.interpFactor);
            this.GPUs = p.Results.GPUs;
            this.RunFFTOnGPU = int32(p.Results.RunFFTOnGPU);
            this.verbose = p.Results.verbose;
            this.nStreamsFP = p.Results.nStreamsFP;
            this.nStreamsBP = p.Results.nStreamsBP;
            this.MaxAxesToAllocate = p.Results.MaxAxesToAllocate;

            if (this.VolumeSize == 0)
                error("VolumeSize is a required input.")
            elseif (this.NumAxes == 0)
                error("NumAxes is a required input.")
            end
            
            
            % Adjust interpFactor to scale to the closest factor of 2^ (i.e. 64, 128, 256, etc)
            % For example, this is particularly needed if the image size is 208 so the GPU can allocate memory correctly
            if(this.VolumeSize < 64)
                this.interpFactor = 128 / single(this.VolumeSize);
                warning("interpFactor adjusted from " + num2str(varargin{3}) + " to " + num2str(this.interpFactor)+  " so that the volume size will be on the order of 2^n.")
            elseif (this.VolumeSize > 64 && this.VolumeSize < 128)
                this.interpFactor = 256 / single(this.VolumeSize);
                warning("interpFactor adjusted from " + num2str(varargin{3}) + " to " + num2str(this.interpFactor)+  " so that the volume size will be on the order of 2^n.")
            elseif (this.VolumeSize > 128 && this.VolumeSize < 256)
                this.interpFactor = single(512 / double(this.VolumeSize));
                warning("interpFactor adjusted from " + num2str(varargin{3}) + " to " + num2str(this.interpFactor)+  " so that the volume size will be on the order of 2^n.")
            elseif (this.VolumeSize > 512 && this.VolumeSize < 512)
                this.interpFactor = 1024 / single(this.VolumeSize);
                warning("interpFactor adjusted from " + num2str(varargin{3}) + " to " + num2str(this.interpFactor)+  " so that the volume size will be on the order of 2^n.")
            end          
                        
            % Create the Volume array
            this.Volume = zeros(repmat(this.VolumeSize, 1, 3), 'single');
                        
            % If the GPUs to use was not given use all the available GPUs
            if isempty(this.GPUs)
                this.GPUs = int32([1:gpuDeviceCount]) - 1; % CUDA GPU device numbers need to start at zero
            end            
            
            % Reset all the GPU devices
            for i = 1:length(this.GPUs)
                if this.verbose == true
                    disp("Resetting GPU number "  + num2str(double(this.GPUs(i) + 1)))
                end
                reset(gpuDevice(double(this.GPUs(i) + 1)));
            end
                
            this.MaskRadius = (single(this.VolumeSize) * this.interpFactor) / 2 - 1;            

            gridder_Varargin = cell(8,1);
            gridder_Varargin{1} = int32(this.VolumeSize);
            gridder_Varargin{2} = int32(this.NumAxes);
            gridder_Varargin{3} = single(this.interpFactor);
            gridder_Varargin{4} = int32(this.extraPadding);            
            gridder_Varargin{5} = int32(length(this.GPUs));
            gridder_Varargin{6} = int32(this.GPUs);
            gridder_Varargin{7} = int32(this.RunFFTOnGPU);
            gridder_Varargin{8} = this.verbose;            
            
            % Check that the GPU index vector is valid given the number of available GPUs on the computer
            if length(this.GPUs) > gpuDeviceCount
                error("Requested more GPUs than are available. Please change the GPUs vector in MultiGPUGridder class.")
            elseif max(this.GPUs(:)) - 1 > gpuDeviceCount
                error("GPU index must range from 0 to the number of available GPUs. Please change the GPUs vector in MultiGPUGridder class.")
            end

            % Create the gridder instance
            this.objectHandle = mexCreateGridder(gridder_Varargin{1:8});           
                       
            % Initialize the output projection images array
            ImageSize = [this.VolumeSize, this.VolumeSize, this.NumAxes];
            this.Images = zeros(ImageSize(1), ImageSize(2), ImageSize(3), 'single');          

            % Load the Kaiser Bessel lookup table
            this.KBTable = single(getKernelFiltTable(this.kerHWidth, this.kerTblSize));            
        
            % If we're running the FFTs on the CPU, allocate the CPU memory to return the arrays to
            if (this.RunFFTOnGPU == false) || this.verbose == true
 
                % Create the CASImages array
                CASImagesSize = size(this.Volume, 1) * this.interpFactor; 
                this.CASImages = zeros([CASImagesSize, CASImagesSize, this.NumAxes], 'single');  
                
            end            
            
            % Create the CASVolume array
            this.CASVolume = zeros(repmat(size(this.Volume, 1) * this.interpFactor + this.extraPadding * 2, 1, 3), 'single');                  

            % Create the coordinate axes array
            this.coordAxes = zeros(9, this.NumAxes, 'single');
            
            % Set the matlab array pointers to the C++ class
            % NOTE: Ensure that the pointers are not overwritten anywhere in the Matlab class
            % it is possible to call Set() again to re-copy the pointers, but this is not as reliable
            this.Set();
        end        
        %% Deconstructor - Delete the C++ class instance 
        function delete(this)
            mexDeleteGridder(this.objectHandle);
        end  
        %% SetVariables - Set the variables of the C++ class instance 
        function Set(this)
            
            if (isempty(this.coordAxes) || isempty(this.Volume) || isempty(this.Images) ...
                || isempty(this.GPUs) || isempty(this.KBTable) || isempty(this.nStreamsFP) || isempty(this.nStreamsBP))
                error("Error: Required array is missing in Set() function.")             
            end              
            
            % Check the sizes of the input variables
            if unique(size(this.Volume)) ~= size(this.Volume,1)
                error("Volume must be a square")
            elseif unique(size(this.Images(:,:,1))) ~= size(this.Images(:,:,1),1)
                error("Images must be a square")
            elseif size(this.Images(:,:,1),1) ~= size(this.Volume,1)
                error("Images must be the same size as Volume")
            elseif size(this.Images(:,:,1),1) <= this.MaskRadius
%                 error("Images must be larger than the MaskRadius")
            elseif size(this.Volume,1) <= this.MaskRadius
                error("Volume must be larger than the MaskRadius")
            end
            
            this.CheckParameters();
            
            [varargout{1:nargout}] = mexSetVariables('SetCoordAxes', this.objectHandle, single(this.coordAxes(:)), int32(size(this.coordAxes(:))));
            [varargout{1:nargout}] = mexSetVariables('SetVolume', this.objectHandle, single(this.Volume), int32(size(this.Volume)));                                 
            [varargout{1:nargout}] = mexSetVariables('SetImages', this.objectHandle, single(this.Images), int32(size(this.Images)));            
            [varargout{1:nargout}] = mexSetVariables('SetGPUs', this.objectHandle, int32(this.GPUs), int32(length(this.GPUs)));
            [varargout{1:nargout}] = mexSetVariables('SetKBTable', this.objectHandle, single(this.KBTable), int32(size(this.KBTable)));           
            [varargout{1:nargout}] = mexSetVariables('SetNumberStreamsFP', this.objectHandle, int32(this.nStreamsFP));             
            [varargout{1:nargout}] = mexSetVariables('SetNumberStreamsBP', this.objectHandle, int32(this.nStreamsBP));
            [varargout{1:nargout}] = mexSetVariables('SetMaskRadius', this.objectHandle, single(this.MaskRadius));
            [varargout{1:nargout}] = mexSetVariables('SetNumAxes', this.objectHandle, int32(size(this.coordAxes,2))); 
            [varargout{1:nargout}] = mexSetVariables('SetCASVolume', this.objectHandle, single(this.CASVolume), int32(size(this.CASVolume)));

            if ~isempty(this.CASImages)
                [varargout{1:nargout}] = mexSetVariables('SetCASImages', this.objectHandle, single(this.CASImages), int32(size(this.CASImages)));
            end
            if ~isempty(this.MaxAxesToAllocate)
                [varargout{1:nargout}] = mexSetVariables('SetMaxAxesToAllocate', this.objectHandle, int32(this.MaxAxesToAllocate));
            end            
            
        end 

        %% ForwardProject - Run the forward projection function
        function ProjectionImages = forwardProject(this, varargin)

            if ~isempty(varargin)
                
                if length(varargin) == 1
                    % A new set of coordinate axes was passed
                    % (:,1:size(single(varargin{1}),2))
                    numAxes = size(single(varargin{1}),2);
                    this.coordAxes(:,1:numAxes) = single(varargin{1});        
                    mexSetVariables('SetNumAxes', this.objectHandle, int32(numAxes)); 
                       
                elseif length(varargin) == 3
                    % Varargin 1: Mean volume
                    % Varargin 2: Coordinate axes
                    % Varargin 3: Mask radius
                    this.Volume = single(varargin{1});
                    numAxes = size(single(varargin{2}),2);
                    this.coordAxes(:,1:numAxes) = single(varargin{2});        
                    mexSetVariables('SetNumAxes', this.objectHandle, int32(numAxes)); 
                    
                    this.MaskRadius = single(varargin{3});                    
                else
                    disp('forwardProject(): Unknown input')
                    return
                end
            end
            
            [origBox,interpBox,CASBox]=getSizes(single(this.VolumeSize), this.interpFactor,3);
            temp = CASFromVol_Gridder(this.Volume, this.kerHWidth, this.interpFactor, this.extraPadding);
            this.CASVolume(:) = temp(:); % Keep the original memory pointer
            
            if size(this.coordAxes,2) < this.nStreamsFP
                error("The number of projection directions must be >= the number of CUDA streams.")
            end          

            mexMultiGPUForwardProject(this.objectHandle);            
            
            % Run the inverse FFT on the CAS images
            if (this.RunFFTOnGPU == false)
                this.Images(:,:,1:numAxes) = imgsFromCASImgs(this.CASImages(:,:,1:numAxes), interpBox, []); 
            end
            
            % Consider if we forward project less number of images then we first allocated for
            ProjectionImages = this.Images(:,:,1:size(this.coordAxes,2));            
            
        end         
        
          function backProject(this, varargin)

            if ~isempty(varargin) > 0
                
                % A new set of images to back project was passed
                this.Images(:,:,1:size(varargin{1},3)) = single(varargin{1}); 
                
                % A new set of coordinate axes to use with the back projection was passed
                numAxes = size(single(varargin{2}),2);
                this.coordAxes(:,1:numAxes) = single(varargin{2});
                mexSetVariables('SetNumAxes', this.objectHandle, int32(numAxes));
                
                if (this.RunFFTOnGPU == false)
                    
                    % Run the forward FFT and convert the images to CAS images
                    [~,interpBox,~]=getSizes(single(this.VolumeSize), this.interpFactor,3);
                    this.CASImages = CASImgsFromImgs(this.Images, interpBox, []);
                    
                end
                
            end
            
            mexMultiGPUBackProject(this.objectHandle);   
            
        end       
    
        %% setVolume - Set the volume
        function setVolume(this, varargin)
            % The new volume will be copied to the GPUs during this.Set()
           this.Volume = single(varargin{1}); 
        end
        %% resetVolume - Reset the volume
        function resetVolume(this)
            
            % Multiply the volume by zero to reset. The resetted volume will be copied to the GPUs during this.Set()
           this.Volume(:) = 0;            
           this.CASVolume(:) = 0;  % Keep the original memory pointer
        end
        %% reconstructVol - Reconstruct the volume by dividing by the plane density
        function Volume = reconstructVol(this, varargin)   
            
            if (this.RunFFTOnGPU == true)
                error("reconstructVol() is currently only supported if RunFFTOnGPU is set to false.")
            end

            this.Images = single(varargin{1});
            this.coordAxes = single(varargin{2});

            % Run a normal back projection first
            this.resetVolume()
            this.backProject();      
            this.getVol();
            tmpCASVolume = this.CASVolume;

            % Now, set the CASImages to all ones and back project a second time to get the plane density array
            this.CASImages(:) = 1;
            this.resetVolume()            
            this.backProject();     
            this.getVol();         
            PlaneDensity = this.CASVolume;
  
            % Convert the CASVolume to Volume
            % Normalize by the plane density
            tmpCASVolume = tmpCASVolume ./(PlaneDensity+1e-6);
            this.Volume=volFromCAS_Gridder(tmpCASVolume,single(this.VolumeSize),this.kerHWidth, this.interpFactor);

            Volume = single(this.Volume);         
        end   
        %% getVol - Convert the CAS Volume to Volume
        function Volume = getVol(this)
        
%            this.Set(); % Run the set function in case one of the arrays has changed
%            this.Set();
%            this.Set();
           
           mexMultiGPUGetVolume(this.objectHandle);           
           
           % Convert the CASVolume to Volume
           this.Volume = volFromCAS_Gridder(this.CASVolume,single(this.VolumeSize),this.kerHWidth, this.interpFactor);

           Volume = single(this.Volume); 

        end
        %% CheckParameters - check that all the parameters and variables are valid
        function CheckParameters(this)
            
            % Each x, y, and z component of the coordinate axes needs to have a norm of one
            for i = 1:size(this.coordAxes,2)
                for j = 1:3                    
                    if norm(this.coordAxes((j-1)*3+1:j*3),2) - 1 > 0.0001 % Account for rounding errors
                        warning("Invalid coordAxes parameter: Each x, y, and z component of the coordinate axes needs to have a norm of one")
                    end
                end
            end                
            
            if this.interpFactor <= 0
                error("Invalid Interp Factor parameter: must be a non-negative value")
            end            
            
        end
        
    end
end