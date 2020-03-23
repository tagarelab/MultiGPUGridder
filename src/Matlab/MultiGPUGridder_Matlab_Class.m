%MultiGPUGridder_Matlab_Class Example MATLAB class wrapper to an underlying C++ class
classdef MultiGPUGridder_Matlab_Class < handle
    properties (SetAccess = public, Hidden = false)
        
        objectHandle; % Handle to the underlying C++ class instance
        
        % Flag to run the forward / inverse FFT on the device (i.e. the GPU)
        RunFFTOnGPU = true;        
        
        % Flag for status output to the console
        verbose = true;
        
        % Int 32 type variables        
        VolumeSize;        
        NumAxes;
        GPUs = int32([0,1,2,3]);
        MaxAxesToAllocate;
        nStreamsFP = 10; % For the forward projection
        nStreamsBP = 4; % For the back projection
        
        % Single type variables        
        interpFactor;
        kerHWidth = 2;        
        kerTblSize = 501;
        extraPadding = 3;    
        KBTable;
        PlaneDensity;
        coordAxes;      
        CASVolume;
        CASImages;
        Volume;
        Images;                 
        MaskRadius;
        KBPreComp;
        
    end
    
    methods
        %% Constructor - Create a new C++ class instance 
        function this = MultiGPUGridder_Matlab_Class(varargin)  
            % Inputs are:            
            % (1) VolumeSize
            % (2) nCoordAxes
            % (3) interpFactor        
            
            % Add the paths of the compiled Mex files and the util folder relative to this file
%             mfilepath=fileparts(which('MultiGPUGridder_Matlab_Class.m'));
%             addpath(fullfile(mfilepath,'./utils'));
%             addpath(fullfile(mfilepath,'../../bin'));
 
            
            this.VolumeSize = int32(varargin{1});
            this.NumAxes = int32(varargin{2});
            this.interpFactor = single(varargin{3});                                  
            this.MaskRadius = (single(this.VolumeSize) * this.interpFactor) / 2 - 1;
            
            if (length(varargin) >= 4)
                this.RunFFTOnGPU = varargin{4};
            end

%             if (length(varargin) >= 5)
%                 if varargin{5} == 1
%                     this.GPUs = int32([0]);
%                 elseif varargin{5} == 2
%                     this.GPUs = int32([0,1]);
%                 elseif varargin{5} == 3
%                     this.GPUs = int32([0,1,2]);
%                 elseif varargin{5} == 4
%                     this.GPUs = int32([0,1,2,3]);
%                 end
%             end
%             
            
            gridder_Varargin = cell(7,1);
            gridder_Varargin{1} = int32(varargin{1});
            gridder_Varargin{2} = int32(varargin{2});
            gridder_Varargin{3} = single(varargin{3});
            gridder_Varargin{4} = int32(length(this.GPUs));
            gridder_Varargin{5} = int32(this.GPUs);
            gridder_Varargin{6} = int32(this.RunFFTOnGPU);
            gridder_Varargin{7} = this.verbose;            
            
            % Check that the GPU index vector is valid given the number of available GPUs on the computer
            if length(this.GPUs) > gpuDeviceCount
                error("Requested more GPUs than are available. Please change the GPUs vector in MultiGPUGridder class.")
            elseif max(this.GPUs(:)) - 1 > gpuDeviceCount
                error("GPU index must range from 0 to the number of available GPUs. Please change the GPUs vector in MultiGPUGridder class.")
            end

            % Create the gridder instance
            this.objectHandle = mexCreateGridder(gridder_Varargin{1:7});           
                       
            % Initialize the output projection images array
            ImageSize = [this.VolumeSize, this.VolumeSize, this.NumAxes];
            this.Images = zeros(ImageSize(1), ImageSize(2), ImageSize(3), 'single');
            
            % Load the Kaiser Bessel lookup table
            this.KBTable = single(getKernelFiltTable(this.kerHWidth, this.kerTblSize)); 

            % Create the Volume array
            this.Volume = zeros(repmat(this.VolumeSize, 1, 3), 'single');              
        
            % If we're running the FFTs on the CPU, allocate the CPU memory to return the arrays to
            if (this.RunFFTOnGPU == false)
 
                % Create the CASImages array
                CASImagesSize = size(this.Volume, 1) * this.interpFactor; 
                this.CASImages = zeros([CASImagesSize, CASImagesSize, this.NumAxes], 'single');  
                
                % Create the PlaneDensity array
                this.PlaneDensity = zeros(repmat(size(this.Volume, 1) * this.interpFactor + this.extraPadding * 2, 1, 3), 'single');
                                                            
                % Create the CASVolume array
                this.CASVolume = zeros(repmat(size(this.Volume, 1) * this.interpFactor + this.extraPadding * 2, 1, 3), 'single');                                                     
           
            end
            
            % Create the Kaiser Bessel pre-compensation array
            % After backprojection, the inverse FFT volume is divided by this array
            InterpVolSize = single(this.VolumeSize) * single(this.interpFactor);
            this.KBPreComp = zeros(repmat(size(this.Volume, 1) * this.interpFactor, 1, 3), 'single');    
           
            preComp=getPreComp(InterpVolSize,this.kerHWidth);
            preComp=preComp';
            this.KBPreComp=single(reshape(kron(preComp,kron(preComp,preComp)),...
                         InterpVolSize,InterpVolSize,InterpVolSize));                    

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
                error("Images must be larger than the MaskRadius")
            elseif size(this.Volume,1) <= this.MaskRadius
                error("Volume must be larger than the MaskRadius")
            elseif size(this.coordAxes,1) ~= 9
                error("The first dimension of coordAxes must be equal to 9")
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
            [varargout{1:nargout}] = mexSetVariables('SetKBPreCompArray', this.objectHandle, single(this.KBPreComp), int32(size(this.KBPreComp)));
          
           
            % Set the optional arrays
            if ~isempty(this.PlaneDensity)
                [varargout{1:nargout}] = mexSetVariables('SetPlaneDensity', this.objectHandle, single(this.PlaneDensity), int32(size(this.PlaneDensity)));
            end
            if ~isempty(this.CASVolume)
                [varargout{1:nargout}] = mexSetVariables('SetCASVolume', this.objectHandle, single(this.CASVolume), int32(size(this.CASVolume)));       
            end
            if ~isempty(this.CASImages)
                [varargout{1:nargout}] = mexSetVariables('SetCASImages', this.objectHandle, single(this.CASImages), int32(size(this.CASImages)));
            end
            if ~isempty(this.MaxAxesToAllocate)
                [varargout{1:nargout}] = mexSetVariables('SetMaxAxesToAllocate', this.objectHandle, int32(this.MaxAxesToAllocate));
            end            
            
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
        function ProjectionImages = forwardProject(this, varargin)

            if ~isempty(varargin)
                
                if length(varargin) == 1
                    % A new set of coordinate axes was passed
                    this.coordAxes = single(varargin{1});            
                    
                elseif length(varargin) == 3
                    % Varargin 1: Mean volume
                    % Varargin 2: Coordinate axes
                    % Varargin 3: Mask radius
                    this.Volume = single(varargin{1});
                    this.coordAxes = single(varargin{2});
                    this.MaskRadius = single(varargin{3});                    
                else
                    disp('forwardProject(): Unknown input')
                    return
                end
            end

            if (this.RunFFTOnGPU == false)
                [origBox,interpBox,CASBox]=getSizes(single(this.VolumeSize), this.interpFactor,3);                
                this.CASVolume = CASFromVol_Gridder(this.Volume, this.kerHWidth, this.interpFactor, this.extraPadding);                
            end
            
            this.Set(); % Run the set function in case one of the arrays has changed
            mexMultiGPUForwardProject(this.objectHandle);            
            
            % Run the inverse FFT on the CAS images
            if (this.RunFFTOnGPU == false)
                this.Images = imgsFromCASImgs(this.CASImages, interpBox, []); 
            end
            
            ProjectionImages = this.Images;           
            
        end         
        %% BackProject - Run the back projection function
        function backProject(this, varargin)

            if ~isempty(varargin) > 0
                
                % A new set of images to back project was passed
                this.Images(:,:,:) = single(varargin{1});            
                
                if (this.RunFFTOnGPU == false)
                    % Run the forward FFT and convert the images to CAS images
                    [~,interpBox,~]=getSizes(single(this.VolumeSize), this.interpFactor,3);
                    newCASImgs = CASImgsFromImgs(this.Images, interpBox, []);
                    this.CASImages(:,:,:) = newCASImgs; % Avoid Matlab's copy-on-write
                end
                
                % A new set of coordinate axes to use with the back projection was passed
                tempAxes = single(varargin{2}); % Avoid Matlab's copy-on-write
                if ~isempty(this.coordAxes)
                    this.coordAxes(:) = tempAxes(:);
                else
                    this.coordAxes = tempAxes(:);
                end
            end

            this.Set(); % Run the set function in case one of the arrays has changed
            this.Set();
            this.Set();
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
           this.Volume = single(0 * this.Volume); 
           
           if (this.RunFFTOnGPU == false)
               this.CASVolume= single(0 * this.CASVolume); 
           end
        end
        %% reconstructVol - Reconstruct the volume by dividing by the plane density
        function Volume = reconstructVol(this, varargin)
            
            this.Set(); % Run the set function in case one of the arrays has changed
            mexMultiGPUReconstructVolume(this.objectHandle);
            
            if (this.RunFFTOnGPU == false)
                % Convert the CASVolume to Volume
                % Normalize by the plane density               
                this.CASVolume = this.CASVolume ./(this.PlaneDensity+1e-6);

                this.Volume=volFromCAS_Gridder(this.CASVolume,single(this.VolumeSize),this.kerHWidth, this.interpFactor);          
                
%                 this.Volume = this.Volume  / single(this.VolumeSize * this.VolumeSize );
                
            else
                this.Volume = this.Volume ./ 4;
            end

            Volume = single(this.Volume);         
        end   
        %% getVol - Convert the CAS Volume to Volume
        function Volume = getVol(this)
        
           this.Set(); % Run the set function in case one of the arrays has changed
           mexMultiGPUGetVolume(this.objectHandle);
           
            if (this.RunFFTOnGPU == false)
                % Convert the CASVolume to Volume
                this.Volume=volFromCAS_Gridder(this.CASVolume,single(this.VolumeSize),this.kerHWidth, this.interpFactor);
            end

           Volume = single(this.Volume); 
           
%            Volume = Volume  / single(this.VolumeSize * this.VolumeSize );
        end
        %% CheckParameters - check that all the parameters and variables are valid
        function CheckParameters(this)
            
            % Each x, y, and z component of the coordinate axes needs to have a norm of one
            for i = 1:size(this.coordAxes,2)
                for j = 1:3                    
                    if norm(this.coordAxes((j-1)*3+1:j*3),2) ~= 1
                        error("Invalid coordAxes parameter: Each x, y, and z component of the coordinate axes needs to have a norm of one")
                    end
                end
            end
                
            
            if this.interpFactor <= 0
                error("Invalid Interp Factor parameter: must be a non-negative value")
            end
            
            
        end
        
    end
end