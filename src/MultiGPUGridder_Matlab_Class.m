%MultiGPUGridder_Matlab_Class Example MATLAB class wrapper to an underlying C++ class
classdef MultiGPUGridder_Matlab_Class < handle
    properties (SetAccess = public, Hidden = false)
        
        objectHandle; % Handle to the underlying C++ class instance

        % Int 32 type variables        
        VolumeSize;        
        NumAxes;
        GPUs = int32([0, 1, 2, 3]);
        MaxAxesToAllocate;
        nStreams = 2;
        
        % Single type variables        
        interpFactor;
        kerHWidth = 2;        
        extraPadding = 3;    
        KBTable;
        PlaneDensity;
        coordAxes;      
        CASVolume;
        CASImages;
        Volume;
        Images;                 
    end
    
    methods
        %% Constructor - Create a new C++ class instance 
        function this = MultiGPUGridder_Matlab_Class(varargin)  
            % Inputs are:            
            % (1) VolumeSize
            % (2) nCoordAxes
            % (3) interpFactor        
            
            this.VolumeSize = int32(varargin{1});
            this.NumAxes = int32(varargin{2});
            this.interpFactor = single(varargin{3});
                        
            gridder_Varargin = [];
            gridder_Varargin{1} = int32(varargin{1});
            gridder_Varargin{2} = int32(varargin{2});
            gridder_Varargin{3} = single(varargin{3});
            gridder_Varargin{4} = int32(length(this.GPUs));
            gridder_Varargin{5} = int32(this.GPUs);
            
            % Create the gridder instance
            this.objectHandle = mexCreateGridder(gridder_Varargin{1:5});           
                       
            % Initilize the output projection images array
            ImageSize = [this.VolumeSize, this.VolumeSize, this.NumAxes];
            this.Images = zeros(ImageSize(1), ImageSize(2), ImageSize(3), 'single');
            
            % Load the Kaiser Bessel lookup table
            KB_Vector = load("KB_Vector.mat");
            this.KBTable = single(KB_Vector.KB_Vector);
            
            % Create the Volume array
            this.Volume = single(zeros(repmat(this.VolumeSize, 1, 3)));  
            
            % Create the CASVolume array
            this.CASVolume = single(zeros(repmat(size(this.Volume, 1) * this.interpFactor + this.extraPadding * 2, 1, 3)));  
                     
            % Create the CASImages array
            CASImagesSize = size(this.Volume, 1) * this.interpFactor; 
            this.CASImages = single(zeros([CASImagesSize, CASImagesSize, this.NumAxes]));    
%             this.CASImages = single(zeros(repmat(size(this.Volume, 1) * this.interpFactor + this.extraPadding * 2, 1, 3)));         

    
            % Create the PlaneDensity array
            this.PlaneDensity = single(zeros(repmat(size(this.Volume, 1) * this.interpFactor + this.extraPadding * 2, 1, 3)));  
           

        end        
        %% Deconstructor - Delete the C++ class instance 
        function delete(this)
            mexDeleteGridder(this.objectHandle);
        end  
        %% SetVariables - Set the variables of the C++ class instance 
        function Set(this)
            
            if (isempty(this.coordAxes) || isempty(this.Volume) || isempty(this.CASVolume) || isempty(this.Images) ...
                || isempty(this.GPUs) || isempty(this.KBTable) || isempty(this.nStreams))
                error("Error: Required array is missing in Set() function.")             
            end                    
            
            [varargout{1:nargout}] = mexSetVariables('SetCoordAxes', this.objectHandle, single(this.coordAxes(:)), int32(size(this.coordAxes(:))));
            [varargout{1:nargout}] = mexSetVariables('SetVolume', this.objectHandle, single(this.Volume), int32(size(this.Volume)));
            [varargout{1:nargout}] = mexSetVariables('SetCASVolume', this.objectHandle, single(this.CASVolume), int32(size(this.CASVolume)));                           
             [varargout{1:nargout}] = mexSetVariables('SetCASImages', this.objectHandle, single(this.CASImages), int32(size(this.CASImages)));  
            [varargout{1:nargout}] = mexSetVariables('SetImages', this.objectHandle, single(this.Images), int32(size(this.Images)));
            [varargout{1:nargout}] = mexSetVariables('SetGPUs', this.objectHandle, int32(this.GPUs), int32(length(this.GPUs)));
            [varargout{1:nargout}] = mexSetVariables('SetKBTable', this.objectHandle, single(this.KBTable), int32(size(this.KBTable)));           
            [varargout{1:nargout}] = mexSetVariables('SetNumberStreams', this.objectHandle, int32(this.nStreams)); 
            [varargout{1:nargout}] = mexSetVariables('SetPlaneDensity', this.objectHandle, single(this.PlaneDensity), int32(size(this.PlaneDensity)));
               
            % Set the optional arrays
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
        function ForwardProject(this, varargin)

            if ~isempty(varargin) > 0
                % A new set of coordinate axes was passed
                this.coordAxes = varargin{1};            
            end

            [origBox,interpBox,CASBox]=getSizes(single(this.VolumeSize), this.interpFactor,3);
            this.CASVolume = CASFromVol(this.Volume, this.kerHWidth, origBox, interpBox, CASBox, []);

            this.Set(); % Run the set function in case one of the arrays has changed
            mexForwardProject(this.objectHandle);
            
        end         
        %% BackProject - Run the back projection function
        function backProject(this, varargin)

            if ~isempty(varargin) > 0
                % A new set of images to back project was passed
                this.Images = varargin{1};

                % A new set of coordinate axes to use with the back projection was passed
                this.coordAxes = varargin{2};
            end

            this.Set(); % Run the set function in case one of the arrays has changed
            mexBackProject(this.objectHandle);
            
            % Convert the CASVolume to Volume
            [origBox,interpBox,CASBox]=getSizes(single(this.VolumeSize), this.interpFactor,3);
            
            % Normalize by the plane density
            this.CASVolume = this.CASVolume ./(this.PlaneDensity+1e-6);
            this.Volume=volFromCAS(this.CASVolume,CASBox,interpBox,origBox,this.kerHWidth);

        end   
        %% setVolume - Set the volume
        function setVolume(this, varargin)
            % The new volume will be copied to the GPUs during this.Set()
           this.Volume = varargin{1}; 
        end
        %% resetVolume - Reset the volume
        function resetVolume(this)
            % Multiply the volume by zero to reset. The resetted volume will be copied to the GPUs during this.Set()
           this.Volume = 0 * this.Volume; 
        end
    end
end