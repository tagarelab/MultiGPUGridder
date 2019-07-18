classdef gpuGridder < handle
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   The gpuGridder object. This object does the up/down sampling, the %
    %   complex to CAS conversion, and calls the CUDA kernels             %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties
        %Basic upsampling parameters
        interpFactor=2.0;
        padWidth=3.0;
        kerHWidth=2.0;
        kerTblSize=501;
        
        %fft
        fftinfo=[];

        
        %GPU for use
        gpuIds=1;
        gpuDev=[];
        
        
        %Various sizes and constants for
        %volumes and images
        origVolSize=0;
        imgSize=0;
        nAxes=0;
        rMax=0;
        origBox=[];
        interpBox=[];
        CASBox=[];
        
        %Constants used for iterating over
        % axes
        axesPerIteration=0;
        
        %The cuda kernels
        cudaFPKer = [];
        cudaBPKer = [];
        
        %Device pointers
        gpuVol=[];
        gpuKerTbl=[];
        gpuCoordAxes=[];
        gpuCASImgs=[];
        gpuCpxImgs=[];
        
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %       gpuGridder methods                                        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj=gpuGridder(volSize,maxAxes,interpFactor)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %    Constructor for the gridder                              %
            %    volSize: size of the volume to be forward/back projected %
            %    maxAxes: number of projection directions                 %
            %    interpFactor: gridding interpolation factor              %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            obj.origVolSize=volSize;
            obj.gpuDev=gpuDevice(obj.gpuIds);
            obj.interpFactor=interpFactor;
            
            %Determine BoxSizes and centers
            %origBox: original volume size, center, halfwidth
            %interpBox: box containing origBox for interpolation,
            %           origBox located at origB:origE 
            %CASBox: box containing the padded interpolated volume
            %           interpBox loated at interpB:interpE
            [obj.origBox,obj.interpBox,obj.CASBox]=getSizes(volSize,obj.interpFactor,obj.padWidth);
            obj.imgSize=obj.interpBox.size;
            obj.rMax=obj.imgSize/2-1;
            
            %Determine maxAxes
            obj.axesPerIteration=getMaxAxes(obj.gpuDev.AvailableMemory,obj.CASBox.size,obj.imgSize,obj.kerTblSize);
            obj.axesPerIteration=min(obj.axesPerIteration,maxAxes); %If maxAxes is smaller
            
            %Allocate memory on GPU
            obj.gpuCoordAxes=zeros([obj.axesPerIteration*9 1],'single','gpuArray');
            obj.gpuCASImgs=zeros([obj.imgSize obj.imgSize obj.axesPerIteration],'single','gpuArray');
            
            %Load the kernel table 
            %Kernel has the kaiser bessel values for gridding
            obj.gpuKerTbl=gpuArray(single(getKernelFiltTable(obj.kerHWidth,obj.kerTblSize)));
            
            %Create the CUDA kernel
            obj.cudaFPKer = parallel.gpu.CUDAKernel('gpuForwardProjectKernel.ptx','gpuForwardProjectKernel.cu');
            obj.cudaBPKer = parallel.gpu.CUDAKernel('gpuBackProjectKernel.ptx','gpuBackProjectKernel.cu');
            
            %Set the fftw plan
            tmp=randn(obj.interpBox.size*[1 1 1],'single');
            fftw('swisdom',[]);
            fftw('planner','measure');
            fft(tmp); 
            obj.fftinfo=fftw('swisdom');
        end
        
                
        function delete(obj)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Object destructor. Explicitly release gpu memory            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                clear obj.gpuVol;
                clear obj.gpuCoordAxes;
                clear obj.gpuCASImgs;
                clear obj.gpuKerTbl;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   Set and Reset volume
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function setInterpCASVol(obj,CASVol)
            obj.gpuVol=gpuArray(CASVol);
        end        
        function setVolume(obj,vol)
            CASVol=CASFromVol(vol,obj.kerHWidth,obj.origBox,obj.interpBox,obj.CASBox,obj.fftinfo);
            obj.setInterpCASVol(CASVol);
        end
        function setVolumeFft(obj,volFft)
            fftw('swisdom',fftinfo);
            vol=real(fftshift(ifftn(fftshft(volFft))));
            obj.setVolume(obj,vol);
        end

        function resetVolume(obj)
            obj.gpuVol=zeros(obj.CASBox.size*[1 1 1],'single','gpuArray');
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function imgs=forwardProject(obj,coordAxes)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Forward Project function
            %   Function assumes that the volume has been set             %
            %   coordAxes: 9XN array where N=num of projection dirs       %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Transfer CoordAxes to Gpu
            nAxes=int32(size(coordAxes,2));
            obj.nAxes=nAxes;
            nImgs=nAxes;
            obj.gpuCoordAxes=gpuArray(coordAxes(:));
            %Call the kernel
            %Block and grid sizes
            blockWidth=4;
            gridWidth=round(obj.imgSize/blockWidth);
            obj.cudaFPKer.ThreadBlockSize=[blockWidth blockWidth 1];
            obj.cudaFPKer.GridSize=[gridWidth gridWidth 1];

            %Call the kernel
            obj.gpuCASImgs=feval(obj.cudaFPKer,...
                            obj.gpuVol, obj.CASBox.size,...
                            obj.gpuCASImgs,obj.imgSize,...
                            obj.gpuCoordAxes, nAxes,single(obj.rMax),...
                            obj.gpuKerTbl, int32(obj.kerTblSize), single(obj.kerHWidth));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Functions to return images
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function imgs=getInterpCASImgs(obj)
            imgs=gather(obj.gpuCASImgs(:,:,1:obj.nAxes));
            
        end
            
        function imgs=getImgs(obj)          
            imgs=imgsFromCASImgs(obj.getInterpCASImgs(),obj.interpBox,obj.fftinfo);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   Back project functions
        %   There are three back project functions:
        %       backProjectInterpCASImg back projects from interpolated 
        %                   images whose FT has been convered to CAS
        %       backProject back projects spatial domain imgs
        %       backProjectFft back projects Fts of spatial domain imgs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
           function backProjectInterpCASImg(obj,imgs,coordAxes)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Back projects interpolated CAS images
            %   This routine is the work horse that is called by other
            %       routines
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Move to GPU 
            nImgs=size(imgs,3);
            nAxes=int32(nImgs);
            obj.gpuCASImgs(:,:,1:nImgs)=imgs;

            obj.gpuCoordAxes=gpuArray(coordAxes(:));
            
            %Block and grid sizes
            blockWidth=4;
            obj.cudaBPKer.ThreadBlockSize=[blockWidth blockWidth blockWidth];
            gridWidth=ceil( (obj.CASBox.size)/blockWidth);
            obj.cudaBPKer.GridSize=[gridWidth gridWidth gridWidth];
            %BackProject
            obj.gpuVol=feval(obj.cudaBPKer,...
                            obj.gpuVol, obj.CASBox.size,...
                            obj.gpuCASImgs,obj.imgSize,...
                            obj.gpuCoordAxes, nAxes,single(obj.rMax),...
                            obj.gpuKerTbl, int32(obj.kerTblSize), single(obj.kerHWidth));
           end
                
           function backProject(obj,imgs,coordAxes)
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %    Back project spatial domain images
           %    Spatial domain images are converted to intepCAS images
           %        and then backprojected using the above backProject
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Convert to CAS
%                 nImgs=size(imgs,3);
%                 nAxes=int32(nImgs);
%                 interpImgs=zeros([obj.interpBox.size obj.interpBox.size nImgs],'single');
%                 interpImgs(obj.interpBox.origB:obj.interpBox.origE,...
%                         obj.interpBox.origB:obj.interpBox.origE,:)=imgs;
%                 interpImgs=fftshift2(fft2(fftshift2(interpImgs)));
%                 interpImgs=real(interpImgs)+imag(interpImgs);

                CASImgs=CASImgsFromImgs(imgs,obj.interpBox,obj.fftinfo);
                obj.backProjectInterpCASImg(CASImgs,coordAxes);
           end
           
           function backProjectFft(obj,imgFft,coordAxes)
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %    Back project fft images in orig size
           %    Images are inverse fft'ed and the above backproject is
           %    called
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               imgs=real(fftshift2(ifft2(fftshift2(imgFft))));
           end
           
           
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %    Back projection volume returning functions. 
           %    These functions return the back projected volume in 
           %        appropriate form
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           
           function vol=getInterpCASVol(obj)
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %    Returns the interpolated CAS fourier transform
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               vol=gather(obj.gpuVol);
           end
           
           function vol=getVol(obj)
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %    Returns the spatial domain volume
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               interpCASVol=obj.getInterpCASVol();
               vol=volFromCAS(interpCASVol,obj.CASBox,obj.interpBox,obj.origBox,...
                                obj.kerHWidth);
           end
           
           
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %    Volume reconstruction functions
           %    These are meant to be used for testing the gridder
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                   
            function volWt=getVolWt(obj,coordAxes)
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %    Returns the density of inserted planes
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                nAxes=int32(size(coordAxes,2));
                interpImgs=ones([obj.interpBox.size obj.interpBox.size nAxes],'single');
                obj.resetVolume();
                obj.backProjectInterpCASImg(interpImgs,coordAxes);
                volWt=obj.getInterpCASVol();
            end
        
           function vol=reconstructVol(obj,coordAxes)
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %    Returns back the spatial domain reconstructed volume
           %    after downweighting by the density of inserted planes
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               volR=obj.getInterpCASVol();
               volWt=obj.getVolWt(coordAxes);
               volR=volR./(volWt+1e-6);
               vol=volFromCAS(volR,obj.CASBox,obj.interpBox,obj.origBox,...
                                obj.kerHWidth); 
                
           end

    end
end
