classdef gpuBatchGridder < handle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Batch gridder object. This object splits multiple forward and back    %
%   projection operations in batches and runs them on the gpuGridder.     %
%                                                                         %
%   Hemant Tagare 7/18/19                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Class variables
    properties
        gridder=[]; %Holds the gpuGridder
        maxAxes=0;  %The total number of projection directions
    end
    
    %Class methods
    methods
        function obj=gpuBatchGridder(volSize,maxAxes,interpFactor)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Constructor                                               %
            %   volSize: original size of the volume to be gridded        %
            %   maxAxes: total number of projection directions            %
            %   interpFactor: Gridding upsampling factor, default=2       %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %Get the gridder and set the maxAxes
            obj.gridder=gpuGridder(volSize,maxAxes,interpFactor);
            obj.maxAxes=maxAxes;
        end
        
        function setCASVolume(obj,vol)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Sets the gpuGridder CAS Volume                            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj.gridder.setInterpCASVol(vol);
        end
        
        function setVolume(obj,vol)
            obj.gridder.setVolume(vol);
        end
        
        function imgs=forwardProject(obj,coordAxes)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Forward projects the volume along coordAxes               %
            %   This method batches the projections, uses the gpuGridder  %
            %   to forward project, and then collects the projected images%
            %   coordAxes = 9 X N array, where N=num of projections       %
            %       Each column of coordAxes is x,y,z of the projection   %
            %       reference frame                                       %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            nAxes=size(coordAxes,2);          %Get the number of projections
            imgSize=obj.gridder.origBox.size; %Get image size
            imgs=zeros([imgSize imgSize nAxes],'single'); %Allocate images
            
            axesPerIteration=nAxes%obj.gridder.axesPerIteration; %Size of each batch
            nIter=ceil(nAxes/axesPerIteration);  %Number of batches
            
            for iter=1:nIter
                %For each batch, forward project and collect images
                axesStart=1+(iter-1)*axesPerIteration;
                axesEnd=min(iter*axesPerIteration,nAxes);
                obj.gridder.forwardProject(coordAxes(:,axesStart:axesEnd));
                imgs(:,:,axesStart:axesEnd)=obj.gridder.getImgs();
            end
        end
        
        function resetVolume(obj)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Reset the volume on the gridder to zeros
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj.gridder.resetVolume();
        end
        
        function backProject(obj,imgs,coordAxes)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Back projects the images    along coordAxes               %
            %   This method batches the projections, uses the gpuGridder  %
            %   to back project. Does not collect the volume              %
            %   coordAxes = 9 X N array, where N=num of projections       %
            %       Each column of coordAxes is x,y,z of the projection   %
            %       reference frame                                       %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            nAxes=size(coordAxes,2);        %Get the number of projections
            axesPerIteration=obj.gridder.axesPerIteration; %Size of each batch
            nIter=ceil(nAxes/axesPerIteration); %Number of batches
            for iter=1:nIter,
                axesStart=1+(iter-1)*axesPerIteration;
                axesEnd=min(iter*axesPerIteration,nAxes);
                obj.gridder.backProject(imgs(:,:,axesStart:axesEnd),coordAxes(:,axesStart:axesEnd));
            end
        end
        
        function vol=getVol(obj)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   Returns the volume on the gpuGridder                      %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vol=obj.gridder.getVol();
        end
        
        function vol=reconstructVol(obj,coordAxes)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %   This is a simplified Weiner filter reconstructor          %
            %   Only used for testing                                     %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            volR=obj.gridder.getInterpCASVol();
            
            nAxes=size(coordAxes,2);
            axesPerIteration=obj.gridder.axesPerIteration;
            nIter=ceil(nAxes/axesPerIteration);
            volWt=0;
            for iter=1:nIter,
                axesStart=1+(iter-1)*axesPerIteration;
                axesEnd=min(iter*axesPerIteration,nAxes);
                volWt=volWt+obj.gridder.getVolWt(coordAxes(:,axesStart:axesEnd));
           end
            volR=volR./(volWt+1e-6);
            vol=volFromCAS(volR,obj.gridder.CASBox,obj.gridder.interpBox,obj.gridder.origBox,...
                            obj.gridder.kerHWidth); 
        end
            
    end
end