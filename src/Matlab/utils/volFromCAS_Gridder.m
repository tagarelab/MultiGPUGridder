function vol=volFromCAS(volCAS,origSize,kernelHWidth,interpFactor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This function converts a Fourier domain CAS-volume back into a spatial domain volume
%   of origSize.
%   
%   
%   The mandatory inputs are the CAS volume, origSize of the volume and 
%   the kernel half width for precompensation. 
%   Optional inputs are interpolation factor and padwidth. They have
%   default values of 2 and 4 resp.
%
%   The output is single precision volCAS, interpolated size, and the 
%   preComp function used in precompensation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Check if optional inputs are available, else set defaults
% interpFactor=2;
% if nargin >= 4  %InterpFactor is available
%     interpFactor=varargin{1};
% end
% padWidth=4;
% if nargin >= 5  %InterpFactor is available
%     padWidth=varargin{2};
% end

%Get volume dimensions
volSize=size(volCAS,1);
volCenter=volSize/2+1;


%Create interpolated vol
interpSize=origSize*interpFactor;
interpCenter=interpSize/2+1;
interpHalfWidth= interpCenter-1;
interpCAS=zeros([interpSize interpSize interpSize],'single');

volB=volCenter-interpHalfWidth;
volE=volCenter+interpHalfWidth-1;
interpCAS=volCAS(volB:volE,volB:volE,volB:volE);

%Precompensate
interpFft=FromCAS(interpCAS);
% clear interpCAS
interpVol=real(fftshift(ifftn(fftshift(interpFft))));

preComp=getPreComp(interpSize,kernelHWidth);
preComp=preComp';
interpVol=interpVol.*reshape(kron(preComp,kron(preComp,preComp)),...
                     interpSize,interpSize,interpSize);
                 
%Extract out the original volume
origCenter=origSize/2+1;
origHalfWidth= origCenter-1;
interpB=interpCenter-origHalfWidth;
interpE=interpCenter+origHalfWidth-1;
vol=interpVol(interpB:interpE,interpB:interpE,interpB:interpE);






