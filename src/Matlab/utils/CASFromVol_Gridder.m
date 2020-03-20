function [volCAS,interpSize]=CASFromVol(vol,kernelHWidth,interpFactor, padWidth)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This function converts a spatial domain volume into a fourier domain
%   interpolated, precompensated, padded and CAS-converted volume. 
%   
%   The mandatory inputs are the volume and the kernel half width for
%   precompensation. Optional inputs are interpolation factor and padwidth.
%   They have default values of 2 and 4 resp.
%
%   The output is single precision volCAS, interpolated size, and the 
%   preComp function used in precompensation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %Check if optional inputs are available, else set defaults
% interpFactor=2;
% if nargin >= 3  %InterpFactor is available
%     interpFactor=varargin{1};
% end
% padWidth=4;
% if nargin >= 4  %padWidth is available
%     padWidth=varargin{2};
% end
    

%Calculate the size, center and half width of the input volume
origSize=size(vol,1);
origCenter=origSize/2+1;
origHalfWidth= origCenter-1;

%Create interpolated vol
interpSize=origSize*interpFactor;
interpCenter=interpSize/2+1;
interpHalfWidth= interpCenter-1;
interpVol=zeros([interpSize interpSize interpSize],'single');
origB=origCenter-origHalfWidth;
origE=origCenter+origHalfWidth-1;
interpB=interpCenter-origHalfWidth;
interpE=interpCenter+origHalfWidth-1;
interpVol(interpB:interpE,interpB:interpE,interpB:interpE)=...
    single(vol(origB:origE,origB:origE,origB:origE));



%Create kernel and precompensate
preComp=getPreComp(interpSize,kernelHWidth);
preComp=preComp';
interpVol=interpVol.*reshape(kron(preComp,kron(preComp,preComp)),...
                    interpSize,interpSize,interpSize);
interpFft=fftshift(fftn(fftshift(interpVol)));
interpCAS=ToCAS(interpFft);
clear interpVol interpFft


%Pad it so interpolaton is not a problem
volSize=interpSize+2*padWidth;
volCAS=zeros(volSize*[1 1 1],'single');
volCenter=volSize/2+1;
volB=volCenter-interpHalfWidth;
volE=volCenter+interpHalfWidth-1;
volCAS(volB:volE,volB:volE,volB:volE)=interpCAS;
