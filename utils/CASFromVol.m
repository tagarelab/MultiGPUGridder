function [volCAS,interpSize]=CASFromVol(vol,kernelHWidth,origBox,interpBox,CASBox,fftinfo)%varargin)

%Set volume into interpVol
interpVol=zeros([interpBox.size interpBox.size interpBox.size],'single');
interpVol(interpBox.origB:interpBox.origE,interpBox.origB:interpBox.origE,interpBox.origB:interpBox.origE)=vol;

%Create kernel and precompensate
% preComp=getPreComp(interpBox.size,kernelHWidth);
% preComp=preComp';
% interpVol=interpVol.*reshape(kron(preComp,kron(preComp,preComp)),...
%                     interpBox.size,interpBox.size,interpBox.size);



%Set fftw plan and transform
fftw('swisdom',fftinfo);
%tic;
interpFft=fftshift(fftn(fftshift(interpVol)));
%toc;

close all
slice = 20;
subplot(2,2,1)
imagesc(interpVol(:,:,slice))
subplot(2,2,2)
imagesc(fftshift(interpVol(:,:,slice)))
subplot(2,2,3)
imagesc(real(fftn(fftshift(interpVol(:,:,slice)))))
subplot(2,2,4)
imagesc(real(fftshift(fftn(fftshift(interpVol(:,:,slice))))))


interpCAS=ToCAS(interpFft);
clear interpVol interpFft

volCAS = interpCAS;
%Pad it so interpolaton is not a problem
% volCAS=zeros(CASBox.size*[1 1 1],'single');
% volCAS(CASBox.interpB:CASBox.interpE,CASBox.interpB:CASBox.interpE,CASBox.interpB:CASBox.interpE)=interpCAS;

