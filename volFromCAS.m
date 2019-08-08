function vol=volFromCAS(volCAS,CASBox,interpBox,origBox,kernelHWidth)

%Get volume dimensions
volSize=size(volCAS,1);
volCenter=volSize/2+1;

%Create interpolated vol
interpCAS=zeros([interpBox.size interpBox.size interpBox.size],'single');
interpCAS=volCAS(CASBox.interpB:CASBox.interpE,...
                CASBox.interpB:CASBox.interpE,...
                CASBox.interpB:CASBox.interpE);

%Precompensate
interpFft=FromCAS(interpCAS);
clear interpCAS
interpVol=real(fftshift(ifftn(fftshift(interpFft))));
%Compensate for the normalization facor
% interpFactor=interpBox.size/origBox.size;
% interpVol=interpVol*(interpFactor)*origBox.size;  
interpVol=interpVol*interpBox.size;
%Precompensate
preComp=getPreComp(interpBox.size,kernelHWidth);
preComp=preComp';
interpVol=interpVol.*reshape(kron(preComp,kron(preComp,preComp)),...
                     interpBox.size,interpBox.size,interpBox.size);

%Extract out the original volume
vol=interpVol(interpBox.origB:interpBox.origE,...
              interpBox.origB:interpBox.origE,...
              interpBox.origB:interpBox.origE);






