function [CASVol, interpBox, fftinfo]  = Vol_Preprocessing(vol, interpFactor)
    % Run MATLAB preprocessing to convert vol to CASVol
    
    %Basic upsampling parameters
    padWidth=3.0;
    kerHWidth=2.0;
%     kerTblSize=501;

    % Assume for now that vol has equal X, Y, and Z dimensions
    volSize = size(vol,1);
    
    [origBox,interpBox,CASBox]=getSizes(volSize,interpFactor,padWidth);

    %Set the fftw plan
    tmp=randn(interpBox.size*[1 1 1],'single');
    fftw('swisdom',[]);
    fftw('planner','measure');
    fft(tmp); 
    fftinfo=fftw('swisdom');

    CASVol=CASFromVol(vol, kerHWidth, origBox, interpBox, CASBox, fftinfo);

    
% 
%     kerHWidth = 2;
%     origBox = [];
%     origBox.size = size(vol,1);
%     origBox.center = size(vol,1)/2 + 1;
%     origBox.halfWidth = size(vol,1)/2 ;
% 
%     interpBox = [];
% 
%     interpBox.size = size(vol,1) * 2;
%     interpBox.center = size(vol,1) + 1;
%     interpBox.halfWidth = size(vol,1) / 2;
%     interpBox.origB = size(vol,1) / 2 + 1;
%     interpBox.origE = size(vol,1) * 2 - size(vol,1)/2;
% 
%     CASBox = [];
%     CASBox.size = 518;
%     CASBox.center = 260;
%     CASBox.interpB = 4;
%     CASBox.interpE = 515;





end