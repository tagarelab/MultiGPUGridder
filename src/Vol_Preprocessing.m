function [CASVol, CASBox, origBox, interpBox, fftinfo]  = Vol_Preprocessing(vol, interpFactor)
    % Run MATLAB preprocessing to convert some input vol to CASVol
    
    %Basic upsampling parameters
    padWidth=3.0;
    kerHWidth=2.0;

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

end