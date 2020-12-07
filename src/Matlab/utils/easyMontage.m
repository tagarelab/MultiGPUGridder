function easyMontage(x,figNum, Size)
%  'Size' A 2-element vector, [NROWS NCOLS]

if exist('figNum','var')
    figure(figNum);
else
    figure('Color', [1 1 1])
end

if exist('Size','var')
    try
        montage(reshape(x,[size(x,1) size(x,2) 1 size(x,3)]),'DisplayRange',[], 'Size', Size);
    catch
        warning("Failed to show figure. Likely contains NaNs")
    end
else
    try
        montage(reshape(x,[size(x,1) size(x,2) 1 size(x,3)]),'DisplayRange',[]);
    catch
        warning("Failed to show figure. Likely contains NaNs or has invaild dimensions.")
    end    
    
end
 
 
