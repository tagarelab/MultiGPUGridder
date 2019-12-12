function easyMontage(x,figNum)
figure(figNum);
try
    montage(reshape(x,[size(x,1) size(x,2) 1 size(x,3)]),'DisplayRange',[]);
catch
   disp("easyMontage Failed. Likely an Invalid or deleted object or a NaN.") 
end