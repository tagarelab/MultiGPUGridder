function easyMontage(x,figNum)
figure(figNum);
montage(reshape(x,[size(x,1) size(x,2) 1 size(x,3)]),'DisplayRange',[]);