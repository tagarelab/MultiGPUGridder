function preComp=getPreCompFromTable(volSize,hWidth)

ovr=4; %Fixed oversampling rate

alpha=kernelAlpha(hWidth);
delta=1/ovr;
x=[-hWidth:delta:hWidth];

% x=[-hWidth/2:1/ovr:hWidth/2];
% x=x(find((x>=-hWidth/2)&(x<=hWidth/2)));
w=gridLibKaiser(hWidth,alpha,x);
w=w-w(1);
w(end)=0;
% w(1)=0;
% w(end)=0;
wHalfWidth=(numel(x)-1)/2;


x=zeros(1,volSize*ovr);
xCenter=(numel(x)/2)+1;
x(xCenter-wHalfWidth:xCenter+wHalfWidth)=w;
 % Compute the FT (=IFT) and normalize it.
preComp=real(fftshift(fft(fftshift(x))));
preCompCenter=volSize*ovr/2+1;
fc0=preComp(preCompCenter);   % pick up the zero-frequency point for normalization.
preCompHalfWidth=volSize/2;
preComp=fc0./(preComp(preCompCenter-preCompHalfWidth:...
                preCompCenter+preCompHalfWidth-1));
    
