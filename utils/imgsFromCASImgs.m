function r=imgsFromCASImgs(r,interpBox,fftinfo)
% Convert the "cosine and sine" real array r back to complex c.  We assume
% the origin is at the center of the arrays (i.e. fftshift has been
% applied.)  The -n/2 point is set to zero, which is at c(1,1).
% This routine handles a stack of 2D arrays

% [s1, s2, s3]=size(r);
% o1=min(2,s1);
% o2=min(2,s2);
% imgs=complex(zeros(s1,s2,s3,'single'));
% imgs(o1:s1,o2:s2,:)=0.5*(r(o1:s1,o2:s2,:)+r(s1:-1:o1,s2:-1:o2,:)...
%     +1i*(r(o1:s1,o2:s2,:)-r(s1:-1:o1,s2:-1:o2,:)));

% fftw('swisdom',fftinfo);
% imgs=real(fftshift2(ifft2(fftshift2(imgs))));  %Inverse Fft
% 
% %Extract out the original images
% interpB=interpBox.origB;
% interpE=interpBox.origE;
% imgs=imgs(interpBox.origB:interpBox.origE,...
%             interpBox.origB:interpBox.origE,:);

        
% Uses much less memory while taking about 10% longer
% For larger volumes (>256) and more projections (>5,000) the above code uses more than 40 GB of memory
r = complex(r);
r(1,:,:) = [];
r(:,1,:) = [];
r = 0.5*(r + rot90(r,2)) + 1i * 0.5*(r - rot90(r,2)) ;
r = padarray(r,[1 1],0,'pre');

fftw('swisdom',fftinfo);
r=real(fftshift2(ifft2(fftshift2(r))));  %Inverse Fft


%Extract out the original images
interpB=interpBox.origB;
interpE=interpBox.origE;
r=r(interpBox.origB:interpBox.origE,...
            interpBox.origB:interpBox.origE,:);
