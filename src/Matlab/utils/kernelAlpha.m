function alpha=kernelAlpha(hWidth)
% Return an alpha value for gridding interpolation.  Invalid arguments
% cause 0 to be returned.
%
   % alphavals=[0 0 5.2 0 10.2 0 13 0 17];  % nice alpha values for the 1.25 x oversampling.
     alphavals=[5.5 10.0 10.2 13 17];  % new values
if hWidth>5
    alpha=0;
else
    alpha=alphavals(hWidth);
end;
if alpha==0
    error(['invalid kernelsize ' num2str(kernelsize)]);
end;
