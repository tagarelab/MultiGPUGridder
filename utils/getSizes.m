function [orig,interp,CAS]=getSizes(volSize,interpFactor,padWidth)
%Original volume
orig.size=volSize;
orig.center=orig.size/2+1;
orig.halfWidth= orig.center-1;
%InterpVol
interp.size=orig.size*interpFactor;
interp.center=interp.size/2+1;
interp.halfWidth= interp.center-1;
interp.origB=interp.center-orig.halfWidth;
interp.origE=interp.center+orig.halfWidth-1;
%CAS volume
CAS.size=interp.size+2*padWidth;
CAS.center=CAS.size/2+1;
CAS.interpB=CAS.center-interp.halfWidth;
CAS.interpE=CAS.center+interp.halfWidth-1;