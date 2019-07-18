function plotProfiles(vol,center,w,fignum,hState)
figure(fignum);
volMax=max(vol(:));
volMin=min(vol(:));
for i=1:w,
    subplot(1,w,i);
    %center+i-1
    plot(squeeze(vol(:,center,center+i-1)));
    ylim([volMin volMax+0.1]);
    if hState==1
        hold on;
    else
        hold off;
    end
end
