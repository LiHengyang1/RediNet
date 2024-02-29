clc
clear;
close all
% rmdir TRAINimage s
mkdir TRAINimage
% rmdir TRAINlabel s
mkdir TRAINlabel
% rmdir VALIimage s
mkdir VALIimage
% rmdir VALIlabel s
mkdir VALIlabel
% rmdir TESTimage s
mkdir TESTimage
% rmdir TESTlabel s
mkdir TESTlabel


reslabel = 64;
resimage = 8;
setnum = 100;
maxiter0 = 20;
address1 = 'C:\Users\Hengyang_Li\Desktop\S\GeneratingDataset\'  
% Please insert the absolute path to save all dataset here. Must end with a \ 
tic

for ii = 1:setnum
    % tic
    focinum = randi(30) + 2;
    Xshift = randi([-4,3],focinum,1);
    Yshift = randi([-4,3],focinum,1);
    Zshift = randi([-4,3],focinum,1);
    imagetarget = zeros(reslabel,reslabel,reslabel,'single');
    for jj = 1:focinum
        imagetarget((reslabel/2+1) + Xshift(jj),(reslabel/2+1) + Yshift(jj),(reslabel/2+1) + Zshift(jj)) = 1;
    end
    cumu = ones(reslabel,reslabel,reslabel,'single');
    phase0 = ones(reslabel,reslabel,reslabel,'single') ;
    imagetarget00 = imagetarget;

    for time = 1:maxiter0
        image = fftshift(fftn(exp(1i * phase0)));
        imagenormal = abs(image) ./ max(max(max(abs(image))));
        cumu = cumu .* (imagetarget ./ (imagenormal + 1e-33)).^1;
        cumu = cumu ./ max(max(max(abs(cumu))));
        image0 = cumu .* imagetarget .* exp(1i * angle(image));
        phase0 = angle(ifftn(ifftshift(image0)));
    end
    if mod(ii,100) == 0
        ii
    end
    phase0 = gather(phase0);
    imageset = single(imagetarget00(29:29+7,29:29+7,29:29+7));
    labelset = single(angle(exp(1i * (phase0 - phase0(1,1,1)))));
    labelset(labelset < -3.12) = 3.14159;
    labelset = uint8(floor((labelset + pi) / 2 / pi * 255.999));
  
    if ii < 0.9*(setnum)+0.01
        t = imageset;
        nn = [address1,'TRAINimage\',num2str(ii),'.mat'];
        save(nn,'t')
        t = labelset;
        nn = [address1,'TRAINlabel\',num2str(ii),'.mat'];
        save(nn,'t')

    elseif ii > 0.95*(setnum) +0.01
        t = imageset;
        nn = [address1,'TESTimage\',num2str(ii),'.mat'];
        save(nn,'t')
        t = labelset;
        nn = [address1,'TESTlabel\',num2str(ii),'.mat'];
        save(nn,'t')

    else
        t = imageset;
        nn = [address1,'VALIimage\',num2str(ii),'.mat'];
        save(nn,'t')
        t = labelset;
        nn = [address1,'VALIlabel\',num2str(ii),'.mat'];
        save(nn,'t')
    end
      % toc
end
toc