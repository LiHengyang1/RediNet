%% 1 Dimensional parameters distract
clear;
close all;
lambda = 1064e-6;
f = 400;
z = f;
N = 1000;
pixel = 0.0125;
L0 = N * pixel;
k = 2 * pi / lambda;
furiournumber = 64;
x = linspace(-L0 / 2 + pixel / 2,L0 / 2 - pixel / 2,N);
[x,y] = meshgrid(x,x);
[theta,r] = cart2pol(x,y);
theta = theta + pi;
lensphase = exp(-1i * k * (x.^2 + y.^2) / 2 / f);
r0 = 6.25;

delta_D1 = 5 * lambda * f / L0 / 1;
delta_D2 = 5 * lambda * f / L0 / 1;
delta_D3 = 0;
differencephase_D1 = k * x / f * delta_D1;
differencephase_D2 = k * y / f * delta_D2;
differencephase_D3 = 0 * r;
differencephase_D1_mapped = floor(mod(differencephase_D1,2 * pi) / 2 / pi * furiournumber) + 1;
differencephase_D2_mapped = floor(mod(differencephase_D2,2 * pi) / 2 / pi * furiournumber) + 1;
differencephase_D3_mapped = floor(mod(differencephase_D3,2 * pi) / 2 / pi * furiournumber) + 1;

%% Solving Fourier Coefficients with Neural Network
imagetarget = zeros(furiournumber,furiournumber,furiournumber);
imagetarget(28 + 4,28 + 4,28 + 1) = 1;
imagetarget(28 + 4,28 + 6,28 + 8) = 1;
imagetarget(28 + 6,28 + 4,28 + 1) = 1;
imagetarget(28 + 6,28 + 6,28 + 6) = 1;

figure;
sliceViewer(imagetarget);

imagetargetESPR = imagetarget(29:29+7,29:29+7,29:29+7);
mappingSpace1 = Fx_NNinference3D(imagetargetESPR,0);        %1无补偿
mappingSpace1 = circshift(mappingSpace1,9,3);               %trick
FourierCoef1 = fftshift(fftn(exp(1i * 2 * pi * mappingSpace1)));
figure
sliceViewer(mappingSpace1);
figure
sliceViewer(abs(FourierCoef1));

%% mapping and multiplexing

finalphase = Fx_Mapping(mappingSpace1, differencephase_D1_mapped, differencephase_D2_mapped, differencephase_D3_mapped, N);
figure;
imagesc(angle(finalphase))
colormap(othercolor('BuOr_12'))
%% Obeserving
Uin = Fx_gaussianbeam(N,N,4,pixel);
Uin(r>6) = 0;

ratio = 0.05;
Uf1 = Fx_CZT(Uin .* finalphase,ratio,N);
figure;
imagesc(abs(Uf1.^2));
colormap(jet)

comp = pi * linspace(1,N,N);
[compx,compy] = meshgrid(comp,comp);
gratingComp = exp(1i * 0.999 * ratio * (compx + compy));
Uf1 = Uf1 .* gratingComp;
figure;
imagesc(angle(Uf1))
Uflookphase = abs(Uf1) .* angle(Uf1);
figure;
imagesc(Uflookphase)
colormap(jet)