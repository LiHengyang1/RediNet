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

delta_D1 = lambda * f / L0 / 1;
delta_D2 = lambda * f / L0 / 1;
delta_D3 = 0.05;
differencephase_D1 = k * x / z * delta_D1 * 28;
differencephase_D2 = k * y / z * delta_D2 * 28;
differencephase_D3 = delta_D3 * (x.^3 + y.^3);
differencephase_D1_mapped = floor(mod(differencephase_D1,2 * pi) / 2 / pi * furiournumber) + 1;
differencephase_D2_mapped = floor(mod(differencephase_D2,2 * pi) / 2 / pi * furiournumber) + 1;
differencephase_D3_mapped = floor(mod(differencephase_D3,2 * pi) / 2 / pi * furiournumber) + 1;

%% Solving Fourier Coefficients with Neural Network
imagetarget = zeros(furiournumber,furiournumber,furiournumber);
imagetarget(34,33,30) = 1;
imagetarget(32,34,34) = 1;
imagetarget(32,32,35) = 1;

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

Uf1 = Fx_CZT(Fx_gaussianbeam(N,N,7,pixel) .* finalphase,0.25,N);
figure;
mesh(abs(Uf1));
colormap('jet')


finalphase0 = finalphase .* lensphase;

pixbig = 1000;
for ii = 1:50
    z = 325 + 3 * ii;
    [~, If(:,:,ii)] = Fx_CZT_SFFT(finalphase0,N,z,lambda,L0,0.1 / z * 475);
end
figure;
sliceViewer(If)
colormap jet