function [Uf,If] = Fx_CZT_SFFT(U0,N,z0,lambda,L0,ratio)

    k = 2 * pi / lambda;
    gridbase = ([0 : N - 1] - (N - 1) / 2).';  
    [U,V] = meshgrid(gridbase,gridbase);
    pixel_L0 = L0 / N;
    xx0 = U .* pixel_L0;
    yy0 = V .* pixel_L0;
    Fresnelcore = exp(1i * k / 2 / z0 * (xx0.^2 + yy0.^2));
    f2 = U0 .* Fresnelcore;  %S-FFT计算菲涅耳衍射时的傅里叶变换函数
    Uf = Fx_CZT(f2,ratio,N) / z0;%对N*N点的离散函数f2作FFT计算
    T = L0 / N;         %能量补偿
    Uf = Uf * T * T;    
    If = Uf .* conj(Uf);%形成衍射场强度分布
end