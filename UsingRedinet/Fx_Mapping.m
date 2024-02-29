function finalphase = Fx_Mapping(mappingSpace1, differencephase_D1_mapped, differencephase_D2_mapped, differencephase_D3_mapped, N)

%% CodeCandy
% tic
% finalphase = arrayfun(@(x,y,z) mappingSpace1(x,y,z),differencephase_D1_mapped,differencephase_D2_mapped,differencephase_D3_mapped);
% toc

%% ForLoop
finalphase = zeros(N);
tic
for ii = 1:N
    for jj = 1:N
        finalphase(ii,jj) = mappingSpace1(differencephase_D1_mapped(ii,jj), differencephase_D2_mapped(ii,jj), differencephase_D3_mapped(ii,jj));
    end
end
toc

finalphase = exp(1i * finalphase * 2 * pi);

end