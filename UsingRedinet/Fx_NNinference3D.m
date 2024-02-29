function mappingSpace = Fx_NNinference3D(imagetargetPY,chooseNet)

imagetargetPY = single(imagetargetPY);
imagetargetPY = py.numpy.array(imagetargetPY);
mappingSpace = py.PYInference.pyfun0(imagetargetPY);
mappingSpace = double(mappingSpace);
% mappingSpace(mappingSpace > 1) = 1;
% mappingSpace(mappingSpace < 0) = 0;

end