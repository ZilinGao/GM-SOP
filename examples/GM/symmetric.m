function a_sym = symmetric(a)
if size(a,1) ~= size(a,2)
    error('symmetric demanding a square matrix')
end
a_sym = 0.5* (a + a');