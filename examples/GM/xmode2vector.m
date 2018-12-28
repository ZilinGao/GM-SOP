dim = 6;
%different index in 3 dim
index = nchoosek([1:dim] , 3);
%1 or 2 index are same
x= [1:dim  * dim]';
y = ceil(1./dim .* x);
y(:,2) = y;
r = ones(dim,2);
pp = mod(x , dim);
y(:,3) = (pp == 0) .* dim + pp;
index(end+1:end+size(y,1) , :) = y ;
index_mode = 

data = rand(9,9,9)*100;
get = data(index);