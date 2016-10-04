function out = mapFeature(X1, X2, X3, X4)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 3;
% count=0;
out = ones(size(X1(:,1)));
A = [1/3, 1/2, 1, 2];
for i = 1:size(A, 2)
    for j = 1:i
    	for k = 1:j
    		for l = 1: k
    			% count = count+1;
        		out(:, end+1) = (X1.^(A(i)-A(j)-A(k)-A(l))).*(X2.^A(j)).*(X3.^A(k)).*(X4.^A(l));
        		% [A(i), A(j), A(k), A(l), A(i)+ A(j)+ A(k)+ A(l)]
        	end
        end
    end
end

% size(out)
out = out(:, 2:end);

end