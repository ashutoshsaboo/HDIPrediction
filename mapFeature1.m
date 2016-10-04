function out = mapFeature1(X1, X2, X3, X4)
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
for i = -3:3
    for j = -3:i
    	for k = -3:j
    		for l = -3:k
    			% count = count+1;
        		out(:, end+1) = (X1.^(i-j-k-l)).*(X2.^j).*(X3.^k).*(X4.^l);
        	end
        end
    end
end

% size(out)

end