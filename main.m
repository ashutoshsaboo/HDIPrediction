close all; clear;

[countryName, LEB, EYS, MYS, GNI, HDI] = textread('full.txt', '%s %f %f %f %f %f');

hidden_layer_size = 100; 
num_labels = 1;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = mapFeature(LEB,EYS, MYS, GNI);
A = [LEB, EYS, MYS, GNI];

X = [X, log(X), log(A), A];
m = size(X, 1);
y = HDI;

[X mu sigma] = featureNormalize(X);

input_layer_size =  size(X, 2);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


best_lambda = 0;
best_J=1000;

for i = 0:0
	lambda = .01*i;
	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	% initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
	initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

	% initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];
	initial_nn_params = [initial_Theta1(:) ; initial_Theta3(:)];

	fprintf('\nTraining Neural Network for lambda = %f\n', lambda);

	%  After you have completed the assignment, change the MaxIter to a larger
	%  value to see how more training helps.
	options = optimset('MaxIter', 1000);

	%  You should also try different values of lambda
	% lambda = 1;

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X, y, lambda);

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)

	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	%                  hidden_layer_size, (input_layer_size + 1));

	% Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), hidden_layer_size, (hidden_layer_size + 1));
	% Theta3 = reshape(nn_params((1 + (hidden_layer_size * (hidden_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));


	% fprintf('Program paused. Press enter to continue.\n');
	% pause;

	J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

	[cvtest_countryName, cvtest_LEB, cvtest_EYS, cvtest_MYS, cvtest_GNI, cvtest_HDI] = textread('cvset.txt', '%s %f %f %f %f %f');

	cvtest_X = mapFeature(cvtest_LEB, cvtest_EYS, cvtest_MYS, cvtest_GNI);
	cvtest_A = [cvtest_LEB, cvtest_EYS, cvtest_MYS, cvtest_GNI];
	cvtest_X = [cvtest_X, log(cvtest_X), log(cvtest_A), cvtest_A];
	cvtest_m = size(cvtest_X, 1);
	cvtest_y = cvtest_HDI;

	[cvtest_X mu sigma] = featureNormalize(cvtest_X);

	[J bleh hx] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, cvtest_X, cvtest_y, lambda);

	
	% for i = 1:m

	fprintf('Cost on CVset: %e\n', J);

	if J < best_J
		best_J = J;
		best_lambda = lambda;
	end
end

lambda = best_lambda;
[cvtest_countryName, cvtest_LEB, cvtest_EYS, cvtest_MYS, cvtest_GNI, cvtest_HDI] = textread('test.txt', '%s %f %f %f %f %f');

cvtest_X = mapFeature(cvtest_LEB, cvtest_EYS, cvtest_MYS, cvtest_GNI);
cvtest_A = [cvtest_LEB, cvtest_EYS, cvtest_MYS, cvtest_GNI];
cvtest_X = [cvtest_X, log(cvtest_X), log(cvtest_A), cvtest_A];
cvtest_m = size(cvtest_X, 1);
cvtest_y = cvtest_HDI;

[cvtest_X mu sigma] = featureNormalize(cvtest_X);

[J bleh hx] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, cvtest_X, cvtest_y, lambda);
fprintf('At best lambda (%f) J on Test set= %e\n', lambda, J);
fprintf('comparing the values predicted vs actual \n');

% [hx cvtest_y];

A_y = [hx - cvtest_y];
A_X = [1:size(A_y, 1)]';
plotData(A_X, A_y);

meanerror = sum(A_y) / size(A_y, 1)
count = 0;
for i=1:size(A_y)
	if A_y(i) < 0.03 && A_y(i) > -0.03
		count = count+1;
	end
end

Acc = (count / size(A_y, 1))*100

