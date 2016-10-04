function [J grad hx] = nnCostFunction2(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));


param1_1 = (1 + (hidden_layer_size * (input_layer_size + 1)));
param1_2 = (((hidden_layer_size * (input_layer_size + 1))) + ((hidden_layer_size * (hidden_layer_size+1))));

Theta2 = reshape(nn_params(param1_1:param1_2), hidden_layer_size, (hidden_layer_size+1));

Theta3 = reshape(nn_params( (1 + param1_2):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% size(y)	5000 1
% exit(1)
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
% Part 3: Implement regularization with the cost function and gradients.
%


Y = y;
% size(Theta1)	25 401
% size(Theta2)	10 26

% size(a1)	5000 401

a1 = [ones(m,1) X];

a2 = sigmoid(a1 * Theta1');
a2 = [ones(m,1) a2];
a3 = sigmoid(a2 * Theta2');
a3 = [ones(m,1) a3];

hx = (a3 * Theta3'); % 5000 10


temp = hx - y;
temp = temp .** 2;
J = sum(temp);
J = J / (2*m);


%adding the regularization term 	

temp1 = Theta1(:, 2:end);
temp2 = Theta2(:, 2:end);
temp3 = Theta3(:, 2:end);

temp1 = temp1 .^ 2;
temp2 = temp2 .^ 2;
temp3 = temp3 .^ 2;

J = J + ( (sum(sum(temp1,2),1) + sum(sum(temp2,2),1) + sum(sum(temp3,2),1)) * (lambda/(2*m)));

% 
%	calculating the gradient 
% 


del3 = zeros(num_labels, (hidden_layer_size + 1));
del2 = zeros(hidden_layer_size, (hidden_layer_size + 1));
del1 = zeros(hidden_layer_size, (input_layer_size + 1));

for i = 1:m
	a_1 = [1 X(i,:)];
	% size(a_1)
	z_2 = a_1 * Theta1';
	% size(z_2)
	a_2 = tanh(z_2);
	a_2 = [1 a_2];
	z_3 = a_2 * Theta2';
	a_3 = tanh(z_3);
	a_3 = [1 a_3];
	z_4 = a_3 * Theta3';
	a_4 = (z_4);

	delta_4 = a_4 - Y(i, :);
	% size(delta_3)					1 10
	% size(Theta2)					10 26
	% size(sigmoidGradient(z_2))	1 25
	temp = delta_4 * Theta3;
	% size(temp(2:end))
	delta_3 = temp(2:end)  .* sigmoidGradient(z_3);
	
	temp = delta_3 * Theta2;
	% size(temp(2:end))
	delta_2 = temp(2:end)  .* sigmoidGradient(z_2);
	% delta_2 = delta_2(2:end);

	del3 = del3 + delta_4' * a_3;
	del2 = del2 + delta_3' * a_2;
	del1 = del1 + delta_2' * a_1;
end
Theta1_grad = del1 / m;
Theta2_grad = del2 / m;
Theta3_grad = del3 / m;

Theta3_grad(:, 2:end) = Theta3_grad(:, 2:end) + Theta3(:, 2:end) * lambda / m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2(:, 2:end) * lambda / m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1(:, 2:end) * lambda / m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
