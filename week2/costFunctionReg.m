function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
J = -(y'*log(h)+(1-y)'*log(1-h))/m;

% =======================without loop===========================
theta(1) = 0; %theta is local variable
regcost = (sum(theta.^2)); %theta'*theta 
grad = ((X'*(h-y)) ./ m) + (lambda/m)*theta;
% ==============================================================

%{ =======================using loop============================
regcost = 0;
for i=2:size(theta)
  regcost += (theta(i)^2);
end 

grad(1) = (X(:,1)'*(h-y)) ./ m;
for i=2:size(theta)
  grad(i) = ((X(:,i)'*(h-y)) ./ m)+ (lambda/m)*theta(i);
end
%}
%===============================================================

J += regcost*lambda/(2*m);

% =============================================================

end
