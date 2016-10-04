function plotData(X, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.


figure; % open a new figure window

plot(X, y, 'rx', 'MarkerSize', 10);
ylabel('Error');
xlabel('Test Example number');




% ============================================================

end
