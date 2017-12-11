% mu: 2x1 matrix
% Sigma: 2x2 matrix
% phi: a number

% ???????????????????????
mu0 = [0;0];
Sigma0 = [1,0;0,1];
mu1 = [5;5];
Sigma1 = [1,0;0,1];
phi = 0.5;
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line', 1);
% 
mu0 = [0;1];
Sigma0 = [1,0;0,1];
mu1 = [1;0];
Sigma1 = [1,0;0,1];
phi = 0.9;
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line (one side)', 2);
% 
mu0 = [0;0];
Sigma0 = [3,0;0,3];
mu1 = [5;5];
Sigma1 = [5,4.5;4.5,5];
phi = 0.5;
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Parabalic', 3);
% % 
mu0 = [1;1];
Sigma0 = [3,0;0,3];
mu1 = [1;1];
Sigma1 = [5,4.5;4.5,5];
phi = 0.5;
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Hyperbola', 4);
% 
mu0 = [0;0];
Sigma0 = [5,4.9;4.9,5];
mu1 = [0;0];
Sigma1 = [10,0;0,10];
phi = 0.5;
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Two parallel lines.', 5);
% % 

%
mu0 = [1;1];
Sigma0 = [7,0;0,7];
mu1 = [1;1];
Sigma1 = [3.5,0;0,3.5];
phi = 0.5;
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Circle', 6);
% 
%
mu0 = [1;1];
Sigma0 = [3,0;0,6];
mu1 = [1;1];
Sigma1 = [1,0;0,3];
phi = 0.5;
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Ellipsoid', 7);
% 

%
mu0 = [1;1];
Sigma0 = [8,1;1,8];
mu1 = [1;1];
Sigma1 = [8,1;1,8];
phi = 0.5;
plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'No boundary', 8);
