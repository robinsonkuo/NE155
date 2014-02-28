%hw3#3
% Set A to be the nxn matrix, where n is the number of unknowns.
% Example A, b and n are shown (Ax = b), replace as applicable:
n = 5; % n unknowns
A = zeros(n,n);
A(1,1) = 4; A(1,2) = -1;
for i = 2:n-1
    A(i,i-1) = -1;
    A(i,i) = 4;
    A(i,i+1) = -1;
end
A(n,n-1) = -1; A(n,n) = 4;
b = ones(n,1) * 100;
D = diag(diag(A)); % diagonal matrix of A

% The above definitions will be used for all methods below
% THE ABOVE MUST BE RUN FOR ANY OF THE SUBSEQUENT PARTS TO WORK (i.e. A,
% b,n and D must be defined)
%%
e = 10^-8;
%% Jacobi Iteration
xprev = zeros(n,1); % The initial guess for the x vector
xcurr = D^-1*(D-A)*xprev+D^-1*b; % The first iteration
iterations = 1;
crit = e;
while crit >= e % Stopping criterion
    xprev = xcurr;
    xcurr = D^-1*(D-A)*xprev+D^-1*b;
    iterations = iterations + 1;
    crit = norm(xcurr-xprev)/norm(xcurr);
end
display(xcurr); % displays the solution vector for Jacobi
display(iterations); % displays the number of iterations required

%% Gauss Seidel
[L U] = lu(A); % Lower and Upper triangular matrices of A
GSx = zeros(n,1); % The initial guess for the x vector
GSiterations = 0;
GScrit = e;
while GScrit >= e % Stopping criterion
    GSiterations = GSiterations + 1;
    GSxprev = GSx;
    for g = 1:n
        summation = 0;
        for gprime = 1:n
            if gprime ~= g
                summation = summation + A(g,gprime)*GSx(gprime);
            end
        end
        GSx(g) = (b(g) - summation)/A(g,g);
    end
    GScrit = norm(GSx-GSxprev)/norm(GSx);
end
display(GSx); % displays the solution vector for GS
display(GSiterations); % displays the number of iterations required for GS

%% Successive Over Relaxation
% Select an omega value (denoted here by w)
w = 1.066;

[L U] = lu(A); % Lower and Upper triangular matrices of A
SORx = zeros(n,1); % The initial guess for the x vector
SORiterations = 0;
SORcrit = e;
while SORcrit >= e % Stopping criterion
    SORiterations = SORiterations + 1;
    SORxlast = SORx; % stores the previous vector in the iteration
    for g = 1:n
        summation = 0;
        for gprime = 1:n
            if gprime ~= g
                summation = summation + A(g,gprime)*SORx(gprime);
            end
        end
        SORx(g) = (1-w)*SORxlast(g) + w*(b(g) - summation)/A(g,g);
    end
    SORcrit = norm(SORx-SORxlast)/norm(SORx);
end
display(SORx); % displays the solution vector for SOR
display(SORiterations); % displays the number of iterations required for SOR