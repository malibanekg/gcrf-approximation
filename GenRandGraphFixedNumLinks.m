function [S] = GenRandGraphFixedNumLinks(n, m)

if m >= n-1
    
 connected = 0;
 while connected == 0
    num_zeros = n*(n-1)/2 -m;

    r = randsample(n*(n-1)/2,num_zeros); 
    r = sort(r);

    S = zeros(n,n);
    k = 0;
    l = 1;
    for i = 1:n-1
         for j = i+1:n
             k = k + 1;
             if (l <= num_zeros) && (k == r(l))
                 S(i,j) = 0;
                 l = l + 1;
             else
                 S(i,j) = 1; % 10 * rand();
             end
         end
    end

    S = S + S';

    %connected check
    degree = sum(S,2);
    DS = diag(degree);
    LS = DS - S;
    LS_SPECTRUM = eig(LS);

    % There were eigenvalues with value -0.00000000000000001
    indices = abs(LS_SPECTRUM) < 1e-12;
    LS_SPECTRUM(indices) = 0;

    connected = sum(LS_SPECTRUM==0)==1;
 end
 
else
 disp('Graph is never connected');
 S = [];
end
end
 
 
 
  

 
 