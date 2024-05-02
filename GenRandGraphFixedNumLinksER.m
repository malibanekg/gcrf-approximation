function [S] = GenRandGraphFixedNumLinksER(n, m)

if m >= n-1
    connected = 0;
    while connected == 0
        addpath('Data_generation')
        %Erdos-Reny graph;
        rate = (2*m) / (n * (n-1));
        S = generate_random_graph(n, 'er', -1, rate);
        S = S - diag(diag(S));
        m1 = sum(sum(S))/2;

        if m1 > m
            num_zeros = m1 - m;
            r = randsample(m1, num_zeros); 
            r = sort(r);

            k = 0; 
            l = 1; 
            for i = 1:n-1
                for j = i+1:n 
                    if (S(i,j) == 1)
                      k = k + 1;
                      if (l <= num_zeros) && (k == r(l)) 
                         S(i,j) = 0;
                         S(j,i) = 0;
                         l = l + 1;
                      end
                    end
                end
            end
        elseif m1 < m
            num_nonzeros = m - m1;
            r = randsample(n * (n-1)/2 - m1, num_nonzeros); 
            r = sort(r);       

            k = 0; 
            l = 1; 
            for i = 1:n-1
                for j = i+1:n 
                    if (S(i,j) == 0)
                      k = k + 1;
                      if (l <= num_nonzeros) && (k == r(l)) 
                         S(i,j) = 1;
                         S(j,i) = 1;
                         l = l + 1;
                      end
                    end
                end
            end
        end
        
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
 