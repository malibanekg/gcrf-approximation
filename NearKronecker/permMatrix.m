function [permutation] = permMatrix(matrix, m1, n1, m2, n2)

% vector size
% v_s = zeros(n2 * m2,1);

% number of blocks // svaki vektor postaje vrsta
% broj vrsta je broj blokova = m1 * n1
% svaki vektor ima m2 * n2 koordinata
permutation = zeros(m1 * n1, m2 * n2);
k = 0;
for i = 1:m1
    for j = 1:n1
        k = k + 1;
        Aij = matrix((i-1)*m2 + 1:i * m2, (j-1)*n2 + 1:j*n2);
        vec = block2vec(Aij);
        permutation(k,:) = vec';
    end
end

end