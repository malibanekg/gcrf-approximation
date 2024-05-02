function [block] = vec2block(vector, m, n)
    % column vector
    block = zeros(m,n);
    k = 1;
    for j = 1:n  
        block(:, j) = vector(((k-1)*m + 1):(k*m))';
        k = k + 1;
    end
end