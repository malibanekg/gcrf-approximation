function [original] = initialMatrix(matrix, m1, n1, m2, n2)
    original = zeros(m1*m2, n1*n2);
    % block = zeros(m2, n2);
    k = 1;
    for j = 1:n1
        for i = 1:m1         
           block = vec2block(matrix(k,:), m2, n2);
           original((i-1)*m2 + 1:i * m2, (j-1)*n2 + 1:j*n2) = block;
           k = k+1;
        end
    end
end