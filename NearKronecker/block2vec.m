function [vector] = block2vec(block)
    [n,m] = size(block);
    vector = zeros(n*m,1);
    k = 0;
    for j = 1:m
        for i = 1:n
            k = k + 1;
            vector(k,1) = block(i,j);
        end
    end
end