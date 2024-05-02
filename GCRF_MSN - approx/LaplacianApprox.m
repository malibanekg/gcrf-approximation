% G and H are similarity matrices
% Laplacian matrices should be created
% --------------------------------------------------
function [Vectors, Spectrum] = LaplacianApprox(S1, S2)

[n1,m1] = size(S1);
[n2,m2] = size(S2);

% create Laplacian matrix LG
dijagonala = sum(S1,2);
DG = diag(dijagonala);
LG = DG - S1;

% create Laplacian matrix LH
dijagonala = sum(S2,2);
DH = diag(dijagonala);
LH = DH - S2;

%% Laplacian of the Kronecker product
% K = kron(DG, DH) - kron(S1, S2);
% [K_VECTORS, K_SPECTRUM] = eig(K);
% K_SPECTRUM = diag(K_SPECTRUM);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Tensor Laplacian Approximation %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Laplacian spectrum of G and H
[G_VECTORS, G_SPECTRUM] = eig(LG);
[H_VECTORS, H_SPECTRUM] = eig(LH);
G_SPECTRUM = diag(G_SPECTRUM);
% There were eigenvalues with value -0.00000000000000001
indices = abs(G_SPECTRUM) < 1e-12;
G_SPECTRUM(indices) = 0;
% ------------------------------------------------------
H_SPECTRUM = diag(H_SPECTRUM);
% There were eigenvalues with value -0.00000000000000001
indices = abs(H_SPECTRUM) < 1e-12;
H_SPECTRUM(indices) = 0;

% Sorted spectrums
% DGs = sort(diag(LG));
% DHs = sort(diag(LH));
DGs = sort(sum(S1,2));
DHs = sort(sum(S2,2));

n = n1 * n2;
mainSpectrum = zeros(n,1);

k = 0;
for i = 1:n1
    for j = 1:n2
        k = k + 1;
        mainSpectrum(k) = G_SPECTRUM(i) * DHs(j) + DGs(i) * H_SPECTRUM(j) - G_SPECTRUM(i) * H_SPECTRUM(j); 
    end
end

% [Spectrum, Indices] = sort(mainSpectrum);
% Spectrum = diag(Spectrum);
Spectrum = diag(mainSpectrum);
Vectors = kron(G_VECTORS, H_VECTORS);
%Vectors = Vectors(:, Indices);

% Vecc = zeros(n,n);
% % Popravka jebena
% for i = 1:n
%     % t = Vectors(:,i);
%     [Vecc(:,i),flag] = pcg(K, mainSpectrum(i) * Vectors(:,i), 1e-12, 300,[], [], Vectors(:,i));
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Cartesian product for test %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IG = eye(n1);
% IH = eye(n2);

% Catesian product = CARTESIAN PRODUCT OD DEGREE - CARTESIAN PRODUCT OF ADJACENCY
% C = kron(LG, IH) + kron(IG, LH);
% [C_VECTORS, C_SPECTRUM] = eig(C);
% C_SPECTRUM = diag(C_SPECTRUM);

end





