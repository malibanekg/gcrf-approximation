% G and H are similarity matrices
% Laplacian matrices should be created
% --------------------------------------------------
function [Vectors, mainSpectrum3] = LaplacianApprox2(S1, S2)

[n1,n1] = size(S1);
[n2,n2] = size(S2);

% create Laplacian matrix LG
degree1 = sum(S1,2);
DG = diag(degree1);
% LG = DG - S1;

% create Laplacian matrix LH
degree2 = sum(S2,2);
DH = diag(degree2);
% LH = DH - S2;

% create normalized Laplacian matrix Lambda of the Kronecker product of G
% and H

S1prim = DG^(-1/2) * S1 * DG^(-1/2);
S2prim = DH^(-1/2) * S2 * DH^(-1/2);

% % IT WORKS!!! Original, expensive
% LambdaGH = kron(eye(n1), eye(n2)) - kron(S1prim, S2prim);
% [Vectors, EigsLambdaGH] = eig(LambdaGH);
% D = kron(DG,DH);
% mainSpectrum3 = diag(D * diag(EigsLambdaGH));

% H0 heuristic
[Vec1, Spec1] = eig(S1prim);
[Vec2, Spec2] = eig(S2prim);

EigsLambdaGH = kron(eye(n1), eye(n2)) - kron(Spec1, Spec2);
mainSpectrum3 = kron(DG,DH) * EigsLambdaGH;
Vectors = kron(Vec1, Vec2);

% % H1 heuristic
% LambdaGH = kron(eye(n1), eye(n2)) - kron(S1prim, S2prim);
% [Vectors, EigsLambdaGH] = eig(LambdaGH);
% SortKronDGDH = diag(sort(diag(kron(DG,DH))));
% mainSpectrum3 = SortKronDGDH * EigsLambdaGH;

% % test!
% [Vectors1, spec1] = eig(S1prim);
% [Vectors2, spec2] = eig(S2prim);
% D = kron(DG,DH);
% EigsLambdaGH = kron(eye(n1), eye(n2)) - kron(spec1, spec2);
% mainSpectrum3 = diag(D * diag(EigsLambdaGH));
% Vectors = kron(Vectors1, Vectors2);

% -------------------------------------------------------------------------
% D = diag(sum(LambdaGH,2));
% Dpom = sort(diag(D));
% D = diag(Dpom);
% Vectors = D^(1/2) * Vectors;

% Later work
% [Vectors1, spec1] = eig(S1prim);
% [Vectors2, spec2] = eig(S2prim);
% D = kron(DG,DH);
% 
% EigsLambdaGH = kron(eye(n1), eye(n2)) - kron(spec1, spec2);
% [EigsLambdaGH, indVectors] = sort(diag(EigsLambdaGH));
% EigsLambdaGH = diag(EigsLambdaGH);
% 
% mainSpectrum3 = D * diag(EigsLambdaGH);
% mainSpectrum3 = diag(mainSpectrum3);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Tensor Laplacian Approximation %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Laplacian spectrum of G and H
% [G_VECTORS, G_SPECTRUM] = eig(LG);
% [H_VECTORS, H_SPECTRUM] = eig(LH);
% % Ss= kron(G_SPECTRUM, H_SPECTRUM);
% % [Ss, indVectors] = sort(diag(Ss));
% Dpom = sort(diag(D));
% D = diag(Dpom);
% Vectors = kron(G_VECTORS, H_VECTORS);

% Vectors = kron(Vectors1, Vectors2);
% Vectors = D^(1/2) * Vectors(:, indVectors);

% n = n1 * n2;
% S = kron(S1, S2);
% D = diag(sum(S,2));
% [Vectors, mainSpectrum4] = eig(eye(n) - D^(-1/2)*S*D^(-1/2));

% S = kron(S1,S2);
% % S = S/sum(sum(S));
% Deg = sum(S);
% Deg = diag(Deg);
% Lap = Deg - S;
% [Vectors, mainSpectrum44] = eig(Lap);

end