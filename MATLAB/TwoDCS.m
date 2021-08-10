% Example 2DCS code for a 2-dipole system
% This will use the contemporary (2017) standard of double-sided Feynman 
% diagrams to compute the expected two-dimensional coherence spectroscopy (2DCS)
% output of a 2-dipole system, using Lindblad decoherence, and the tetradic 
% version of the ladder operators.

% Written in one long script for reduced overhead

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variables
% Hamiltonian
omega1 = 100;
omega2 = 100;
delta = 10; % Small delta separates out peaks and makes it look cooler
coupling = -10;

% Decoherence
gammaZ = 0.1;
gammaX = 0.1;

% Light polarisation (theoretically can do polarisation on each pulse)
Ev = [1 1 0];

% Frequency sweeps (good candidate for GPU arrayfun)
w1Start = -300;
w1End = 300;
w3Start = 0;
w3End = 300;
wResolution = 100;

% Waiting time (could be looped)
t2 = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants
hbar = 1;

% Define basic Pauli spin operations
I = eye(2);
z = [1 0;0 -1];
x = [0 1;1 0];
y = [0 -1i;1i 0];
create = [0 1; 0 0];
annihilate = create.';

% Density operator initial state (ground state is typical for 2DCS)
rho = kron(annihilate * create, annihilate * create);

% Define dipole-level operations
zQ1 = kron(z, I);
zQ2 = kron(I, z);
xQ1 = kron(x, I);
xQ2 = kron(I, x);
yQ1 = kron(y, I);
yQ2 = kron(I, y);

% Define Hamiltonian and eigenbasis
H = (omega1 + delta) * zQ1 + (omega2 - delta) * zQ2 + coupling * (xQ1 * xQ2 + yQ1 * yQ2);
[V, eV] = eig(H);
NH = size(H);

% Decoherence operators
II = eye(size(H, 1));
zL = zQ1 + zQ2; % Decoherence
xL = xQ1 + xQ2; % Relaxation

% Decoherence channels (put onto GPU for arrayfun)
L = -1i * (kron(II, H) - kron(H.', II));
L = L + gammaZ * (kron(conj(zL), zL) - 1/2 * (kron(II, zL' * zL) + kron((zL' * zL).', II)));
L = L + gammaX * (kron(conj(xL), xL) - 1/2 * (kron(II, xL' * xL) + kron((xL' * xL).', II)));
NL = size(L, 1);
L = sparse(L);

% Dipole operators
muQ1 = [1;0;0];
muQ2 = [0;1;0];

% Optical creation and annihilation operators
muQ1Create = Ev * muQ1 * kron(create, I);
muQ1Annihilate = muQ1Create';
muQ2Create = Ev * muQ2 * kron(I, create);
muQ2Annihilate = muQ2Create';

% Tetradic form of opetical creation and annihilation operators
muQ1CreateL = kron(II, muQ1Create);
muQ1CreateR = kron(muQ1Annihilate, II);
muQ1AnnihilateL = kron(II, muQ1Annihilate);
muQ1AnnihilateR = kron(muQ1Create, II);
muQ2CreateL = kron(II, muQ2Create);
muQ2CreateR = kron(muQ2Annihilate, II);
muQ2AnnihilateL = kron(II, muQ2Annihilate);
muQ2AnnihilateR = kron(muQ2Create, II);

% Total tetradic dipole operators (put onto GPU for arrayfun)
muTotalCreateR = muQ1CreateR + muQ2CreateR;
muTotalCreateL = muQ1CreateL + muQ2CreateL;
muTotalAnnihilateR = muQ1AnnihilateR + muQ2AnnihilateR;
muTotalAnnihilateL = muQ1AnnihilateL + muQ2AnnihilateL;

% Loop vectors
w1Vector = linspace(w1Start, w1End, wResolution);
w3Vector = linspace(w3Start, w3End, wResolution);

% Pre-allocate output matrices
% Rephasing pathways
GSB = zeros(wResolution, wResolution);
SE = zeros(wResolution, wResolution);
ESA = zeros(wResolution, wResolution);
% Non-rephasing pathways
nrGSB = zeros(wResolution, wResolution);
nrSE = zeros(wResolution, wResolution);
nrESA = zeros(wResolution, wResolution);
% 2-quantum pathways
QQ = zeros(wResolution, wResolution);
QQ2 = zeros(wResolution, wResolution);

% Initial state
rho = rho / trace(rho);
rho = reshape(rho, size(rho, 1)^2, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculations (candidate to be remade into function for use with arrayfun)

% Calculate greens function for t2 (could be looped)
G2 = sparse(expm(L * t2)); 

for w1Count = 1:wResolution
  w1 = w1Vector(w1Count);
  
  % Fourier Transformed Greens function for the coherence time.
  Gw1 = (L + speye(NL) * 1i * w1) \ speye(NL);
  for w3Count = 1:wResolution
    w3 = w3Vector(w3Count);
    
    % Fourier Transformed Greens function for the signal time.
    Gw3 = -(L + speye(NL) * 1i * w3) \ speye(NL);
    
    % Calculate the rephasing Feynman pathways (R1, R2, R3 in thesis)
    rightSide = Gw1 * muTotalAnnihilateR * rho;
    leftSide = muTotalAnnihilateL * Gw3;
    GSB(w3Count, w1Count) = trace(reshape( leftSide * (muTotalCreateL * (G2 * (muTotalCreateR * rightSide))), NH));
    ESA(w3Count, w1Count) = trace(reshape( leftSide * (muTotalCreateL * (G2 * (muTotalCreateL * rightSide))), NH));
    SE(w3Count, w1Count) = trace(reshape( leftSide * (muTotalCreateR * (G2 * (muTotalCreateL * rightSide))), NH));
    
    % Calculate the non-rephasing Feynman pathways (R5, R6, R7 in thesis)
    nrRightSide = Gw1 * muTotalCreateL * rho;
    nrESA(w3Count, w1Count) = trace(reshape( leftSide * (muTotalCreateL * (G2 * (muTotalAnnihilateR * nrRightSide))), NH));
    nrGSB(w3Count, w1Count) = trace(reshape( leftSide * (muTotalCreateL * (G2 * (muTotalAnnihilateL * nrRightSide))), NH));
    nrSE(w3Count, w1Count) = trace(reshape( leftSide * (muTotalCreateR * (G2 * (muTotalAnnihilateR * nrRightSide))), NH));
    
    % Calculate the two-quantum Feynman pathways (R4, R8 in thesis)
    QQ(w3Count, w1Count) = trace(reshape( leftSide * (muTotalAnnihilateL * (G2 * (muTotalCreateL * nrRightSide))), NH));
    QQ2(w3Count, w1Count) = trace(reshape( leftSide * (muTotalAnnihilateR * (G2 * (muTotalCreateL * nrRightSide))), NH));
  end
end

% NOTE: be careful of pcolor. I don't know if Octave and MATLAB treat the matrix
% the same way. I had to switch what I thought was correct in the indexing
% during the for loop to get this to work how I expected it to work.

figure(1)
% -w1 because reasons ¯\_(?)_/¯
[x, y] = meshgrid(w1Vector, w3Vector);
pcolor(-x, y, abs(GSB + SE - ESA));
title("Rephasing pathways")
xlabel("w1 (arb. frequency)")
ylabel("w2 (arb. frequency)")
shading flat

figure(2)
pcolor(x, y, abs(nrGSB + nrSE - nrESA));
title("Non-rephasing pathways")
xlabel("w1 (arb. frequency)")
ylabel("w2 (arb. frequency)")
shading flat

figure(3)
pcolor(x, y, abs(QQ - QQ2));
title("Double quantum pathways")
xlabel("w1 (arb. frequency)")
ylabel('w2 (arb. frequency)')
shading flat