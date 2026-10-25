%% Diagonalization 6x6

syms D tAA tAB UA UB eta real

H = [0,       0,       tAB,    -tAB,    0,       0,       tAA,    tAA,    0;
     0,       0,       0,       0,       tAB,    -tAB,   -tAA,   -tAA,   0;
     tAB,     0,       D,       0,       0,      -tAA,   -tAB,    0,     -tAB;
     -tAB,     0,       0,       D,      -tAA,    0,       0,      tAB,    tAB;
     0,       tAB,     0,      -tAA,     D,       0,       tAB,    0,      tAB;
     0,      -tAB,    -tAA,     0,       0,       D,       0,     -tAB,   -tAB;
     tAA,    -tAA,    -tAB,     0,       tAB,     0,       UA,     0,      0;
     tAA,    -tAA,     0,       tAB,     0,      -tAB,     0,      UA,     0;
     0,       0,      -tAB,     tAB,     tAB,    -tAB,     0,      0,      UB + 2*D];

H6 = H(1:6, 1:6);
basis = eye(6);

GA = (basis(:, 1) - basis(:, 2)) / sqrt(2);
G1 = (basis(:, 3) - basis(:, 5)) / sqrt(2);
G2 = (basis(:, 4) - basis(:, 6)) / sqrt(2);
Gp = (G1 + G2) / sqrt(2);
Gm = (G1 - G2) / sqrt(2);

TA = (basis(:, 1) + basis(:, 2)) / sqrt(2);
T1 = (basis(:, 3) + basis(:, 5)) / sqrt(2);
T2 = (basis(:, 4) + basis(:, 6)) / sqrt(2);
Tp = (T1 + T2) / sqrt(2);
Tm = (T1 - T2) / sqrt(2);

basis3  = eye(3);
H6p = [TA' * H6 * TA, TA' * H6 * T1, TA' * H6 * T2; 
       T1' * H6 * TA, T1' * H6 * T1, T1' * H6 * T2;
       T2' * H6 * TA, T2' * H6 * T1, T2' * H6 * T2];
H6m = [GA' * H6 * GA, GA' * H6 * G1, GA' * H6 * G2; 
       G1' * H6 * GA, G1' * H6 * G1, G1' * H6 * G2;
       G2' * H6 * GA, G2' * H6 * G1, G2' * H6 * G2];

u = (basis3(:, 2) + basis3(:, 3)) / sqrt(2);
v = (basis3(:, 2) - basis3(:, 3)) / sqrt(2);

%%
clc 
clear

syms D tAA tAB UA UB eta real

H = [0,       0,       tAB,    -tAB,    0,       0,       tAA,    tAA,    0;
     0,       0,       0,       0,       tAB,    -tAB,   -tAA,   -tAA,   0;
     tAB,     0,       D,       0,       0,      -tAA,   -tAB,    0,     -tAB;
     -tAB,     0,       0,       D,      -tAA,    0,       0,      tAB,    tAB;
     0,       tAB,     0,      -tAA,     D,       0,       tAB,    0,      tAB;
     0,      -tAB,    -tAA,     0,       0,       D,       0,     -tAB,   -tAB;
     tAA,    -tAA,    -tAB,     0,       tAB,     0,       UA,     0,      0;
     tAA,    -tAA,     0,       tAB,     0,      -tAB,     0,      UA,     0;
     0,       0,      -tAB,     tAB,     tAB,    -tAB,     0,      0,      UB + 2*D];

basis9 = eye(9);

GA = (basis9(:, 1) - basis9(:, 2)) / sqrt(2);
G1 = (basis9(:, 3) - basis9(:, 5)) / sqrt(2);
G2 = (basis9(:, 4) - basis9(:, 6)) / sqrt(2);
Gp = (G1 + G2) / sqrt(2);
Gm = (G1 - G2) / sqrt(2);

TA = (basis9(:, 1) + basis9(:, 2)) / sqrt(2);
T1 = (basis9(:, 3) + basis9(:, 5)) / sqrt(2);
T2 = (basis9(:, 4) + basis9(:, 6)) / sqrt(2);
Tp = (T1 + T2) / sqrt(2);
Tm = (T1 - T2) / sqrt(2);


up = (T1 + T2) / sqrt(2);
um = (G1 + G2) / sqrt(2);
vp = (T1 - T2) / sqrt(2);
vm = (G1 - G2) / sqrt(2); 

basis9(:, )' * H * um

%% Two-electron Hamiltonian (A5)-(A6) and eigenstates of the 6x6 block
% Basis (A5):
%  |1> = c+_{A1dn} c+_{A2up} |0>     |6> = c+_{A1up} c+_{Bdn}  |0>
%  |2> = c+_{A1up} c+_{A2dn} |0>     |7> = c+_{A2up} c+_{A2dn} |0>
%  |3> = c+_{A2up} c+_{Bdn}  |0>     |8> = c+_{A1up} c+_{A1dn} |0>
%  |4> = c+_{A1dn} c+_{Bup}  |0>     |9> = c+_{Bup}  c+_{Bdn}  |0>
%  |5> = c+_{A2dn} c+_{Bup}  |0>
clear; clc;

syms tAB tAA Delta UA UB ea real

%% Full 9x9 Hamiltonian (A6)
H = [ 2*ea,  0,     tAB,        -tAB,       0,          0,          tAA,     tAA,     0;
      0,     2*ea,  0,           0,         tAB,       -tAB,       -tAA,    -tAA,     0;
      tAB,   0,     Delta+2*ea,  0,         0,         -tAA,       -tAB,     0,      -tAB;
     -tAB,   0,     0,           Delta+2*ea,-tAA,       0,          0,       tAB,     tAB;
      0,     tAB,   0,          -tAA,       Delta+2*ea, 0,          tAB,     0,       tAB;
      0,    -tAB,  -tAA,         0,         0,          Delta+2*ea, 0,      -tAB,    -tAB;
      tAA,  -tAA,  -tAB,         0,         tAB,        0,          UA+2*ea, 0,       0;
      tAA,  -tAA,   0,           tAB,       0,         -tAB,        0,       UA+2*ea, 0;
      0,     0,    -tAB,         tAB,       tAB,       -tAB,        0,       0,       UB+2*Delta+2*ea];

assert(isequal(H, H.'), 'H is not symmetric');

H  = subs(H, ea, 0);          % epsilon_a = 0
H6 = H(1:6, 1:6);             % 6x6 block (U -> infinity sector)
disp('Full Hamiltonian (ea = 0):'); disp(H)
disp('6x6 block H6:');          disp(H6)

%% Exchange operator P: |1><->|2>, |3><->|5>, |4><->|6>
I6 = sym(eye(6));
k  = @(n) I6(:, n);
P  = [k(2) k(1) k(5) k(6) k(3) k(4)];
fprintf('[P,H6] = 0 ? %d\n', isequal(simplify(P*H6 - H6*P), sym(zeros(6))));
fprintf('P^2 = 1  ? %d\n\n', isequal(P*P, I6));

%% Eigenstates, sector by sector (eta = +1 triplet, eta = -1 singlet)
Evals  = sym(zeros(6,1));
Evecs  = sym(zeros(6,6));
labels = strings(6,1);
s2 = sqrt(sym(2));
c  = 0;

for eta = [1, -1]
    % symmetry-adapted basis
    k1 = (k(1) + eta*k(2))/s2;
    k2 = (k(3) + eta*k(5))/s2;
    k3 = (k(4) + eta*k(6))/s2;
    Heta = simplify([k1 k2 k3].' * H6 * [k1 k2 k3]);
    fprintf('H_6,eta for eta = %+d  (basis {|1eta>,|2eta>,|3eta>}):\n', eta); disp(Heta)

    % mirror-symmetric / antisymmetric combinations
    U = (k2 + k3)/s2;
    V = (k2 - k3)/s2;

    % (i) |U_eta> : exact eigenstate, E = Delta - eta*tAA
    EU = Delta - eta*tAA;
    c = c + 1;  Evals(c) = EU;  Evecs(:,c) = U;
    labels(c) = sprintf("U   (eta=%+d)", eta);

    % (ii) 2x2 block in {|1eta>, |V_eta>}
    H2 = simplify([k1 V].' * H6 * [k1 V]);
    fprintf('H^(2) for eta = %+d  (basis {|1eta>,|V_eta>}):\n', eta); disp(H2)

    a  = Delta + eta*tAA;
    Ep = (a + sqrt(a^2 + 8*tAB^2))/2;
    Em = (a - sqrt(a^2 + 8*tAB^2))/2;

    for E = [Ep, Em]
        num = s2*tAB*k1 + E*V;                    % unnormalized
        phi = num / sqrt(2*tAB^2 + E^2);          % normalized
        c = c + 1;  Evals(c) = E;  Evecs(:,c) = phi;
        if isequal(E, Ep), pm = "+"; else, pm = "-"; end
        labels(c) = sprintf("phi%s (eta=%+d)", pm, eta);
    end
end

%% Symbolic verification: (H6 - E) |psi> = 0
fprintf('Symbolic check of eigen-equations:\n');
for n = 1:6
    v = Evecs(:,n) * sqrt(simplify(Evecs(:,n).'*Evecs(:,n)));  % strip norm
    r = simplify(expand((H6 - Evals(n)*I6) * v));
    fprintf('  %-16s  residual = 0 ? %d\n', labels(n), isequal(r, sym(zeros(6,1))));
end

%% Display results
fprintf('\nEigenvalues of the 6x6 block:\n');
for n = 1:6
    fprintf('  %-16s E = %s\n', labels(n), char(simplify(Evals(n))));
end
disp('Eigenvectors (columns, in basis |1>..|6>):'); disp(Evecs)

%% Numerical cross-check against eig()
vals = [tAB tAA Delta];  num = [0.7 0.3 1.9];      % arbitrary test values
Hn   = double(subs(H6,    vals, num));
En   = sort(double(subs(Evals, vals, num)));
Vn   = double(subs(Evecs, vals, num));
fprintf('max |E_analytic - eig| = %.2e\n', max(abs(En - sort(eig(Hn)))));
fprintf('max |V''V - I|          = %.2e\n', max(max(abs(Vn.'*Vn - eye(6)))));

%% ======================================================================
%% Hamiltonian in the new basis
%  New basis (columns of W):
%   1..6 : eigenstates of the 6x6 block  {U+, phi+(+), phi-(+), U-, phi+(-), phi-(-)}
%   7..9 : doubly occupied states kept as they are  {|A2 updn>, |A1 updn>, |B updn>}
%  W is real orthogonal, so H_new = W.' * H * W.
W = sym(zeros(9));
W(1:6,1:6) = Evecs;
W(7:9,7:9) = sym(eye(3));
 
% Build H_new block by block (the 6x6 block is diagonal by construction)
C     = simplify(Evecs.' * H(1:6,7:9));   % <psi_n | H | doubly occupied>
H_dd  = H(7:9,7:9);                        % doubly occupied block (unchanged)
H_new = [diag(Evals), C;
         C.',         H_dd];
 
newLabels = [labels; "A2 updn"; "A1 updn"; "B updn"];
disp('New basis ordering:'); disp(newLabels)
disp('Hamiltonian in the new basis:'); disp(H_new)
disp('Couplings <psi_n|H|D> (rows: eigenstates, cols: |7>,|8>,|9>):'); disp(C)
 
% The triplet states (eta=+1) decouple from all doubly occupied states
fprintf('Triplets decoupled from doubly occupied ? %d\n', ...
        isequal(C(1:3,:), sym(zeros(3))));
 
% Numerical check that H_new really equals W.'*H*W
pars = [tAB tAA Delta UA UB];  pn = [0.7 0.3 1.9 5.0 4.0];
Hn_direct = double(subs(W.'*H*W, pars, pn));
Hn_block  = double(subs(H_new,   pars, pn));
fprintf('max |W''HW - H_new| = %.2e\n', max(abs(Hn_direct(:) - Hn_block(:))));
fprintf('spectrum preserved: max |eig diff| = %.2e\n', ...
        max(abs(sort(eig(Hn_block)) - sort(eig(double(subs(H,pars,pn)))))));