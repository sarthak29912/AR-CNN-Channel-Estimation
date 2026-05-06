% --- generate_channel_data.m ---
clear; clc;

%% 1. Define System Parameters based on the paper
N = 12;            % Number of Base Station Antennas [cite: 284]
M = 225;           % Number of RIS Elements [cite: 284]
V = 10;            % Number of coherence intervals for training phase
Q = 16;            % AR model order (number of coefficients to predict)
fd = 50;           % Doppler frequency in Hz (User mobility) [cite: 287]
Ts = 1e-3;         % Sampling time / Coherence interval duration (example)
fn = fd * Ts;      % Normalized Doppler frequency [cite: 144]
num_samples = 1000; % Let's start with 1,000 instead of 70,000 to test your PC's memory!

fprintf('Initializing Data Generation...\n');

%% 2. Preallocate Memory for the Dataset
% X_train will hold the stacked Real/Imaginary channel estimates [cite: 183, 184]
% Size: (2*N) rows x M columns x V depth x num_samples 
X_train = zeros(2*N, M, V, num_samples); 

% Y_train will hold the target AR coefficients for the neural network to learn
Y_train = zeros(Q, num_samples);

%% 3. Generate the Target AR Coefficients (Levinson-Durbin Recursion)
% The ACF is governed by the zeroth-order Bessel function [cite: 144]
R = zeros(Q+1, 1);
for q_lag = 0:Q
    R(q_lag+1) = besselj(0, 2 * pi * fn * q_lag); 
end
R(1) = R(1) + 1e-3; % Add small epsilon to diagonal for stability [cite: 169, 172]

% Construct Toeplitz matrix and solve for AR coefficients [cite: 153, 155]
chi = toeplitz(R(1:Q));
w = R(2:Q+1);
true_AR_coeffs = - (chi \ w); 

%% 4. Simulate the Channel Data over V intervals
for s = 1:num_samples
    % Simulate complex channel estimates for V intervals
    % (In a full simulation, this includes BS-UE and RIS-UE paths [cite: 77, 80])
    % Here we generate standard complex Gaussian fading as a placeholder
    H_complex = randn(N, M, V) + 1i * randn(N, M, V); 
    
    % Preprocessing: Split into Real and Imaginary parts [cite: 184]
    for v = 1:V
        real_part = real(H_complex(:, :, v));
        imag_part = imag(H_complex(:, :, v));
        
        % Stack them vertically: size becomes (2*N) x M
        X_train(:, :, v, s) = [real_part; imag_part]; 
    end
    
    % Assign the target label (the network needs to predict these coefficients)
    Y_train(:, s) = true_AR_coeffs;
end

fprintf('Data Generation Complete! X_train size: %s\n', mat2str(size(X_train)));
save('training_data.mat', 'X_train', 'Y_train', '-v7.3');
fprintf('Data saved to training_data.mat\n');