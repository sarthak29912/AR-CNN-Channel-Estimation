clear; clc; close all;

%% 1. Parameters
fd = 50;
Ts = 1e-3;
fn = fd * Ts;

Pmax = 12;
P_start = 8;
Qvals = [8 16 24];
epsilon = 0.1;          
Nreal = 5000;

training_len = 100;
total_len = training_len + Pmax;

%% 2. ACF
maxLag = max(total_len-1, Pmax + max(Qvals)); 
lags = 0:maxLag;
R = besselj(0, 2*pi*fn*lags).';   % column vector

%%  correlated channel
C = toeplitz(R(1:total_len));
L = chol(C + 1e-12*eye(total_len), 'lower');

w = (randn(total_len, Nreal) + 1j*randn(total_len, Nreal))/sqrt(2);
h_ref = (L*w).';   % Nreal x total_len

%% 4. Storage
P_range = P_start:Pmax;
NMSE_results = zeros(length(Qvals), length(P_range));

%% 5. Loop over Q
for qi = 1:length(Qvals)
    Q = Qvals(qi);

    % Correct diagonal loading:
    % modify R[0], then build Toeplitz matrix
    rhat = R;
    rhat(1) = rhat(1) + epsilon;
    Rm = toeplitz(rhat(1:Q));

    fprintf('Q = %d, rcond(Rm) = %.3e\n', Q, rcond(Rm));

    % Precompute direct p-step predictor coefficients
    Cpred = zeros(Q, Pmax);
    for p = 1:Pmax
        wp = R(p+1 : p+Q);     % [R[p], R[p+1], ..., R[p+Q-1]]
        Cpred(:,p) = Rm \ wp;
    end

    errPow = zeros(1, length(P_range));
    sigPow = zeros(1, length(P_range));

    for r = 1:Nreal
        x = h_ref(r,:);

        past = x(1:training_len);
        future = x(training_len+1 : training_len+Pmax);

        % z = [h[n], h[n-1], ..., h[n-Q+1]]
        z = flip(past(end-Q+1:end)).';

        for p = 1:Pmax
            xhat = Cpred(:,p)' * z;

            if p >= P_start
                idx = p - P_start + 1;
                errPow(idx) = errPow(idx) + abs(xhat - future(p))^2;
                sigPow(idx) = sigPow(idx) + abs(future(p))^2;
            end
        end
    end

    NMSE_results(qi,:) = 10*log10(errPow ./ sigPow);
end

%% 6. Plot
figure('Color','w','Position',[100 100 800 500]);
hold on; grid on; box on;

plot(P_range, NMSE_results(1,:), '-rd', 'LineWidth',1.8,'MarkerFaceColor','r','MarkerSize',8);
plot(P_range, NMSE_results(2,:), '-ks', 'LineWidth',1.8,'MarkerFaceColor','k','MarkerSize',8);
plot(P_range, NMSE_results(3,:), '-bo', 'LineWidth',1.8,'MarkerFaceColor','b','MarkerSize',8);

xlabel('Prediction Interval (P)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('NMSE (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title('Direct p-step Prediction', 'FontSize', 13);
legend('Q=8','Q=16','Q=24','Location','SouthEast');
set(gca, 'FontSize', 11);
hold off;
