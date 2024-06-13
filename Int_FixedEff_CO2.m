% Define the file path
data = readtable("C:\Users\user\Desktop\Bachelor Thesis\After Allocation\My Code\Extension\Air Pollution Data New.xlsx");
data = table2array(data);
Y = data(:,3);
X = data(:,4:11);
T = 20;
K = 8;
N = size(Y,1)/T;
G = 3;
conv = 10^(-10);
group_alloc_first_it = randi(G, N, 1); % initial random group allocation
countries_per_group_first = histcounts(group_alloc_first_it, 1:G+1);
r = 3; % number of factors
Lambda_IFE = zeros(N,r);
MC_sim = 1000;
MC_IFE_thetas = zeros(MC_sim,K);

% Check for empty groups
for value = 1:G
    if countries_per_group_first(value) == 0
        group_alloc_first_it(randi([1,21])) = value;
        countries_per_group_first(value) = 1;
    end
end

% Create lagged variables
X_lagged = nan(N*T, K);
for country = 1:N
    for t = 2:T
        X_lagged((country-1)*T+t, :) = X((country-1)*T+t-1, :);
    end
end

% Remove rows with NaN values (first period for each country)
valid_rows = ~isnan(X_lagged(:, 1));
X_lagged = X_lagged(valid_rows, :);
Y_lagged = Y(valid_rows);

% Adjust the number of periods and declare F
T = T - 1;
F_first_it = randn(T,r);


%% Step 1 of Algorithm 3: Compute thetas
proj_matrix = eye(T)-F_first_it*F_first_it'/T; % Construct the projecion matrix
var1 = 0;
var2 = 0;
for i = 1:N
    X_port = X_lagged((i-1)*T+1:i*T,:);
    Y_port = Y_lagged((i-1)*T+1:i*T);
    var1 = var1 + X_port'*proj_matrix*X_port;
    var2 = var2 + X_port'*proj_matrix*Y_port;
end
thetas_opt_first_it = var1\var2;

%% Step 2 of Algorithm 3: Compute F's
U = Y_lagged - X_lagged * thetas_opt_first_it;
residuals = reshape(U,T,N)';
[V,D] = eig(residuals'*residuals); % This is the covariance matrix
[~,eigenvalues] = sort(diag(D), 'descend');
F_first_it = V(:,eigenvalues(1:r));

%% Step 3 of Algorithm 3: Computes λ's
Lambda_first_it = residuals*F_first_it/T;

%% Step 4 of Algorithm 3: Compute the new group allocation
group_class_intermediate = zeros(N, G);
for country = 1:N
    y = Y_lagged((country-1)*T+1:country*T); % Selects the data related to the dependent variable for each period of each country
    x = X_lagged((country-1)*T+1:country*T, :); % Selects the data related to the independent variables for each period of each country
    for group = 1:G
        u = 0.0;
        for period = 1:T
            u = u + (y(period) - x(period,:) * thetas_opt_first_it - Lambda_first_it(country,group) * F_first_it(period,group)')^2; % Step 2 of Algorithm 1
        end
        group_class_intermediate(country, group) = u;
    end
end

% Group classification
[group_class, new_group_alloc] = min(group_class_intermediate, [], 2);

group_alloc_first_it = new_group_alloc;
countries_per_group_first = histcounts(group_alloc_first_it, 1:G+1);

delta = 1;
while delta > 10^(-10)
    %% Step 1 of Algorithm 3: Compute thetas
    proj_matrix=eye(T)-F_first_it*F_first_it'/T; % Construct the projecion matrix
    var1 = 0;
    var2 = 0;
    for i = 1:N
        X_port = X_lagged((i-1)*T+1:i*T,:);
        Y_port = Y_lagged((i-1)*T+1:i*T);
        var1 = var1 + X_port'*proj_matrix*X_port;
        var2 = var2 + X_port'*proj_matrix*Y_port;
    end
    thetas = var1\var2;

    %% Step 2 of Algorithm 3: Compute F's
    U = Y_lagged - X_lagged * thetas;
    residuals = reshape(U,T,N)';
    [V,D] = eig(residuals'*residuals); % This is the covariance matrix
    [~,eigenvalues] = sort(diag(D), 'descend');
    F_IFE = V(:,eigenvalues(1:r));

    %% Step 3 of Algorithm 3: Computes λ's
    Lambda_IFE = residuals*F_IFE/T;

    %% Step 4 of Algorithm 3: Compute the new group allocation
    group_class_intermediate = zeros(N, G);
    for country = 1:N
        y = Y_lagged((country-1)*T+1:country*T); % Selects the data related to the dependent variable for each period of each country
        x = X_lagged((country-1)*T+1:country*T, :); % Selects the data related to the independent variables for each period of each country
        for group = 1:G
            u = 0.0;
            for period = 1:T
                u = u + (y(period) - x(period,:) * thetas - Lambda_IFE(country,group) * F_IFE(period,group)')^2; % Step 2 of Algorithm 1
            end
            group_class_intermediate(country, group) = u;
        end
    end
    group_alloc_opt = group_alloc_first_it; % Optimal results for group allocation are group_alloc_opt
    % Group classification
    [group_class, new_group_alloc] = min(group_class_intermediate, [], 2);

    delta = norm(thetas - thetas_opt_first_it, 'fro')^2 + norm(Lambda_IFE*F_IFE' - Lambda_first_it*F_first_it','fro')^2;

    Lambda_first_it = Lambda_IFE; % Optimal results for lambda are Lambda_first_it
    F_first_it = F_IFE; % Optimal results for F are F_first_it
    thetas_opt_first_it = thetas;  % Optimal results for thetas are thetas_opt_first_it

    group_alloc_first_it = new_group_alloc;
    countries_per_group_first = histcounts(group_alloc_first_it, 1:G+1);
end
thetas_opt_IFE = thetas_opt_first_it;
thetas_true = thetas_opt_IFE;
sigma2 = 0.5;

% DGP for IFE
for j = 1:MC_sim
    disp(j)
    % Compute the variance of the residuals
    fm = zeros(N,T);
    V=randn(N*T,1);
    err = sqrt(sigma2).*V/std(V); % IID normal DGP
    err = reshape(err,T,N)';
    x1 = reshape(X_lagged(:,1),T,N);
    x2 = reshape(X_lagged(:,2),T,N);
    x3 = reshape(X_lagged(:,3),T,N);
    x4 = reshape(X_lagged(:,4),T,N);
    x5 = reshape(X_lagged(:,5),T,N);
    x6 = reshape(X_lagged(:,6),T,N);
    x7 = reshape(X_lagged(:,7),T,N);
    x8 = reshape(X_lagged(:,8),T,N);
    Lambda_IFE = Lambda_first_it;
    F_IFE = F_first_it;
    for i = 1:N
        for t = 1:T
            fm(i,t) = x1(t,i)*thetas_opt_IFE(1) + x2(t,i)*thetas_opt_IFE(2) + x3(t,i)*thetas_opt_IFE(3) + x4(t,i)*thetas_opt_IFE(4) + x5(t,i)*thetas_opt_IFE(5) + x6(t,i)*thetas_opt_IFE(6) + x7(t,i)*thetas_opt_IFE(7) + x8(t,i)*thetas_opt_IFE(8) + Lambda_IFE(i,:) * F_IFE(t,:)' + err(i,t);
        end
    end
    Y0 = reshape(fm',N*T,1);
    clear Y_lagged
    Y_lagged = Y0;

    delta = 1;
    while delta > 10^(-4)
        proj_matrix=eye(T)-F_IFE*F_IFE'/T; % Construct the projection matrix
        var1 = 0;
        var2 = 0;
        for i = 1:N
            X_port = X_lagged((i-1)*T+1:i*T,:);
            Y_port = Y_lagged((i-1)*T+1:i*T);
            var1 = var1 + X_port'*proj_matrix*X_port;
            var2 = var2 + X_port'*proj_matrix*Y_port;
        end
        thetas = var1\var2;

        U = Y_lagged - X_lagged * thetas;
        residuals = reshape(U,T,N)';
        [V,D] = eig(residuals'*residuals); % This is the covariance matrix
        [~,eigenvalues] = sort(diag(D), 'ascend');
        F_IFE = V(:,eigenvalues(1:r));

        Lambda_IFE = residuals*F_IFE/T;

        group_class_intermediate = zeros(N, G);
        for country = 1:N
            y = Y_lagged((country-1)*T+1:country*T); % Selects the data related to the dependent variable for each period of each country
            x = X_lagged((country-1)*T+1:country*T, :); % Selects the data related to the independent variables for each period of each country
            for group = 1:G
                u = 0.0;
                for period = 1:T
                    u = u + (y(period) - x(period,:) * thetas - Lambda_IFE(country,group) * F_IFE(period,group)')^2; % Step 2 of Algorithm 1
                end
                group_class_intermediate(country, group) = u;
            end
        end
        group_alloc_opt = group_alloc_first_it; % Optimal results for group allocation are group_alloc_first_it
        % Group classification
        [group_class, new_group_alloc] = min(group_class_intermediate, [], 2);

        delta = norm(thetas - thetas_opt_first_it, 'fro')^2 + norm(Lambda_IFE*F_IFE' - Lambda_first_it*F_first_it','fro')^2;

        Lambda_first_it = Lambda_IFE; % Optimal results for lambda are Lambda_first_it
        F_first_it = F_IFE; % Optimal results for F are F_first_it
        thetas_opt_first_it = thetas;  % Optimal results for thetas are thetas_opt_first_it

        group_alloc_first_it = new_group_alloc;
        countries_per_group_first = histcounts(group_alloc_first_it, 1:G+1);
    end
    MC_IFE_thetas(j,:) = thetas_opt_first_it';
end

disp("The bias is:")
disp(abs(thetas_true' - mean(MC_IFE_thetas)))
disp("The standard errors are:")
disp(std(MC_IFE_thetas))