% Define the file path
data = readtable("C:\Users\576253im\Desktop\Thesis\Air Pollution Data New.xlsx");
data = table2array(data);
Y = data(:,3);
X = data(:,4:11);
T = 20;
K = 8;
N = size(Y,1)/T;
G = 3;
thetas_opt_first_it = zeros(K,G);
conv = 10^(-10);
group_alloc_first_it = randi(G, N, 1);
countries_per_group_first = histcounts(group_alloc_first_it, 1:G+1);
r = 3; % number of factors
MC_sim = 500;
MC_IFE_thetas = zeros(r, K, MC_sim);
thetas = zeros(K,G);
obj_value_initial = 10^10;
final_thetas = zeros(K,G);

% Check for empty groups
for value = 1:G
    if countries_per_group_first(value) == 0
        group_alloc_first_it(randi([1,N])) = value; % Correct index range
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

oops = 0;
while obj_value_initial >= 21.1710
    oops = oops + 1;
    disp(oops)
    group_alloc_first_it = randi(G, N, 1);
    countries_per_group_first = histcounts(group_alloc_first_it, 1:G+1);
    F_first_it = 0.7*randn(T,r);

    %% Step 1 of Algorithm 3: Compute thetas
    proj_matrix = eye(T) - F_first_it * F_first_it' / T; % Construct the projection matrix
    for g = 1:G
        var1 = 0;
        var2 = 0;
        countries_in_group = find(group_alloc_first_it == g);
        for i = countries_in_group'
            X_port = X_lagged((i-1)*T+1:i*T, :);
            Y_port = Y_lagged((i-1)*T+1:i*T);
            var1 = var1 + X_port' * proj_matrix * X_port;
            var2 = var2 + X_port' * proj_matrix * Y_port;
        end
        thetas_opt_first_it(:, g) = var1 \ var2;
    end

    %% Step 2 of Algorithm 3: Compute F's
    U = zeros(N*T, 1); % Initialize the residual vector
    for i = 1:N
        for g = 1:G
            if group_alloc_first_it(i) == g
                U((i-1)*T+1:i*T) = Y_lagged((i-1)*T+1:i*T) - X_lagged((i-1)*T+1:i*T, :) * thetas_opt_first_it(:, g);
            end
        end
    end
    residuals = reshape(U, T, N)';
    [V, D] = eig(residuals' * residuals); % This is the covariance matrix
    [~, eigenvalues] = sort(diag(D), 'descend');
    F_first_it = V(:, eigenvalues(1:r));

    %% Step 3 of Algorithm 3: Computes λ's
    Lambda_first_it = residuals * F_first_it / T;

    %% Step 4 of Algorithm 3: Compute the new group allocation
    new_group_alloc = zeros(N, 1);
    country_res = zeros(N, 1);
    for i = 1:N
        y_it = Y((i-1)*T+1:i*T);
        x_it = X((i-1)*T+1:i*T, :);
        min_ssr = inf;
        best_group = 1;
        for g = 1:G
            residuals = y_it - x_it * thetas_opt_first_it(:, g) - (Lambda_first_it(i, :) * F_first_it')';
            ssr = sum(residuals .^ 2);
            if ssr < min_ssr
                min_ssr = ssr;
                best_group = g;
                country_res(i) = ssr;
            end
        end
        new_group_alloc(i) = best_group;
    end
    
    % Checks for empty groups
    countries_per_group = histcounts(new_group_alloc, 1:G+1);
    for value = 1:G
        if countries_per_group(value) == 0
            [~, ii] = max(abs(country_res)); % Selects the country with the biggest squared difference between alpha and residuals
            new_group_alloc(ii) = value;
            countries_per_group(value) = 1;
            country_res(ii) = 0.0;
        end
    end

    delta = 1;
    s = 0;
    while delta > 0 && s <= 100
        s = s + 1;
        %% Step 1 of Algorithm 3: Compute thetas
        proj_matrix = eye(T) - F_first_it * F_first_it' / T; % Construct the projection matrix
        for g = 1:G
            var1 = 0;
            var2 = 0;
            countries_in_group = find(group_alloc_first_it == g);
            for i = countries_in_group'
                X_port = X_lagged((i-1)*T+1:i*T, :);
                Y_port = Y_lagged((i-1)*T+1:i*T);
                var1 = var1 + X_port' * proj_matrix * X_port;
                var2 = var2 + X_port' * proj_matrix * Y_port;
            end
            if rcond(var1) > eps
                thetas(:, g) = var1 \ var2;
            end
        end

        %% Step 2 of Algorithm 3: Compute F's
        U = zeros(N*T, 1); % Initialize the residual vector
        for i = 1:N
            for g = 1:G
                if group_alloc_first_it(i) == g
                    U((i-1)*T+1:i*T) = Y_lagged((i-1)*T+1:i*T) - X_lagged((i-1)*T+1:i*T, :) * thetas(:, g);
                end
            end
        end
        residuals = reshape(U, T, N)';
        [V, D] = eig(residuals' * residuals); % This is the covariance matrix
        [~, eigenvalues] = sort(diag(D), 'descend');
        F_IFE = V(:, eigenvalues(1:r));

        %% Step 3 of Algorithm 3: Computes λ's
        Lambda_IFE = residuals * F_IFE / T;

        %% Step 4 of Algorithm 3: Compute the new group allocation
        new_group_alloc = zeros(N, 1);
        country_res = zeros(N, 1);
        for i = 1:N
            y_it = Y((i-1)*T+1:i*T);
            x_it = X((i-1)*T+1:i*T, :);
            min_ssr = inf;
            best_group = 1;
            for g = 1:G
                residuals = y_it - x_it * thetas(:, g) - (Lambda_IFE(i, :) * F_IFE')';
                ssr = sum(residuals .^ 2);
                if ssr < min_ssr
                    min_ssr = ssr;
                    best_group = g;
                    country_res(i) = ssr;
                end
            end
            new_group_alloc(i) = best_group;
        end

        % Checks for empty groups
        countries_per_group = histcounts(new_group_alloc, 1:G+1);
        for value = 1:G
            if countries_per_group(value) == 0
                [~, ii] = max(abs(country_res)); % Selects the country with the biggest squared difference between alpha and residuals
                new_group_alloc(ii) = value;
                countries_per_group(value) = 1;
                country_res(ii) = 0.0;
            end
        end

        % Update the objective function
        obj_value = 0;
        for value = 1:N
            for c = 1:G
                aux = (Lambda_IFE(i, :) * F_IFE')';
                for t = 1:T
                    if group_alloc_first_it(value) == c
                        obj_value = obj_value + (Y((value-1)*T+t) - X((value-1)*T+t, :) * thetas(:, g) - aux(t,1))^2;
                    end
                end
            end
        end
        delta = norm(thetas - thetas_opt_first_it, 'fro')^2 + norm(Lambda_IFE * F_IFE' - Lambda_first_it * F_first_it', 'fro')^2;

        Lambda_first_it = Lambda_IFE;
        F_first_it = F_IFE;
        thetas_opt_first_it = thetas;
        group_alloc_first_it = new_group_alloc;

        if obj_value < obj_value_initial
            thetas_final = thetas_opt_first_it;
            obj_value_initial = obj_value;
            final_groups = group_alloc_first_it;
            final_F = F_IFE;
            final_Lambda = Lambda_IFE;
        end
    end
end
sigma2 = obj_value_initial/(N*T-G*T-N-K);

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

    for i = 1:N
        for t = 1:T
            fm(i,t) = x1(t,i)*thetas_final(1) + x2(t,i)*thetas_final(2) + x3(t,i)*thetas_final(3) + x4(t,i)*thetas_final(4) + x5(t,i)*thetas_final(5) + x6(t,i)*thetas_final(6) + x7(t,i)*thetas_final(7) + x8(t,i)*thetas_final(8) + final_Lambda(i,:) * final_F(t,:)' + err(i,t);
        end
    end
    Y0 = reshape(fm',N*T,1);
    clear Y_lagged
    Y_lagged = Y0;

    % Initial allocation is the true parameter values
    F_first_it = final_F;
    thetas_opt_first_it = thetas_final;
    group_alloc_first_it = final_groups;
    obj_value_initial = 10^10; % Resets the objective value

    delta = 1;
    s = 0;
    while delta > 0 && s <= 100
        s = s + 1;
        % Step 1 of Algorithm 3: Compute thetas
        proj_matrix = eye(T) - F_first_it * F_first_it' / T; % Construct the projection matrix
        for g = 1:G
            var1 = 0;
            var2 = 0;
            countries_in_group = find(group_alloc_first_it == g);
            for i = countries_in_group'
                X_port = X_lagged((i-1)*T+1:i*T, :);
                Y_port = Y_lagged((i-1)*T+1:i*T);
                var1 = var1 + X_port' * proj_matrix * X_port;
                var2 = var2 + X_port' * proj_matrix * Y_port;
            end
            if rcond(var1) > eps
                thetas(:, g) = var1 \ var2;
            end
        end

        % Step 2 of Algorithm 3: Compute F's
        U = zeros(N*T, 1); % Initialize the residual vector
        for i = 1:N
            for g = 1:G
                if group_alloc_first_it(i) == g
                    U((i-1)*T+1:i*T) = Y_lagged((i-1)*T+1:i*T) - X_lagged((i-1)*T+1:i*T, :) * thetas(:, g);
                end
            end
        end
        residuals = reshape(U, T, N)';
        [V, D] = eig(residuals' * residuals); % This is the covariance matrix
        [~, eigenvalues] = sort(diag(D), 'descend');
        F_IFE = V(:, eigenvalues(1:r));

        % Step 3 of Algorithm 3: Computes λ's
        Lambda_IFE = residuals * F_IFE / T;

        % Step 4 of Algorithm 3: Compute the new group allocation
        new_group_alloc = zeros(N, 1);
        country_res = zeros(N, 1);
        for i = 1:N
            y_it = Y((i-1)*T+1:i*T);
            x_it = X((i-1)*T+1:i*T, :);
            min_ssr = inf;
            best_group = 1;
            for g = 1:G
                residuals = y_it - x_it * thetas(:, g) - (Lambda_IFE(i, :) * F_IFE')';
                ssr = sum(residuals .^ 2);
                if ssr < min_ssr
                    min_ssr = ssr;
                    best_group = g;
                    country_res(i) = ssr;
                end
            end
            new_group_alloc(i) = best_group;
        end

        % Checks for empty groups
        countries_per_group = histcounts(new_group_alloc, 1:G+1);
        for value = 1:G
            if countries_per_group(value) == 0
                [~, ii] = max(abs(country_res)); % Selects the country with the biggest squared difference between alpha and residuals
                new_group_alloc(ii) = value;
                countries_per_group(value) = 1;
                country_res(ii) = 0.0;
            end
        end

        % Update the objective function
        obj_value = 0;
        for value = 1:N
            for c = 1:G
                aux = (Lambda_IFE(i, :) * F_IFE')'; % Computes the interactive fixed effect
                for t = 1:T
                    if group_alloc_first_it(value) == c
                        obj_value = obj_value + (Y((value-1)*T+t) - X((value-1)*T+t, :) * thetas(:, g) - aux(t,1))^2;
                    end
                end
            end
        end
        delta = norm(thetas - thetas_opt_first_it, 'fro')^2 + norm(Lambda_IFE * F_IFE' - Lambda_first_it * F_first_it', 'fro')^2;

        Lambda_first_it = Lambda_IFE;
        F_first_it = F_IFE;
        thetas_opt_first_it = thetas;
        group_alloc_first_it = new_group_alloc;

        if obj_value < obj_value_initial
            thetas_final_2 = thetas_opt_first_it;
            obj_value_initial = obj_value;
            final_groups_2 = group_alloc_first_it;
            final_F_2 = F_IFE;
            final_Lambda_2 = Lambda_IFE;
        end
    end
    MC_IFE_thetas(:, :, j) = thetas_final_2';
end


disp("The bias is:")
disp(abs(thetas_final' - mean(MC_IFE_thetas,3)))
disp("The standard errors are:")
disp(std(MC_IFE_thetas, 0, 3) / sqrt(MC_sim))