% Attempting to do the replication for the heterogeneous coefficients as
% well
% Load the final data
load('final_data');

% Variable declaration
X = ldem_linc(:,1:2);
Y = dem;
T = 7; % Number of periods
K = 2; % Number of variables
N = size(Y,1)/T; % Number of countries
G = 3; % Number of groups  
sim = 2500; % Number of simulations
thetas_opt = zeros(K, G); % Modify to store group-specific thetas
opt_group_assign = zeros(N,1);
alphas_opt = zeros(T,G);
group_class = zeros(N,1);
countries_per_group_opt = zeros(G,1);
obj_value_initial = 10^10; % Set a high initial objective value

% Begin the simulations 
for j = 1:sim
    disp(j)
    % Initialize the variables
    theta_random_value = randn;
    thetas = theta_random_value * ones(K, G); % Modify to initialize group-specific thetas
    W = Y - X * thetas(:,1); % Initial W computation for the first group

    % Select G random centers
    V = randi(N, G, 1); % Selects G random countries from the set
    alphas_intermediate = zeros(T, G);
    for value = 1:G
        alphas_intermediate(:, value) = W((V(value)-1)*T+1:V(value)*T); % Selects the alpha values for all periods of these G countries 
    end

    delta = 1;
    while delta > 0
        % Step 1: Assignment
        group_class_intermediate = zeros(N, G);
        for country = 1:N
            y = Y((country-1)*T+1:country*T); % Select the data related to the dependent variable for each period of each country
            x = X((country-1)*T+1:country*T, :); % Select the data related to the independent variables for each period of each country
            for group = 1:G
                u = 0.0;
                for period = 1:T
                    u = u + (y(period) - x(period,:) * thetas(:, group) - alphas_intermediate(period, group))^2; % Step 2 of Algorithm 1 with group-specific thetas
                end
                group_class_intermediate(country, group) = u;
            end
        end

        % Group classification
        [group_class, group_assign_intermediate] = min(group_class_intermediate, [], 2);
        countries_per_group = histcounts(group_assign_intermediate, 1:G+1);

        % Checks for empty groups as per Hansen and Mladenovic
        for value = 1:G
            if countries_per_group(value) == 0
                [~, ii] = max(abs(group_class)); % Selects the country with the biggest squared difference between alpha and residuals 
                group_assign_intermediate(ii) = value;
                countries_per_group(value) = 1;
                group_class(ii) = 0.0;
            end
        end

        % Step 2: Update
        x1gt = zeros(T, G);
        x2gt = zeros(T, G);
        ygt = zeros(T, G);

        for value = 1:N
            for c = 1:G
                if group_assign_intermediate(value) == c
                    for t = 1:T
                        x1gt(t, c) = x1gt(t, c) + X((value-1)*T+t, 1) / countries_per_group(c); % Computes the within-group mean of covariate 1 for each time period
                        x2gt(t, c) = x2gt(t, c) + X((value-1)*T+t, 2) / countries_per_group(c); % Computes the within-group mean of covariate 2 for each time period
                        ygt(t, c) = ygt(t, c) + Y((value-1)*T+t) / countries_per_group(c); % Computes the within-group mean of the response variable for each time period
                    end
                end
            end
        end

        % Compute demeaned vectors and update thetas for each group
        X_demeaned = zeros(N*T, K);
        Y_demeaned = zeros(N*T, 1);
        thetas_new = zeros(K, G);
        for c = 1:G
            for value = 1:N
                if group_assign_intermediate(value) == c
                    for t = 1:T
                        X_demeaned((value-1)*T+t, 1) = X((value-1)*T+t, 1) - x1gt(t, c); % Demean the first covariate
                        X_demeaned((value-1)*T+t, 2) = X((value-1)*T+t, 2) - x2gt(t, c); % Demean the second covariate
                        Y_demeaned((value-1)*T+t) = Y((value-1)*T+t) - ygt(t, c); % Demean the response variables
                    end
                end
            end
            theta_c = (X_demeaned' * X_demeaned) \ (X_demeaned' * Y_demeaned); % Compute the thetas for group c using an OLS regression
            thetas_new(:, c) = theta_c;
        end

        % Update the objective function
        obj_value = 0;
        for value = 1:N
            for c = 1:G
                if group_assign_intermediate(value) == c
                    for t = 1:T
                        obj_value = obj_value + (Y((value-1)*T+t) - X((value-1)*T+t, :) * thetas_new(:, c) - alphas_intermediate(t, c))^2;
                    end
                end
            end
        end
        disp(obj_value)
        % Compute time-trends
        alphas = zeros(T, G);
        for value = 1:N
            for c = 1:G
                if group_assign_intermediate(value) == c
                    for t = 1:T
                        alphas(t, c) = alphas(t, c) + (Y((value-1)*T+t) - X((value-1)*T+t, 1) * thetas_new(1, c) - X((value-1)*T+t, 2) * thetas_new(2, c)) / countries_per_group(c); % Step 3 of Algorithm 1
                    end
                end
            end
        end

        delta = sum((thetas_new(:) - thetas(:)).^2) + sum((alphas(:) - alphas_intermediate(:)).^2); % Necessary to test the convergence of the algorithm
        thetas = thetas_new;
        alphas_intermediate = alphas;

        % Store the optimal group assignments, theta values, time trends,
        % etc.
        if obj_value < obj_value_initial
            thetas_opt = thetas;
            alphas_opt = alphas;
            obj_value_initial = obj_value;
            opt_group_assign = group_assign_intermediate;
            countries_per_group_opt = countries_per_group;
        end
    end
end
