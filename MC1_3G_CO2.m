%Load the MC datasets
load('inputMCXG3_het_CO2.txt');
load('inputMCYG3_het_CO2.txt');

N = 21;
G = 3;
T = 19;
K = 8;
sim = 1000;
num_simulations = 500;
opt_group_assign_all = zeros(num_simulations, N);
opt_thetas_assign_all = zeros(num_simulations,G*K);

% Begin the simulations
for sth = 1:num_simulations
    disp(sth)
    obj_value_initial = 10^10; % Set a high initial objective value
    thetas_opt = zeros(K,1);
    opt_group_assign = zeros(N,1);
    alphas_opt = zeros(T,G);
    group_class = zeros(N,1);
    countries_per_group_opt = zeros(G,1);

    % Obtain the data for X and Y for all countries and periods in the current simulation
    X = inputMCXG3_het_CO2((sth-1)*N*T+1:sth*N*T,:);
    Y = inputMCYG3_het_CO2((sth-1)*N*T+1:sth*N*T);

    for j = 1:sim
        % Initialize the variables
        thetas = randn(K,G);
        W = Y - X * thetas;

        % Select G random centers
        V = randi(N, G, 1); % Selects G random countries from the set
        alphas_intermediate = zeros(T, G);
        for value = 1:G
            alphas_intermediate(:, value) = W((V(value)-1)*T+1:V(value)*T); % Selects the alpha values for all periods of these G countries
        end

        s = 0;
        delta = 1;
        while delta > 0 && s <= 100
            s = s + 1;
            % Step 1: Assignment
            group_class_intermediate = zeros(N, G);
            for country = 1:N
                y = Y((country-1)*T+1:country*T); % Select the data related to the dependent variable for each period of each country
                x = X((country-1)*T+1:country*T, :); % Select the data related to the independent variables for each period of each country
                for group = 1:G
                    u = 0.0;
                    for period = 1:T
                        u = u + (y(period) - x(period,:) * thetas(:,group) - alphas_intermediate(period, group))^2; % Step 2 of Algorithm 1
                    end
                    group_class_intermediate(country, group) = u;
                end
            end

            % Group classification
            [group_class, group_assign_intermediate] = min(group_class_intermediate, [], 2);
            countries_per_group = histcounts(group_assign_intermediate, 1:G+1);

            % Check for empty groups as per Hansen and Mladenovic
            for value = 1:G
                if countries_per_group(value) == 0
                    [~, ii] = max(abs(group_class)); % Select the country with the biggest squared difference between alpha and residuals
                    group_assign_intermediate(ii) = value;
                    countries_per_group(value) = 1;
                    group_class(ii) = 0.0;
                end
            end

            % Step 2: Update
            x1gt = zeros(T, G);
            x2gt = zeros(T, G);
            x3gt = zeros(T, G);
            x4gt = zeros(T, G);
            x5gt = zeros(T, G);
            x6gt = zeros(T, G);
            x7gt = zeros(T, G);
            x8gt = zeros(T, G);
            ygt = zeros(T, G);

            for value = 1:N
                for c = 1:G
                    if group_assign_intermediate(value) == c
                        for t = 1:T
                            x1gt(t, c) = x1gt(t, c) + X((value-1)*T+t, 1) / countries_per_group(c); % Computes the within-group mean of covariate 1 for each time period
                            x2gt(t, c) = x2gt(t, c) + X((value-1)*T+t, 2) / countries_per_group(c); % Computes the within-group mean of covariate 2 for each time period
                            x3gt(t, c) = x3gt(t, c) + X((value-1)*T+t, 3) / countries_per_group(c); % Computes the within-group mean of covariate 3 for each time period
                            x4gt(t, c) = x4gt(t, c) + X((value-1)*T+t, 4) / countries_per_group(c); % Computes the within-group mean of covariate 4 for each time period
                            x5gt(t, c) = x5gt(t, c) + X((value-1)*T+t, 5) / countries_per_group(c); % Computes the within-group mean of covariate 5 for each time period
                            x6gt(t, c) = x6gt(t, c) + X((value-1)*T+t, 6) / countries_per_group(c); % Computes the within-group mean of covariate 6 for each time period
                            x7gt(t, c) = x7gt(t, c) + X((value-1)*T+t, 7) / countries_per_group(c); % Computes the within-group mean of covariate 7 for each time period
                            x8gt(t, c) = x8gt(t, c) + X((value-1)*T+t, 8) / countries_per_group(c); % Computes the within-group mean of covariate 8 for each time period
                            ygt(t, c) = ygt(t, c) + Y((value-1)*T+t) / countries_per_group(c); % Computes the within-group mean of the response variable for each time period
                        end
                    end
                end
            end

            % Compute demeaned vectors
            X_demeaned = zeros(N*T, K);
            Y_demeaned = zeros(N*T, 1);
            thetas_new = zeros(K,G);
            for c = 1:G
                for value = 1:N
                    if group_assign_intermediate(value) == c
                        for t = 1:T
                            X_demeaned((value-1)*T+t, 1) = X((value-1)*T+t, 1) - x1gt(t, c); % Demeans the first covariate
                            X_demeaned((value-1)*T+t, 2) = X((value-1)*T+t, 2) - x2gt(t, c); % Demeans the second covariate
                            X_demeaned((value-1)*T+t, 3) = X((value-1)*T+t, 3) - x3gt(t, c); % Demeans the third covariate
                            X_demeaned((value-1)*T+t, 4) = X((value-1)*T+t, 4) - x4gt(t, c); % Demeans the fourth covariate
                            X_demeaned((value-1)*T+t, 5) = X((value-1)*T+t, 5) - x5gt(t, c); % Demeans the fifth covariate
                            X_demeaned((value-1)*T+t, 6) = X((value-1)*T+t, 6) - x6gt(t, c); % Demeans the sixth covariate
                            X_demeaned((value-1)*T+t, 7) = X((value-1)*T+t, 7) - x7gt(t, c); % Demeans the seventh covariate
                            X_demeaned((value-1)*T+t, 8) = X((value-1)*T+t, 8) - x8gt(t, c); % Demeans the eigth covariate
                            Y_demeaned((value-1)*T+t) = Y((value-1)*T+t) - ygt(t, c); % Demeans the response variable
                        end
                    end
                end
                theta_c = (X_demeaned' * X_demeaned)\ (X_demeaned' * Y_demeaned);
                thetas_new(:,c) = theta_c;
            end

            % Update the objective function
            obj_value = 0;
            for value = 1:N
                for c = 1:G
                    if group_assign_intermediate(value) == c
                        for t = 1:T
                            obj_value = obj_value + (Y((value-1)*T+t) - X((value-1)*T+t,:) * thetas_new(:,c) - alphas_intermediate(t,c))^2;
                        end
                    end
                end
            end

            % Compute time-trends
            alphas = zeros(T, G);
            for value = 1:N
                for c = 1:G
                    if group_assign_intermediate(value) == c
                        for t = 1:T
                            alphas(t, c) = alphas(t, c) + (Y((value-1)*T+t) - X((value-1)*T+t, 1) * thetas_new(1,c) - X((value-1)*T+t, 2) * thetas_new(2,c) - X((value-1)*T+t, 3) * thetas_new(3,c)- X((value-1)*T+t, 4) * thetas_new(4,c)- X((value-1)*T+t, 5) * thetas_new(5,c) - X((value-1)*T+t, 6) * thetas_new(6,c) - X((value-1)*T+t, 7) * thetas_new(7,c)- X((value-1)*T+t, 8) * thetas_new(8,c))/ countries_per_group(c); % Step 3 of Algorithm 1
                        end
                    end
                end
            end

            delta = sum((thetas_new(:) - thetas(:)).^2) + sum((alphas(:) - alphas_intermediate(:)).^2); % Necessary to test the convergence of the algorithm
            thetas = thetas_new;
            alphas_intermediate = alphas;

            % Store the optimal group assignments, theta values, time trends, etc.
            if obj_value < obj_value_initial
                thetas_opt = thetas;
                alphas_opt = alphas;
                obj_value_initial = obj_value;
                opt_group_assign = group_assign_intermediate;
                countries_per_group_opt = countries_per_group;
            end
        end
    end
    opt_group_assign_all(sth, :) = opt_group_assign';
    opt_thetas_assign_all(sth,:,:) = reshape(thetas_opt,1,G*K);
end
return;

% Open file for writing
fileID1 = fopen('assign_G3_Het_C02.txt', 'w');
fileID2 = fopen('theta_G3_Het_new_CO2.txt', 'w');

% Save all transposed results to the text files
for sth = 1:num_simulations
    fprintf(fileID1, '%d ', opt_group_assign_all(sth, :));
    fprintf(fileID1, '\n');
    fprintf(fileID2, '%d ', opt_thetas_assign_all(sth, :));
    fprintf(fileID2, '\n');
end

% Close files
fclose(fileID1);
fclose(fileID2);

