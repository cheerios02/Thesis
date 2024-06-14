% Load the optimal group assignments and necessary data for all simulations 
load('inputMCXG3_HC.txt');
load('inputMCYG3_HC.txt');
load('BigG_perm_G3_HC');

% Define the number of periods, variables, replications, countries, and groups
T = 7;
Var = 2;
N = 90;
repNum = 500;
optGroup = BigG_perm;
G = 3;

% Initialize variables
thetas = zeros(repNum, G*Var);  
std_cluster_vect = zeros(repNum, G*T+Var, G);
std_theta_inc_vect = zeros(repNum, G);

% Initiate the simulations
for sim = 1:repNum
    X = inputMCXG3_HC((sim-1)*N*T+1:sim*N*T, :); % Obtain the data for X for all countries and periods in the current simulation
    Y = inputMCYG3_HC((sim-1)*N*T+1:sim*N*T); % Obtain the data for Y for all countries and periods in the current simulation
    opt_group_assign = optGroup(:, sim); % Obtain the optimal group assignment for all countries in the current simulation
    which_group = zeros(N, G);

    for g = 1:G
        which_group(:, g) = (opt_group_assign == g); % Stores the optimal group of the country in the current simulation
    end

    Y_group_aver=zeros(N*T,1);
    X_group_aver=zeros(N*T,Var);

    country_sum_per_group=sum(which_group); % Obtain the sum of total countries in each group in the current simulation

    for i=1:N
        if country_sum_per_group(opt_group_assign(i))>1 % If the country is not the only one in the group
            for t=1:T
                Yt=Y(t:T:N*T);
                Y_group_aver((i-1)*T+t)=mean(Yt(opt_group_assign==opt_group_assign(i))); % For each period, obtain the average of the Y data for the countries that belong to the same group
                Xt=X(t:T:N*T,:);
                X_group_aver((i-1)*T+t,:)=mean(Xt(opt_group_assign==opt_group_assign(i),:));
            end
        else % If the country is the only one in the group
            for t=1:T
                Yt=Y(t:T:N*T);
                Y_group_aver((i-1)*T+t)=mean(Yt(opt_group_assign==opt_group_assign(i)));
                Xt=X(t:T:N*T,:);
                X_group_aver((i-1)*T+t,:)=Xt(opt_group_assign==opt_group_assign(i),:);
            end
        end
    end

    % Initialize group-specific variables
    denominator = zeros(Var,Var,G);
    numerator = zeros(Var,G);
    theta_aux = zeros(Var,G);

    for g = 1:G
        for i = 1:N
            if opt_group_assign(i) == g
                for t = 1:T
                    denominator(:,:,g) = denominator(:,:,g) + (X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(X((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:));
                    numerator(:,g) = numerator(:,g) + (X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(Y((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:));
                end
            end
        end
        theta_aux(:,g) = denominator(:,:,g) \ numerator(:,g); % Compute the thetas using an OLS regression
    end
    gitot = kron(which_group,eye(T));
    thetas(sim,:) = reshape(theta_aux,1,Var*G);

    % Compute residuals for each group
    ei_group = cell(G, 1); % Cell array to store residuals for each group
    Rei_group = cell(G, 1); % Cell array to store reshaped residuals for each group
    Omega = zeros(N*T, N*T); % Initialize Omega

    for g = 1:G
        % Compute residuals for group g
        ei_group{g} = Y - Y_group_aver - (X - X_group_aver) * theta_aux(:,g);
        
        % Reshape the residuals for group g
        Rei_group{g} = reshape(ei_group{g}, T, N)';
        
        % Initialize Mi for group g
        Mi = zeros(T, T);

        for i = 1:N
            Mi = Rei_group{g}(i, :)' * Rei_group{g}(i, :);
            Omega((i-1)*T+1:i*T,(i-1)*T+1:i*T) = Mi;
        end

        % Continue with the remaining part of your code
        gitot = kron(which_group, eye(T));
        Xtot = [gitot X];
        V = inv(Xtot' * Xtot) * Xtot' * Omega * Xtot * inv(Xtot' * Xtot);
        V = V * N * T / (N * T - G * T - Var);
        std_cluster = sqrt(diag(V));
        std_cluster_vect(sim, :, g) = std_cluster';

        % Compute total income effect: standard errors for each group
        Mat_inc = [theta_aux(2, g)/(1-theta_aux(1, g))^2, 1/(1-theta_aux(1, g))];
        Var_theta_inc = Mat_inc*V(G*T+1:G*T+Var,G*T+1:G*T+Var)*Mat_inc';
        std_theta_inc = sqrt(Var_theta_inc);
        std_theta_inc_vect(sim, g) = std_theta_inc;
    end
end

true_thetas = [0.2992;0.0382;0.5597;0.0645;0.4479;0.0843];
true_thetas_bias = reshape(true_thetas,Var,G);

disp("The true theta values are:")
disp(true_thetas_bias)
disp(true_thetas_bias(2, :) ./ (1 - true_thetas_bias(1, :)))

% Bias:
disp('The mean theta results across all simulations are are:')
mean_theta = reshape(mean(thetas), [Var, G]);
disp(mean_theta)
disp(mean_theta(2, :) ./ (1 - mean_theta(1, :)))

% Standard Deviation
disp('The standard deviation for theta across all simulations is:')
std_theta = reshape(std(thetas), [Var, G]);
disp(std_theta)
disp(std_theta(2, :) ./ (1 - std_theta(1, :)))

disp('The medians using the large-T clustered variance standard error approach across all simulations are:');
for g = 1:G
    median(std_cluster_vect(:, G*T+1:G*T+Var,g))
end
median(std_theta_inc_vect)

% Coverage
coverage_probs = zeros(Var+1,G);
for g = 1:G
    asympt_std = std_cluster_vect(:,G*T+1:G*T+Var,g);
    for count = 1:repNum
        for k = 1:Var
            if (thetas(count,(g-1)*Var+k)-2.24*asympt_std(count,k) < true_thetas((g-1)*Var+k,1)) && (thetas(count,(g-1)*Var+k)+2.24*asympt_std(count,k)>true_thetas((g-1)*Var+k,1))
                coverage_probs(k,g) = coverage_probs(k,g) + 1/repNum;
            end
        end
    end

    std_cum_inc = std_theta_inc_vect(:,g);
    for count = 1:repNum
        if (thetas(count,2*g)/(1-thetas(count,2*g-1))-2.24*std_cum_inc(count,1) < true_thetas(2*g,1)/(1-true_thetas(2*g-1,1)) && (thetas(count,2*g)/(1-thetas(count,2*g-1))+2.24*std_cum_inc(count,1)>(true_thetas(2*g,1)/(1-true_thetas(2*g-1,1)))))
            coverage_probs(3,g) = coverage_probs(3,g) + 1/repNum;
        end
    end
end

disp('The empirical coverage probabilities (at a 5% level) across all simulations are: ')
disp(coverage_probs)
