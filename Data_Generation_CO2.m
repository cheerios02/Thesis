% Define the file path
data = readtable("C:\Users\user\Desktop\Bachelor Thesis\After Allocation\My Code\Extension\Air Pollution Data New.xlsx");
data = table2array(data);
Y = data(:,3);
X = data(:,4:11);
T = 20;
Var = 8;
N = size(Y,1)/T;
G = 3;
sim = 2500;

% Create lagged variables
X_lagged = nan(N*T, Var);
for country = 1:N
    for t = 2:T
        X_lagged((country-1)*T+t, :) = X((country-1)*T+t-1, :);
    end
end

% Remove rows with NaN values (first period for each country)
valid_rows = ~isnan(X_lagged(:, 1));
X_lagged = X_lagged(valid_rows, :);
Y_lagged = Y(valid_rows);

% Adjust the number of periods
T = T - 1;

opt_group_assign= [3 3 1 2 1 3 3 1 1 2 2 1 1 2 1 2 3 3 1 3 2]';

which_group=zeros(N,G);
for g=1:G
    which_group(:,g)=(opt_group_assign==g); 
end

Y_group_aver=zeros(N*T,1);
X_group_aver=zeros(N*T,Var);

MYbar_gt=zeros(G*T,1);
MXbar_gt=zeros(G*T,Var);

country_sum_per_group=sum(which_group); % Obtain the sum of total countries in each group in the current simulation

for i=1:N
    if country_sum_per_group(opt_group_assign(i))>1 % If the country is not the only one in the group
        for t=1:T
            Yt=Y_lagged(t:T:N*T);
            Y_group_aver((i-1)*T+t)=mean(Yt(opt_group_assign==opt_group_assign(i))); % For each period, obtain the average of the Y data for the countries that belong to the same group
            Xt=X_lagged(t:T:N*T,:);
            X_group_aver((i-1)*T+t,:)=mean(Xt(opt_group_assign==opt_group_assign(i),:));
        end
    else % If the country is the only one in the group
        for t=1:T
            Yt=Y_lagged(t:T:N*T);
            Y_group_aver((i-1)*T+t)=mean(Yt(opt_group_assign==opt_group_assign(i)));
            Xt=X_lagged(t:T:N*T,:);
            X_group_aver((i-1)*T+t,:)=Xt(opt_group_assign==opt_group_assign(i),:);
        end
    end
end

denominator=zeros(Var,Var);
numerator=zeros(Var,1);

for i=1:N
    for t=1:T
        denominator=denominator+(X_lagged((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(X_lagged((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:));
        numerator=numerator+(X_lagged((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(Y_lagged((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:));
        MYbar_gt((opt_group_assign(i)-1)*T+t)=Y_group_aver((i-1)*T+t);
        MXbar_gt((opt_group_assign(i)-1)*T+t,:)=X_group_aver((i-1)*T+t,:);
    end
end

thetas=denominator\numerator; % Obtain the thetas through OLS

a=MYbar_gt-MXbar_gt*thetas;
gitot = kron(which_group,eye(T));
delta_hat = gitot*a;

obj=0;
ei = zeros(N*T,1);
for i=1:N
    for t=1:T
        obj=obj+(Y_lagged((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:)-(X_lagged((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:))*thetas).^2;
        ei((i-1)*T+t) = Y_lagged((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:)-(X_lagged((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:))*thetas;
    end
end

% Compute the variance of the residuals
sigma2 = obj/(N*T-G*T-N-Var);

% Monte Carlo simulation 
MC_sim = 500;

% MonteCarlo_loop
for j = 1:MC_sim     
    disp(j)
    V=randn(N*T,1);
    err = sqrt(sigma2).*V/std(V); % IID normal DGP
    err = reshape(err,T,N)';
    Rdelta = reshape(delta_hat,T,N)';
    dm = zeros(N,T);
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
            dm(i,t) = x1(t,i)*thetas(1) + x2(t,i)*thetas(2) + x3(t,i)*thetas(3) + x4(t,i)*thetas(4) + x5(t,i)*thetas(5) + x6(t,i)*thetas(6) + x7(t,i)*thetas(7) + x8(t,i)*thetas(8) + Rdelta(i,t) + err(i,t);
        end
    end
    Y0 = reshape(dm',N*T,1);
    clear Y
    Y = Y0;
    XM((j-1)*N*T+1:j*N*T,:) = X_lagged;
    YM((j-1)*N*T+1:j*N*T,1) = Y;
end

% Open file for writing
fileID1 = fopen('inputMCXG3_new_CO2.txt', 'w');
fileID2 = fopen('inputMCYG3_new_CO2.txt', 'w');

for sth = 1:T*N*MC_sim
    fprintf(fileID1, '%d ', XM(sth, :));
    fprintf(fileID1, '\n');
    fprintf(fileID2, '%d ', YM(sth, :));
    fprintf(fileID2, '\n');
end

% Close files
fclose(fileID1);
fclose(fileID2);

% 
% 
% 
% 
% 
% % Compute the optimal thetas and alphas for later 
% % Begin the simulations
% for j = 1:sim
%     disp(j)
%     % Initialize the variables
%     theta_random_value = randn;
%     thetas = theta_random_value * ones(K, 1);
%     W = Y_lagged - X_lagged * thetas;
% 
%     % Select G random centers
%     V = randi(N, G, 1); % Selects G random countries from the set
%     alphas_intermediate = zeros(T, G);
%     for value = 1:G
%         alphas_intermediate(:, value) = W((V(value)-1)*T+1:V(value)*T); % Selects the alpha values for all periods of these G countries
%     end
% 
%     delta = 1;
%     while delta > 0
%         % Step 1: Assignment
%         group_class_intermediate = zeros(N, G);
%         for country = 1:N
%             y = Y_lagged((country-1)*T+1:country*T); % Selects the data related to the dependent variable for each period of each country
%             x = X_lagged((country-1)*T+1:country*T, :); % Selects the data related to the independent variables for each period of each country
%             for group = 1:G
%                 u = 0.0;
%                 for period = 1:T
%                     u = u + (y(period) - x(period,:) * thetas - alphas_intermediate(period, group))^2; % Step 2 of Algorithm 1
%                 end
%                 group_class_intermediate(country, group) = u;
%             end
%         end
% 
%         % Group classification
%         [group_class, group_assign_intermediate] = min(group_class_intermediate, [], 2);
%         countries_per_group = histcounts(group_assign_intermediate, 1:G+1);
% 
%         % Checks for empty groups as per Hansen and Mladenovic
%         for value = 1:G
%             if countries_per_group(value) == 0
%                 [~, ii] = max(abs(group_class)); % Selects the country with the biggest squared difference between alpha and residuals
%                 group_assign_intermediate(ii) = value;
%                 countries_per_group(value) = 1;
%                 group_class(ii) = 0.0;
%             end
%         end
% 
%         % Step 2: Update
%         x1gt = zeros(T, G);
%         x2gt = zeros(T, G);
%         x3gt = zeros(T, G);
%         x4gt = zeros(T, G);
%         x5gt = zeros(T, G);
%         x6gt = zeros(T, G);
%         x7gt = zeros(T, G);
%         x8gt = zeros(T, G);
%         ygt = zeros(T, G);
% 
%         for value = 1:N
%             for c = 1:G
%                 if group_assign_intermediate(value) == c
%                     for t = 1:T
%                         x1gt(t, c) = x1gt(t, c) + X_lagged((value-1)*T+t, 1) / countries_per_group(c); % Computes the within-group mean of covariate 1 for each time period
%                         x2gt(t, c) = x2gt(t, c) + X_lagged((value-1)*T+t, 2) / countries_per_group(c); % Computes the within-group mean of covariate 2 for each time period
%                         x3gt(t, c) = x3gt(t, c) + X_lagged((value-1)*T+t, 3) / countries_per_group(c); % Computes the within-group mean of covariate 3 for each time period
%                         x4gt(t, c) = x4gt(t, c) + X_lagged((value-1)*T+t, 4) / countries_per_group(c); % Computes the within-group mean of covariate 4 for each time period
%                         x5gt(t, c) = x5gt(t, c) + X_lagged((value-1)*T+t, 5) / countries_per_group(c); % Computes the within-group mean of covariate 5 for each time period
%                         x6gt(t, c) = x6gt(t, c) + X_lagged((value-1)*T+t, 6) / countries_per_group(c); % Computes the within-group mean of covariate 6 for each time period
%                         x7gt(t, c) = x7gt(t, c) + X_lagged((value-1)*T+t, 7) / countries_per_group(c); % Computes the within-group mean of covariate 7 for each time period
%                         x8gt(t, c) = x8gt(t, c) + X_lagged((value-1)*T+t, 8) / countries_per_group(c); % Computes the within-group mean of covariate 8 for each time period
%                         ygt(t, c) = ygt(t, c) + Y_lagged((value-1)*T+t) / countries_per_group(c); % Computes the within-group mean of the response variable for each time period
%                     end
%                 end
%             end
%         end
% 
%         % Compute demeaned vectors
%         X_demeaned = zeros(N*T, K);
%         Y_demeaned = zeros(N*T, 1);
%         for value = 1:N
%             for c = 1:G
%                 if group_assign_intermediate(value) == c
%                     for t = 1:T
%                         X_demeaned((value-1)*T+t, 1) = X_lagged((value-1)*T+t, 1) - x1gt(t, c); % Demeans the first covariate
%                         X_demeaned((value-1)*T+t, 2) = X_lagged((value-1)*T+t, 2) - x2gt(t, c); % Demeans the second covariate
%                         X_demeaned((value-1)*T+t, 3) = X_lagged((value-1)*T+t, 3) - x3gt(t, c); % Demeans the third covariate
%                         X_demeaned((value-1)*T+t, 4) = X_lagged((value-1)*T+t, 4) - x4gt(t, c); % Demeans the fourth covariate
%                         X_demeaned((value-1)*T+t, 5) = X_lagged((value-1)*T+t, 5) - x5gt(t, c); % Demeans the fifth covariate
%                         X_demeaned((value-1)*T+t, 6) = X_lagged((value-1)*T+t, 6) - x6gt(t, c); % Demeans the sixth covariate
%                         X_demeaned((value-1)*T+t, 7) = X_lagged((value-1)*T+t, 7) - x7gt(t, c); % Demeans the seventh covariate
%                         X_demeaned((value-1)*T+t, 8) = X_lagged((value-1)*T+t, 8) - x8gt(t, c); % Demeans the eighth covariate
%                         Y_demeaned((value-1)*T+t) = Y_lagged((value-1)*T+t) - ygt(t, c); % Demeans the response variable
%                     end
%                 end
%             end
%         end
% 
%         theta = (X_demeaned' * X_demeaned) \ (X_demeaned' * Y_demeaned); % Computes the thetas using an OLS regression, Step 3 of Algorithm 1
% 
%         % Updates the objective function
%         obj_value = sum((Y_demeaned - X_demeaned * theta).^2);
% 
%         % Computes time-trends
%         alphas = zeros(T, G);
%         for value = 1:N
%             for c = 1:G
%                 if group_assign_intermediate(value) == c
%                     for t = 1:T
%                         alphas(t, c) = alphas(t, c) + (Y_lagged((value-1)*T+t) - X_lagged((value-1)*T+t, 1) * theta(1) - X_lagged((value-1)*T+t, 2) * theta(2) - X_lagged((value-1)*T+t, 3) * theta(3) - X_lagged((value-1)*T+t, 4) * theta(4) - X_lagged((value-1)*T+t, 5) * theta(5) - X_lagged((value-1)*T+t, 6) * theta(6) - X_lagged((value-1)*T+t, 7) * theta(7) - X_lagged((value-1)*T+t, 8) * theta(8)) / countries_per_group(c); % Step 3 of Algorithm 1
%                     end
%                 end
%             end
%         end
% 
%         delta = sum((theta - thetas).^2) + sum((alphas(:) - alphas_intermediate(:)).^2); % Necessary to test the convergence of the algorithm
%         thetas = theta;
%         alphas_intermediate = alphas;
% 
%         % Stores the optimal group assignments, theta values, time trends, etc.
%         if obj_value < obj_value_initial
%             thetas_opt = theta;
%             alphas_opt = alphas;
%             obj_value_initial = obj_value;
%             opt_group_assign = group_assign_intermediate;
%             countries_per_group_opt = countries_per_group;
%         end
%     end
% end
% 
% % Compute the variance of the residuals
% sigma2 = obj_value_initial/(N*T-G*T-N-K);
% 
% which_group=zeros(N,G);
% 
% for g=1:G
%     which_group(:,g)=(opt_group_assign==g); 
% end
% 
% gitot = kron(which_group,eye(T));
% delta_hat = gitot*reshape(alphas_opt',1,T*G)';
% 
% % Monte Carlo simulation 
% MC_sim = 500;
% MC_IFE_thetas = zeros(MC_sim,K);
% MC_IFE_group_assign = zeros(MC_sim,N);
% 
% % MonteCarlo_loop
% for j = 1:MC_sim     
%     disp(j)
%     eigenvectors=randn(N*T,1);
%     err = sqrt(sigma2).*eigenvectors/std(eigenvectors); % IID normal DGP
%     err = reshape(err,T,N)';
%     Rdelta = reshape(delta_hat,T,N)';
%     dm = zeros(N,T);
%     x1 = reshape(X_lagged(:,1),T,N);
%     x2 = reshape(X_lagged(:,2),T,N);
%     x3 = reshape(X_lagged(:,3),T,N);
%     x4 = reshape(X_lagged(:,4),T,N);
%     x5 = reshape(X_lagged(:,5),T,N);
%     x6 = reshape(X_lagged(:,6),T,N);
%     x7 = reshape(X_lagged(:,7),T,N);
%     x8 = reshape(X_lagged(:,8),T,N);
% 
%     for i = 1:N
%         for t = 1:T
%             dm(i,t) = x1(t,i)*thetas_opt(1) + x2(t,i)*thetas_opt(2) + x3(t,i)*thetas_opt(3) + x4(t,i)*thetas_opt(4) + x5(t,i)*thetas_opt(5) + x6(t,i)*thetas_opt(6) + x7(t,i)*thetas_opt(7) + x8(t,i)*thetas_opt(8) + Rdelta(i,t) + err(i,t);
%         end
%     end
%     Y0 = reshape(dm',N*T,1);
%     clear Y
%     Y = Y0;
% 
%     XM((j-1)*N*T+1:j*N*T,:) = X_lagged;
%     YM((j-1)*N*T+1:j*N*T,1) = Y;
% end
% 
% 
% % Open file for writing
% fileID1 = fopen('inputMCXG3_CO2.txt', 'w');
% fileID2 = fopen('inputMCYG3_CO2.txt', 'w');
% 
% for sth = 1:T*N*MC_sim
%     fprintf(fileID1, '%d ', XM(sth, :));
%     fprintf(fileID1, '\n');
%     fprintf(fileID2, '%d ', YM(sth, :));
%     fprintf(fileID2, '\n');
% end
% 
% % Close files
% fclose(fileID1);
% fclose(fileID2);