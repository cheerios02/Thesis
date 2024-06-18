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

opt_group_assign= [3 3 1 3 3 3 3 1 3 3 2 1 1 2 3 2 3 3 3 3 3]';

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

denominator = zeros(Var,Var,G);
numerator = zeros(Var,G);
thetas = zeros(Var,G);

for g = 1:G
    for i=1:N
        if opt_group_assign(i) == g
            for t=1:T
                denominator(:,:,g)=denominator(:,:,g)+(X_lagged((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(X_lagged((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:));
                numerator(:,g)=numerator(:,g)+(X_lagged((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(Y_lagged((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:));
                MYbar_gt((g-1)*T+t)=Y_group_aver((i-1)*T+t);
                MXbar_gt((g-1)*T+t,:)=X_group_aver((i-1)*T+t,:);
            end
        end
    end
    thetas(:,g) = denominator(:,:,g) \ numerator(:,g);
end

a=MYbar_gt-MXbar_gt*thetas;
gitot = kron(which_group,eye(T));
delta_hat = gitot*a;

obj=0;
ei = zeros(N*T,1);
for i=1:N
    for t=1:T
        g = opt_group_assign(i);
        obj=obj+(Y_lagged((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:)-(X_lagged((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:))*thetas(:,g)).^2;
        ei((i-1)*T+t) = Y_lagged((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:)-(X_lagged((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:))*thetas(:,g);
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
    Rdelta = reshape(delta_hat(:,g),T,N)';
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
        g = opt_group_assign(i);
        for t = 1:T
            dm(i,t) = x1(t,i)*thetas(1,g) + x2(t,i)*thetas(2,g) + x3(t,i)*thetas(3,g) + x4(t,i)*thetas(4,g) + x5(t,i)*thetas(5,g) + x6(t,i)*thetas(6,g) + x7(t,i)*thetas(7,g) + x8(t,i)*thetas(8,g) + Rdelta(i,t) + err(i,t);
        end
    end
    Y0 = reshape(dm',N*T,1);
    clear Y
    Y = Y0;
    XM((j-1)*N*T+1:j*N*T,:) = X_lagged;
    YM((j-1)*N*T+1:j*N*T,1) = Y;
end
return;
% Open file for writing
fileID1 = fopen('inputMCXG3_het_CO2.txt', 'w');
fileID2 = fopen('inputMCYG3_het_CO2.txt', 'w');

for sth = 1:T*N*MC_sim
    fprintf(fileID1, '%d ', XM(sth, :));
    fprintf(fileID1, '\n');
    fprintf(fileID2, '%d ', YM(sth, :));
    fprintf(fileID2, '\n');
end