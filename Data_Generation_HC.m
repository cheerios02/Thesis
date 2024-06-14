load('final_data');

% Define the variables, number of countries, number of groups etc.
X = ldem_linc(:,1:2);
Y = dem;
T = 7;
Var = 2;
N = size(Y,1)/T;
G = 3;

% The optimal group assignment for each of the 90 countries
if G==3
    opt_group_assign = [
        3 2 2 1 2 3 2 3 3 3 2 2 3 1 1 1 1 2 2 2 2 2 1 3 1 2 2 2 1 2 3 1 3 2 3 2 2 2 1 2 2 2 2 3 2 1 3 2 2 2 3 3 3 1 3 2 3 1 3 2 2 2 2 3 3 3 2 2 3 1 1 1 2 2 1 1 1 2 2 1 2 3 3 1 3 2 2 3 1 3
        ];
end

if G==5
    opt_group_assign = [
        1 3 3 2 3 4 3 1 1 4 3 3 3 2 2 2 2 3 3 3 3 3 2 5 2 3 3 3 2 3 5 2 1 3 1 3 3 3 2 3 3 3 3 2 3 2 1 3 3 3 4 5 4 2 4 3 4 5 3 3 3 3 3 5 5 1 3 3 4 2 2 2 1 3 2 2 2 1 3 2 1 3 4 2 1 3 3 4 2 4
        ];
end


if G==10
    opt_group_assign = [
        3 7 7 1 7 9 4 8 10 9 7 7 2 1 1 1 1 7 7 7 7 7 1 5 1 5 7 7 1 7 6 1 3 4 5 4 7 7 1 7 7 7 7 10 7 1 10 7 7 10 9 10 9 1 9 4 9 6 10 7 7 5 7 9 5 10 5 10 9 1 1 4 8 7 1 1 1 3 7 1 8 10 10 1 8 7 7 9 1 9
        ]';
end


which_group=zeros(N,G);

for g=1:G
    which_group(:,g)=(opt_group_assign==g); % Stores the optimal group of the country in that simulation
end

Y_group_aver=zeros(N*T,1);
X_group_aver=zeros(N*T,Var);

MYbar_gt=zeros(G*T,1);
MXbar_gt=zeros(G*T,Var);

country_sum_per_group=sum(which_group); % Obtain the sum of total countries in each group in that simulation

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
thetas = zeros(Var,G);

for g = 1:G
    for i = 1:N
        if opt_group_assign(i) == g
            for t = 1:T
                denominator(:,:,g) = denominator(:,:,g) + (X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(X((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:));
                numerator(:,g) = numerator(:,g) + (X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(Y((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:));
                MYbar_gt((g-1)*T+t) = Y_group_aver((i-1)*T+t);
                MXbar_gt((g-1)*T+t,:) = X_group_aver((i-1)*T+t,:);
            end
        end
    end
    thetas(:,g) = denominator(:,:,g) \ numerator(:,g); % Compute the thetas using an OLS regression
end

a = MYbar_gt - MXbar_gt * thetas;
gitot = kron(which_group,eye(T));
delta_hat = gitot * a;

obj = 0;
ei = zeros(N*T,1);
for i = 1:N
    for t = 1:T
        g = opt_group_assign(i);
        obj = obj + (Y((i-1)*T+t,:) - Y_group_aver((i-1)*T+t,:) - (X((i-1)*T+t,:) - X_group_aver((i-1)*T+t,:)) * thetas(:,g)).^2;
        ei((i-1)*T+t) = Y((i-1)*T+t,:) - Y_group_aver((i-1)*T+t,:) - (X((i-1)*T+t,:) - X_group_aver((i-1)*T+t,:)) * thetas(:,g);
    end
end

% Compute the variance of the residuals
sigma2_naive = obj / (N*T - G*T - N - Var);

montecarlo_replications = 500;
for j = 1:montecarlo_replications
    V = randn(N*T,1);
    err = sqrt(sigma2_naive) .* V / std(V);
    err = reshape(err,T,N)';
    dm = zeros(N,T+1);
    ym = zeros(N,T);
    Lag_dem = reshape(X(:,1),T,N)';
    Lag_inc = reshape(X(:,2),T,N)';
    for i = 1:N
        g = opt_group_assign(i);
        Rdelta = reshape(delta_hat(:,g),T,N)';
        dm(i,1) = Lag_dem(i,1);
        ym(i,:) = Lag_inc(i,:);
        for t = 2:T+1
            dm(i,t) = [dm(i,t-1), ym(i,t-1)] * thetas(:,g) + Rdelta(i,t-1) + err(i,t-1);
        end
    end

    Rdm = reshape(dm(:,2:T+1)',N*T,1);
    Rdm_1 = reshape(dm(:,1:T)',N*T,1);
    X0 = [Rdm_1 X(:,2)];
    Y0 = Rdm;
    clear X Y
    X = X0;
    Y = Y0;

    XM((j-1)*N*T+1:j*N*T,:) = X;
    YM((j-1)*N*T+1:j*N*T,1) = Y;
end

% Open file for writing
fileID1 = fopen('inputMCXG5_HC.txt', 'w');
fileID2 = fopen('inputMCYG5_HC.txt', 'w');

for sth = 1:T*N*montecarlo_replications
    fprintf(fileID1, '%d ', XM(sth, :));
    fprintf(fileID1, '\n');
    fprintf(fileID2, '%d ', YM(sth, :));
    fprintf(fileID2, '\n');
end

% Close files
fclose(fileID1);
fclose(fileID2);

