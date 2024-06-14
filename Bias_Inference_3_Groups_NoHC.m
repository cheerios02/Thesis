% Load the optimal group assignments and necessary data for all simulations 
load('inputMCXG3_dgp1.txt');
load('inputMCYG3_dgp1.txt');
load('BigG_perm_G3_dgp1');

% Define the number of periods, variables, replications, countries, and groups
T = 7;
Var = 2;
N = 90;
repNum = 1000;
optGroup = BigG_perm;
G = 3;

% Initialize variables
thetas = zeros(repNum,Var);
std_cluster_vect = zeros(repNum,G*T+Var);
std_theta_inc_vect = zeros(repNum);

% Initiate the simulations
for sim = 1:repNum
    X = inputMCXG3_dgp1((sim-1)*N*T+1:sim*N*T,:); % Obtain the data for X for all countries and periods in the current simulation
    Y = inputMCYG3_dgp1((sim-1)*N*T+1:sim*N*T); % Obtain the data for Y for all countries and periods in the current simulation
    opt_group_assign = optGroup(:,sim); % Obtain the optimal group assignment for all countries in the current simulation
    which_group=zeros(N,G);
    
    for g=1:G
        which_group(:,g)=(opt_group_assign==g); % Stores the optimal group of each country in the current simulation
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
    
    denominator=zeros(Var,Var);
    numerator=zeros(Var,1);
    
    for i=1:N
        for t=1:T
            denominator=denominator+(X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(X((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:));
            numerator=numerator+(X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(Y((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:));
        end
    end
    
    theta_par=denominator\numerator; % Compute the thetas using an OLS regression
    gitot = kron(which_group,eye(T));
    thetas(sim,:) = theta_par'; % Obtain the estimates of the model parameters for each simulation

    % Related to the large-T clustered variance standard error approach
    ei=Y-Y_group_aver-(X-X_group_aver)*theta_par;
    Rei = reshape(ei,T,N)';
    Omega = zeros(N*T,N*T);
    Mi = zeros(T,T);
    for i = 1:N
        Mi = Rei(i,:)'*Rei(i,:);
        Omega((i-1)*T+1:i*T,(i-1)*T+1:i*T) = Mi;
    end
    
    % Related to the large-T clustered variance standard error approach
    gitot=kron(which_group,eye(T));
    Xtot=[gitot X];
    V = inv(Xtot'*Xtot)*Xtot'*Omega*Xtot*inv(Xtot'*Xtot);
    V=V*N*T/(N*T-G*T-Var);
    std_cluster = sqrt(diag(V));
    std_cluster_vect(sim,:) = std_cluster';

    % Related to the large-T clustered variance standard error approach
    % Total income effect: standard errors
    Mat_inc=[theta_par(2)/(1-theta_par(1))^2 1/(1-theta_par(1))];
    Var_theta_inc=Mat_inc*V(G*T+1:G*T+Var,G*T+1:G*T+Var)*Mat_inc';
    std_theta_inc=sqrt(Var_theta_inc);
    std_theta_inc_vect(sim) = std_theta_inc;
end
return;
%% Bias:
disp('The bias results across all simulations are are:')
[mean(thetas),mean(thetas(:,2)./(1-thetas(:,1)))]

%% Standard Deviation
disp('The standard deviation for theta across all simulations is:')
[std(thetas),std(thetas(:,2)./(1-thetas(:,1)))]
disp('The medians using the large-T clustered variance standard error approach across all simulations are:');
median([std_cluster_vect(:,G*T+1:G*T+Var),std_theta_inc_vect(:,1)])

%% Coverage
asympt_std = std_cluster_vect(:,G*T+1:G*T+Var);
coverage_probs = zeros(Var+1,1);
true_thetas = [0.406464039646224;0.0894185088297439];

for count = 1:repNum
    for k = 1:Var
        if (thetas(count,k)-1.96*asympt_std(count,k) < true_thetas(k,1)) && (thetas(count,k)+1.96*asympt_std(count,k)>true_thetas(k,1))
            coverage_probs(k,1) = coverage_probs(k,1) + 1/repNum;
        end
    end
end

std_cum_inc = std_theta_inc_vect(:,1);
for count = 1:repNum
    if (thetas(count,2)/(1-thetas(count,1))-1.96*std_cum_inc(count,1) < true_thetas(2,1)/(1-true_thetas(1,1)) && (thetas(count,2)/(1-thetas(count,1))+1.96*std_cum_inc(count,1)>(true_thetas(2,1)/(1-true_thetas(1,1)))))
        coverage_probs(3,1) = coverage_probs(3,1) + 1/repNum;
    end
end

disp('The empirical coverage probabilities (at a 5% level) across all simulations are: ')
coverage_probs