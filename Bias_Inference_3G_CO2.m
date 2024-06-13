% Load the optimal group assignments and necessary data for all simulations 
load('inputMCXG3_new_CO2.txt');
load('inputMCYG3_new_CO2.txt');
load('BigG_perm_G3_CO2');

% Define the number of periods, variables, replications, countries, and groups
T = 19;
Var = 8;
N = 21;
repNum = 500;
optGroup = BigG_perm;
G = 3;

% Initialize variables
thetas = zeros(repNum,Var);
std_cluster_vect = zeros(repNum,G*T+Var);
std_theta_inc_vect = zeros(repNum);

% Initiate the simulations
for sim = 1:repNum
    X = inputMCXG3_new_CO2((sim-1)*N*T+1:sim*N*T,:); % Obtain the data for X for all countries and periods in the current simulation
    Y = inputMCYG3_new_CO2((sim-1)*N*T+1:sim*N*T); % Obtain the data for Y for all countries and periods in the current simulation
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
end
%% Bias:
disp('The means of the thetas across all simulations are are:')
[mean(thetas)]

%% Standard Deviation
disp('The standard deviation for theta across all simulations is:')
[std(thetas)]