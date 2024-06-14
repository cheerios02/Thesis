% Load the optimal group assignments and necessary data for all simulations 
load('inputMCXG3_het_CO2.txt');
load('inputMCYG3_het_CO2.txt');
load('assign_G3_Het_C02.txt');

% Define the number of periods, variables, replications, countries, and groups
T = 19;
Var = 8;
N = 21;
repNum = 500;
optGroup = assign_G3_Het_C02';
G = 3;

% Initialize variables
thetas = zeros(repNum,G*Var);

% Initiate the simulations
for sim = 1:repNum
    X = inputMCXG3_het_CO2((sim-1)*N*T+1:sim*N*T,:); % Obtain the data for X for all countries and periods in the current simulation
    Y = inputMCYG3_het_CO2((sim-1)*N*T+1:sim*N*T); % Obtain the data for Y for all countries and periods in the current simulation
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
    
    denominator=zeros(Var,Var, G);
    numerator=zeros(Var,G);
    theta_aux = zeros(Var,G);

    for g = 1:G
        for i=1:N
            if opt_group_assign(i) == g
                for t=1:T
                    denominator(:,:,g) = denominator(:,:,g)+(X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(X((i-1)*T+t,:)-X_group_aver((i-1)*T+t,:));
                    numerator(:,g) = numerator(:,g) +(X((i-1)*T+t,:)'-X_group_aver((i-1)*T+t,:)')*(Y((i-1)*T+t,:)-Y_group_aver((i-1)*T+t,:));
                end
            end
        end
        [m, n] = size(denominator(:,:,g));
        if rank(denominator(:,:,g)) < min(m,n)
            return;
        end
        theta_aux(:,g) = denominator(:,:,g) \ numerator(:,g);
        gitot = kron(which_group,eye(T));
        thetas(sim,:) = reshape(theta_aux,1,Var*G); % Obtain the estimates of the model parameters for each simulation
    end
end
    %% Bias:
    disp('The means of the thetas across all simulations are are:')
    mean_theta = reshape(mean(thetas), [Var,G]);
    disp(mean_theta)

    %% Standard Deviation
    disp('The standard deviation for theta across all simulations is:')
    std_theta = reshape(std(thetas), [Var,G]);
    disp(std_theta)