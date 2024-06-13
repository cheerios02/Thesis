% Load the necessary datasets and set the number of countries and
% simulations
load('assign_G3_CO2.txt');
BigG = assign_G3_CO2';
N = 21;
repNum = 500;
G = 3;

opt_group_assign= [3 3 1 2 1 3 3 1 1 2 2 1 1 2 1 2 3 3 1 3 2]';

% Denote all possible permutations
diff_perm_num=60; % There are 3! possible permutations. We select a sufficiently high number so we are confident all permutations are included
permutations = zeros(G,diff_perm_num);
for i = 1:diff_perm_num
        permutations(:,i) = randperm(G)';
end
BigG_perm = zeros(N,repNum);

obj_value = zeros(diff_perm_num,1);
for i = 1:repNum
     % Store the optimal group allocation for the current simulation
     group_col = BigG(:,i);
    for j = 1:diff_perm_num
       % Reorder the group allocation according to the j-th permutation and calculate the squared error
       groups_reordered = permutations(group_col,j);
       obj_value(j,1) = sum((opt_group_assign - groups_reordered).^2);
    end
    % Obtain the relabelling of the groups with the smallest deviation for the current simulation and store it 
    [min_error,min_error_pos] = min(obj_value);
    BigG_perm(:,i) = permutations(group_col,min_error_pos);
end

% Compute the misclassification probability
v = BigG_perm - kron(opt_group_assign,ones(1,repNum));
missclas_prob = 1 - mean(mean(v==0));
disp('The misclassification probability for 3 groups is:')
missclas_prob

save('BigG_perm_G3_CO2.mat', 'BigG_perm');
