% Load the necessary datasets and set the number of countries and
% simulations
load('assignment_G10_HC.txt');
BigG = assignment_G10_HC';
N = 90;
repNum = 500;
G = 10;

opt_group_assign = [
    3 7 7 1 7 9 4 8 10 9 7 7 2 1 1 1 1 7 7 7 7 7 1 5 1 5 7 7 1 7 6 1 3 4 5 4 7 7 1 7 7 7 7 10 7 1 10 7 7 10 9 10 9 1 9 4 9 6 10 7 7 5 7 9 5 10 5 10 9 1 1 4 8 7 1 1 1 3 7 1 8 10 10 1 8 7 7 9 1 9
    ];

% Denote all possible permutations
% Denote all possible permutations
diff_perm_num=500000; % There are 10! possible permutations. We select 500.000 iterations due to computational constraints
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
       % Reorder the group allocation according to the j-th permutation and
       % calculate the squared error
       groups_reordered = permutations(group_col,j);
       obj_value(j,1) = sum((opt_group_assign - groups_reordered).^2);
    end
    % Obtain the relabelling of the groups with the smallest deviation for the current simulation and
    % store it 
    [min_error,min_error_pos] = min(obj_value);
    BigG_perm(:,i) = permutations(group_col,min_error_pos);
end

% Compute the misclassification probability
v = BigG_perm - kron(opt_group_assign,ones(1,repNum));
missclas_prob = 1 - mean(mean(v==0));
disp('The misclassification probability for 10 groups is:')
missclas_prob

save('BigG_perm_G10_HC.mat', 'BigG_perm')