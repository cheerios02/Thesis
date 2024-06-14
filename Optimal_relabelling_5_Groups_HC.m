% Load the necessary datasets and set the number of countries and
% simulations
load('assignment_G5_HC.txt');
BigG = assignment_G5_HC';
N = 90;
repNum = 500;
G = 5;

opt_group_assign = [
    1 3 3 2 3 4 3 1 1 4 3 3 3 2 2 2 2 3 3 3 3 3 2 5 2 3 3 3 2 3 5 2 1 3 1 3 3 3 2 3 3 3 3 2 3 2 1 3 3 3 4 5 4 2 4 3 4 5 3 3 3 3 3 5 5 1 3 3 4 2 2 2 1 3 2 2 2 1 3 2 1 3 4 2 1 3 3 4 2 4
    ]';

% Denote all possible permutations
% Denote all possible permutations
diff_perm_num=750; % There are 5! possible permutations. We select a sufficiently high number so we are confident all permutations are included
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
disp('The misclassification probability for 5 groups is:')
missclas_prob

save('BigG_perm_G5_HC.mat', 'BigG_perm')