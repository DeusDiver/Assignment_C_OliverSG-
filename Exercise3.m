clc; clear; close all;

% Load coordinates (10 taxis and 10 customers in Ålesund)
taxis = [
    62.4630, 6.1223; % Hessa taxi
    62.4682, 6.1132; % Nordvest Taxi
    62.4695, 6.1257; % Maxi Taxi
    62.4708, 6.1541; % Ålesund Rutebilstasjon
    62.4771, 6.1658; % Kenneths Taxi
    62.4716, 6.2124; % Per Henning Transport
    62.4746, 6.2211; % Marius Conradi
    62.5281, 6.1202; % Vigra Taxi
    62.4640, 6.2601; % William Waage Taxi
    62.4705, 6.2937; % Ålesund Taxi
];

customers = [
62.5598, 6.1145; % Vigra Lufthavn
62.4659, 6.0967; % Atlanterhavsparken
62.4704, 6.1463; % Quality Hotel Waterfront
62.4728, 6.1526; % Hotel Brosundet
62.4704, 6.1869; % Colorline Stadion
62.4729, 6.2357; % NTNU Ålesund
62.5078, 6.1036; % Ålesund Airport Hotel
62.4679, 6.3497; % AMFI Moa
62.4745, 6.1644; %Fjellstua
62.4638, 6.3127; % Ålesund Sjukehus
];

% Distance matrix
distMatrix = pdist2(taxis, customers, 'euclidean'); 
n = size(distMatrix, 1); % Number of taxis/customers

%%
% PSO Parameters
swarmSize = 100;       
maxIter = 200;         
w = 0.99;               % Inertia weight
c1 = 2.0;              % Cognitive coefficient
c2 = 2.0;              % Social coefficient
wDamp = 0.99;          % Inertia damping ratio

%% 
% Initialize Particles (Permutation Encoding)
positions = rand(swarmSize, n); % Each particle is a permutation vector
velocities = zeros(swarmSize, n);

% Initialize personal best
pbest_pos = positions;
pbest_cost = inf(swarmSize, 1);

% Calculate initial costs
for i = 1:swarmSize
    [~, perm] = sort(positions(i,:));
    pbest_cost(i) = calculateCost(perm, distMatrix);
end

% Initialize global best
[gbest_cost, idx] = min(pbest_cost);
gbest_pos = positions(idx, :);
convergence = zeros(maxIter, 1);


% PSO Main Loop
for iter = 1:maxIter
    r1 = rand(swarmSize, n);
    r2 = rand(swarmSize, n);
    velocities = w*velocities + ...
                 c1*r1.*(pbest_pos - positions) + ...
                 c2*r2.*(gbest_pos - positions);
    
   
    positions = positions + velocities;
    
 
    for i = 1:swarmSize
        [~, perm] = sort(positions(i,:)); 
        current_cost = calculateCost(perm, distMatrix);
        
        % Update personal best
        if current_cost < pbest_cost(i)
            pbest_pos(i,:) = positions(i,:);
            pbest_cost(i) = current_cost;
            
            % Update global best
            if current_cost < gbest_cost
                gbest_cost = current_cost;
                gbest_pos = positions(i,:);
            end
        end
    end
    
    % Track convergence
    convergence(iter) = gbest_cost;
    
    % Update inertia weight
    w = w * wDamp;
    
    % Progress
    if mod(iter, 1) == 0
        fprintf('Iteration %d, Best Cost: %.2f km\n', iter, gbest_cost);
    end
end


% Convergence plot
figure;
plot(convergence, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost (km)');
title('PSO Convergence');
grid on;

% Best assignment permutation
[~, best_perm] = sort(gbest_pos);

% Figures
figure;
plot(taxis(:,2), taxis(:,1), 's', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
hold on;
plot(customers(:,2), customers(:,1), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
for i = 1:n
    plot([taxis(i,2), customers(best_perm(i),2)], ...
         [taxis(i,1), customers(best_perm(i),1)], 'k--');
end
xlabel('Longitude');
ylabel('Latitude');
title('Optimal Taxi-Customer Assignments');
legend('Taxis', 'Customers', 'Assignments');
axis equal;

%% 
% Cost Function
function cost = calculateCost(perm, distMatrix)
    cost = sum(distMatrix(sub2ind(size(distMatrix), 1:length(perm), perm)));
end


% Some code is recycled from the previous Exercise, And chat.deepseek.com
% Has been used to help set up the PSO