 % City placement by coordinates (Latitude/Longitude from google)
cities = [
    59.9139, 10.7522; % Oslo
    60.5337,  8.2088; % Geilo
    60.3913,  5.3221; % Bergen
    %58.9690,  5.7331; % Stavanger
    63.4305, 10.3951; % Trondheim
    69.6492, 18.9553; % Tromsø
    58.1599,  8.0182; % Kristiansand
    67.2802, 14.4040; % Bodø
    62.7372,  7.1608; % Molde
    62.4722,  6.1495; % Ålesund
    62.0758,  9.1280; % Dombås
    %61.1153, 10.4663  % Lillehammer
]; % I added a couple of extra cities to see how the PSO would behave

numCities = size(cities, 1); % Able to add more cities to the "cities"
                             % without changing anything else

% Distance matrix
distMatrix = pdist2(cities, cities, 'euclidean');

%% 
% PSO Parameters
swarmSize = 100;       
maxIter = 200;         
w = 0.99;               % Inertia weight
c1 = 2.0;              % Cognitive coefficient
c2 = 2.0;              % Social coefficient
wDamp = 0.99;          % Inertia damping ratio

%%
% Position matrix
positions = rand(swarmSize, numCities); 
velocities = zeros(swarmSize, numCities);

% Initialize personal best
pbest_pos = positions;
pbest_cost = inf(swarmSize, 1);

% Calculate initial costs
for i = 1:swarmSize
    [~, route] = sort(positions(i,:));
    pbest_cost(i) = calculateCost(route, distMatrix);
end

% Initialize global best
[gbest_cost, idx] = min(pbest_cost);
gbest_pos = positions(idx, :);
convergence = zeros(maxIter, 1);

for iter = 1:maxIter
    r1 = rand(swarmSize, numCities);
    r2 = rand(swarmSize, numCities);
    velocities = w*velocities + ...
                 c1*r1.*(pbest_pos - positions) + ...
                 c2*r2.*(gbest_pos - positions);
    

    positions = positions + velocities;
    
    % Update costs and best positions
    for i = 1:swarmSize
        [~, route] = sort(positions(i,:));
        current_cost = calculateCost(route, distMatrix);
        
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
    
    % Store convergence data
    convergence(iter) = gbest_cost;
    
    % Update inertia weight
    w = w * wDamp;
    
    % Progress
    if mod(iter, 50) == 0
        fprintf('Iteration %d, Best Cost: %.2f\n', iter, gbest_cost);
    end
end


% Figures


figure;
plot(convergence, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
title('PSO Convergence');
grid on;


[~, best_route] = sort(gbest_pos);
best_route = [best_route, best_route(1)]; % Return to start

figure;
plot(cities(:,2), cities(:,1), 'o', 'MarkerSize', 10);
hold on;
plot(cities(best_route,2), cities(best_route,1), 'r-', 'LineWidth', 1.5);
text(cities(:,2), cities(:,1), num2str((1:numCities)'), 'FontSize', 12);
xlabel('Longitude');
ylabel('Latitude');
title('Optimized Route');
axis equal;

%% 
% Cost function
function cost = calculateCost(route, distMatrix)
    n = length(route);
    cost = 0;
    for i = 1:n-1
        cost = cost + distMatrix(route(i), route(i+1));
    end
    cost = cost + distMatrix(route(n), route(1)); % Return to start
end

% Inspiration from "PSO_Basic.m" From blackboard, and chat.deepseek.com