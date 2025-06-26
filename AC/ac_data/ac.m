% Number of spatial points
N = 400;
% Spatial domain
x = linspace(-1, 1, N);
dx = x(2) - x(1);

% Initial condition
u0 = x.^2 .* cos(pi*x);

% Time span and steps
dt = 0.005;
t_final = 1.0;
tspan = 0:dt:t_final; % Array of time steps from 0 to t_final with step dt

% Derivative function
function dudt = pde(t, u, dx, N)
    % Enforce boundary conditions: u(t,-1) = u(t,1) = -1
    u(1) = -1;
    u(N) = -1;

    % Shift u for boundary conditions
    u_shift_right = [u(2:end); u(end)];
    u_shift_left = [u(1); u(1:end-1)];

    % Second derivative (centered difference for internal points)
    uxx = (u_shift_right - 2*u + u_shift_left) / (dx^2);

    % Non-linear term 5*u^3 - 5*u
    nonlinear_term = 5 * u.^3 - 5 * u;
    
    % PDE: u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0
    dudt = 0.0001 * uxx - nonlinear_term;

    % Enforce boundary conditions for dudt
    dudt(1) = 0;
    dudt(N) = 0;
end

% Solver options
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

% Solve the PDE at all specified times
[t, U] = ode15s(@(t, u) pde(t, u, dx, N), tspan, u0, opts);

% Save the results and plot
for k = 1:length(t)
    u_final = U(k, :); % Extract the solution at time t(k)
    filename = sprintf('solution_t_%1.3f', t(k));
    filename = strrep(filename, '.', '_'); % Replace dot with underscore
    save(filename, 'x', 'u_final');
end
