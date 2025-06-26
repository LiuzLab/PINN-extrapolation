import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import os
from math import pi

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)
pi = tf.constant(np.pi, dtype=DTYPE)
viscosity = .01/pi

class PINN_NeuralNet(tf.keras.Model):
    def __init__(self, lb, ub,
            output_dim=1,
            num_hidden_layers=6,
            num_neurons_per_layer=32,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub

        n_af = 3

        # Initialize a list of coefficient sets for each layer
        # Including randomly initialized a and b for each tanh(ax+b)
        self.coeffs = {
            'a': tf.Variable(tf.random.normal([n_af]), trainable=True, dtype=tf.float32),
            'b': tf.Variable(tf.random.normal([n_af]), trainable=True, dtype=tf.float32),
            'c': tf.Variable(tf.random.normal([n_af]), trainable=True, dtype=tf.float32)  # Pre-softmax weights
        }

        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        

        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                        activation='tanh',
                        kernel_initializer=kernel_initializer)
                for _ in range(self.num_hidden_layers - 1)]
        
        # The last hidden layer uses the custom activation function
        self.hidden.append(tf.keras.layers.Dense(
            num_neurons_per_layer,
            activation=lambda x: self.lc_tanh(x, n_af),
            kernel_initializer=kernel_initializer
        ))

        self.out = tf.keras.layers.Dense(output_dim)

    def lc_tanh(self, x, n_af):
        exp_c_sum = tf.reduce_sum(tf.exp(self.coeffs['c']))
        normalized_c = self.coeffs['c'] / exp_c_sum

        terms = [normalized_c[i] * tf.tanh(self.coeffs['a'][i] * x + self.coeffs['b'][i]) for i in range(n_af)]
        return tf.reduce_sum(terms, axis=0)

    def call(self, X):
        Z = self.scale(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)
    
class PINNSolver():
    def __init__(self, model, X_r, X_r_val):
        self.model = model

        # Store collocation points for training
        self.t = X_r[:, 0:1]
        self.x = X_r[:, 1:2]

        # Store collocation points for validation
        self.t_val = X_r_val[:, 0:1]
        self.x_val = X_r_val[:, 1:2]

        # Initialize global iteration counter
        self.iter = 0

    def get_r(self, t, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(x)

            u = self.model(tf.stack([t[:,0], x[:,0]], axis=1))

            u_x = tape.gradient(u, x)
        
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)

        del tape

        return self.fun_r(t, x, u, u_t, u_x, u_xx)

    def loss_fn(self, t, x):
        r = self.get_r(t, x)
        phi_r = tf.reduce_mean(tf.square(r))
        return phi_r

    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_fn(self.t, self.x)

        g = tape.gradient(loss, self.model.trainable_variables)
        del tape

        return loss, g
    
    def val_loss(self):
        return self.loss_fn(self.t_val, self.x_val).numpy()

    def fun_r(self, t, x, u, u_t, u_x, u_xx):
        pi = np.pi
        term1 = u_t * (t - t * x**2)
        term2 = u_x * (-tf.sin(pi * x) * t * (1 - x**2) + 0.04 / pi * t * x)
        term3 = u * u_x * t**2 * (1 - x**2)**2
        term4 = -u_xx * 0.01 / pi * t * (1 - x**2)
        term5 = u**2 * t**2 * (-2 * x) * (1 - x**2)
        term6 = u * (1 - x**2 + tf.sin(pi * x) * 2 * x * t - t * (1 - x**2) * tf.cos(pi * x) * pi + 0.02 / pi * t)
        term7 = -0.01 / pi * tf.sin(pi * x) * pi**2
        term8 = tf.sin(pi * x) * tf.cos(pi * x) * pi

        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

    
    def solve_for_TL(self, optimizer, N=1001, tol=1e-2):
        @tf.function
        def train_step():
            loss, grad_theta = self.get_grad()
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss

        for i in range(N):
            loss = train_step()

            val_loss = self.val_loss()

            self.current_loss = loss.numpy()
            self.callback()

    def solve_with_ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        def get_weight_tensor():
            weight_list = []
            shape_list = []

            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())

            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()

        def set_weight_tensor(weight_list):
            idx = 0
            for v in self.model.variables:
                vs = v.shape

                if len(vs) == 2:
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw

                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]

                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1

                v.assign(tf.cast(new_val, DTYPE))

        def get_loss_and_grad(w):

            set_weight_tensor(w)
            loss, grad = self.get_grad()

            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss

            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())

            grad_flat = np.array(grad_flat,dtype=np.float64)

            return loss, grad_flat


        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)

    def callback(self, xr=None):
        val_loss = self.val_loss()
        if self.iter % 50 == 0:
            print(f'It {self.iter:05d}: loss = {self.current_loss:10.8e} val_loss = {val_loss}')
        self.iter += 1

    def plot_solution(self, t_fixed):        
        def read_mat_file(t_fixed):
            t_fixed_str = f"{t_fixed:.3f}".replace('.', '_')                
            file_name = f"../burgers_data/solution_t_{t_fixed_str}.mat"
            
            mat_contents = scipy.io.loadmat(file_name)
            
            u_final = mat_contents['u_final']
            x = mat_contents['x']
            
            return u_final, x

        u_matrix, x = read_mat_file(t_fixed)
        x = x.squeeze()

        N = len(x)
        xspace = np.linspace(-1, 1, N) 
        
        tspace = np.full_like(xspace, t_fixed)
        
        Xgrid = np.vstack([tspace, xspace]).T
        
        # Predict the solution using the model
        upred = self.model(tf.cast(Xgrid, DTYPE))
        U = upred.numpy().flatten()

        # Apply the modified formula
        U = U * t_fixed * (1. - xspace**2)- np.sin(pi * xspace)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xspace, U, 'r--', label='Prediction')
        ax.plot(x, u_matrix[-1], label='Reference', color='blue')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title(f't = {t_fixed}')
        ax.legend()

        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-0.1, 1.1])

        script_name = os.path.splitext(os.path.basename(__file__))[0]
        file_name = f"{script_name}_solution_t_{t_fixed:.3f}"
        counter = 1
        pdf_file = f"{file_name}_{counter}.pdf"
        while os.path.exists(pdf_file):
            counter += 1
            pdf_file = f"{file_name}_{counter}.pdf"
        
        fig.savefig(pdf_file, format='pdf')
        plt.close(fig)
        print(f"Saved plot as {pdf_file}")

        return ax
    
    def compute_errors(self, t_start, t_end):

        def read_mat_file(t_fixed):
            t_fixed_str = f"{t_fixed:.3f}".replace('.', '_')
            file_name = f"../burgers_data/solution_t_{t_fixed_str}.mat"
            
            mat_contents = scipy.io.loadmat(file_name)
            
            u_final = mat_contents['u_final']
            x = mat_contents['x']
            
            # Return the extracted data
            return u_final, x

        sum_of_squares = 0 
        sum_of_absolute = 0 
        total_points = 0 

        t_fixed = t_start
        while t_fixed <= t_end:
            # Load data for the current t_fixed
            u_matrix, x = read_mat_file(t_fixed)
            x = x.squeeze()

            N = len(x)
            xspace = np.linspace(-1, 1, N)
            
            tspace = np.full_like(xspace, t_fixed)
            
            Xgrid = np.vstack([tspace, xspace]).T
            
            # Predict the solution using the model
            upred = self.model(tf.cast(Xgrid, DTYPE))
            U = upred.numpy().flatten()

            # Apply the modified formula
            U = U * t_fixed * (1. - xspace**2) - np.sin(pi * xspace)

            difference = U - u_matrix
            
            sum_of_squares += np.sum(difference**2)
            sum_of_absolute += np.sum(np.abs(difference))
            total_points += N

            t_fixed = round(t_fixed + 0.01, 2)

        # Normalize the accumulated errors at the end
        normalized_l2_error = np.sqrt(sum_of_squares / total_points)
        average_mae = sum_of_absolute / total_points
        
        return normalized_l2_error, average_mae
    

# Below are the functions for the TL
    
def freeze_all_but_last(model):
    for layer in model.layers[:-1]: 
        layer.trainable = False
    return model


def add_l2_regularizer_to_last_layer(model, l2_lambda=0.01):
    last_layer = model.layers[-1]
    
    last_layer.kernel_regularizer = tf.keras.regularizers.l2(l2_lambda)
    
    model.compile()
    
    return model


def select_high_loss_points(model, X, N_select=80):
    t = X[:, 0:1]
    x = X[:, 1:2]

    # Compute the residuals for all points
    r = Solver.get_r(t, x)

    # Compute the squared loss for each point (mean squared residual)
    losses = tf.reduce_mean(tf.square(r), axis=1)

    # Sort the losses in descending order and get the top N_select indices
    top_indices = tf.argsort(losses, direction='DESCENDING')[:N_select]

    # Select the points with the highest loss
    X_high_loss = tf.gather(X, top_indices)

    return X_high_loss

# -------------------------------------

N_r = 8000

tmin = 0.0
tmax = 0.5
tmin_val = tmax+0.01
tmax_val = 0.8

# Specify boundaries
lb = tf.constant([tmin, -1.], dtype=DTYPE)
ub = tf.constant([tmax, 1.], dtype=DTYPE)

# Collocation points for training
t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([t_r, x_r], axis=1)

# Collocation points for validation
t_r_val = tf.random.uniform((N_r,1), tmin_val, tmax_val, dtype=DTYPE)
x_r_val = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r_val = tf.concat([t_r_val, x_r_val], axis=1)

# Initialize model
model = PINN_NeuralNet(lb, ub)
model.build(input_shape=(None,2))

# Initilize PINN solver
Solver = PINNSolver(model, X_r, X_r_val)

Solver.solve_with_ScipyOptimizer(method="BFGS", options={'maxiter': 20000, 'gtol': 1e-6, 'disp': True})

Solver.plot_solution(0.82)
Solver.plot_solution(0.99)

errors = []
new_errors = Solver.compute_errors(0.8,0.99)
errors.append(new_errors)

print("L2 error = " + str(new_errors[0]))
print("MAE = ", str(new_errors[1]))

# After training, print the coefficients a, b, c
print("Coefficients a:", model.coeffs['a'].numpy())
print("Coefficients b:", model.coeffs['b'].numpy())
print("Coefficients c:", model.coeffs['c'].numpy())


input_shape = [2] 
l2_lambda = 0.001
model = add_l2_regularizer_to_last_layer(model, l2_lambda=l2_lambda)
freeze_all_but_last(model)

# Forward pass
N_r_extra = 4000
t_r_extra = tf.random.uniform((N_r_extra, 1), 0.0, tmax_val, dtype=DTYPE)
x_r_extra = tf.random.uniform((N_r_extra, 1), lb[1], ub[1], dtype=DTYPE)
X_r_extra = tf.concat([t_r_extra, x_r_extra], axis=1)

# Use the function to select the points with the highest loss
X_r_selected = select_high_loss_points(model, X_r_extra, N_select=80)

Solver_new = PINNSolver(model, X_r_extra, X_r_val)

new_layer_optim = tf.keras.optimizers.Adam(learning_rate=5e-2)

Solver_new.solve_for_TL(new_layer_optim, N=151)

Solver_new.plot_solution(0.82)
Solver_new.plot_solution(0.99)

print("L2 error = " + str(Solver.compute_errors(0.8,0.99)[0]))
print("MAE " + str(Solver.compute_errors(0.8,0.99)[1]))


# After training, print the coefficients a, b, c
print("Coefficients a:", model.coeffs['a'].numpy())
print("Coefficients b:", model.coeffs['b'].numpy())
print("Coefficients c:", model.coeffs['c'].numpy())
