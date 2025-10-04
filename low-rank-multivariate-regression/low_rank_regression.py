import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_args():
    """
    Parse command-line arguments for the experiment setup.
    
    Returns:
        argparse.Namespace: Parsed arguments including dimensions, rank,
                            dataset sizes, noise parameters, and training settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2, help="Rank of the weight matrix")
    parser.add_argument("--in_dim", type=int, default=10, help="Input dimension")
    parser.add_argument("--out_dim", type=int, default=10, help="Output dimension")
    parser.add_argument("--train_sizes", type=int, nargs='+', default=list(range(10, 101, 10)), help="List of training sizes")
    parser.add_argument("--test_size", type=int, default=500, help="Test size")
    parser.add_argument("--reps", type=int, default=50, help="Number of repetitions")
    parser.add_argument("--mu", type=float, default=0, help="Mean of the normal distribution")
    parser.add_argument("--sigma_1", type=float, default=1, help="Standard deviation of the normal distribution for weights")
    parser.add_argument("--sigma_2", type=float, default=0.1, help="Standard deviation of the normal distribution for noise")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Threshold for convergence")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations")
    args = parser.parse_args()

    return args

def generate_dataset(args):
    """
    Generate synthetic training and test datasets using a low-rank matrix factorization model.
    
    Args:
        args (argparse.Namespace): Experiment parameters.
    
    Returns:
        tuple: Lists of training inputs (X_trains), test input (X_test),
               training outputs (y_trains), and test output (y_test).
    """
    np.random.seed(args.seed)

    in_dim = args.in_dim
    out_dim = args.out_dim
    rank = args.rank
    test_size = args.test_size
    train_sizes = args.train_sizes
    reps = args.reps
    mu = args.mu
    sigma_1 = args.sigma_1
    sigma_2 = args.sigma_2

    # True weight matrices
    A = np.random.normal(mu, sigma_1, (in_dim, rank))
    B = np.random.normal(mu, sigma_1, (rank, out_dim))

    # Training datasets for each training size
    eps_trains = [np.random.normal(mu, sigma_2, (reps, out_dim, size)) for size in train_sizes]
    X_trains = [np.random.normal(mu, sigma_1, (reps, in_dim, size)) for size in train_sizes]
    y_trains = [B.T @ A.T @ X_train + eps_train for X_train, eps_train in zip(X_trains, eps_trains)]

    # Test dataset
    eps_test = np.random.normal(mu, sigma_1, (reps, out_dim, test_size))
    X_test = np.random.normal(mu, sigma_1, (reps, in_dim, test_size))
    y_test = B.T @ A.T @ X_test + eps_test

    return X_trains, X_test, y_trains, y_test

def rmse(y_pred, y_test):
    """
    Compute the Root Mean Square Error (RMSE) between predictions and ground truth.
    
    Args:
        y_pred (np.ndarray): Predicted outputs, shape (reps, out_dim, test_size).
        y_test (np.ndarray): True outputs, shape (reps, out_dim, test_size).
    
    Returns:
        np.ndarray: RMSE values per repetition.
    """
    mse = np.mean((y_pred - y_test) ** 2, axis=(1, 2))
    return np.sqrt(mse)

class LR():
    """
    Standard Linear Regression model (solves least squares).
    """
    def __init__(self):
        pass

    def __repr__(self):
        return "Linear Regression"

    def train(self, X, Y):
        """
        Train linear regression using closed-form least squares.
        """
        self.W = np.linalg.inv(X @ np.transpose(X, (0, 2, 1))) @ X @ np.transpose(Y, axes=(0, 2, 1))
        return self.W

    def predict(self, X):
        """
        Predict outputs given inputs X using learned weights W.
        """
        return np.transpose(self.W, (0, 2, 1)) @ X

class ALS():
    """
    Alternating Least Squares (ALS) for low-rank regression.
    """
    def __init__(self, mu, sigma, in_dim, out_dim, rank, threshold, max_iter, reps):
        self.A = np.random.normal(mu, sigma, (reps, in_dim, rank))
        self.B = np.random.normal(mu, sigma, (reps, rank, out_dim))
        self.threshold = threshold
        self.max_iter = max_iter
    
    def __repr__(self):
        return f"Alternating Least Squares (rank={self.A.shape[2]})"
    
    def train(self, X, Y):
        """
        Train ALS by alternating optimization of factors A and B until convergence.
        """
        buffer_A, buffer_B = [], []
        for x, y, a, b in zip(X, Y, self.A, self.B):
            
            prev_loss = float('inf')

            for _ in range(self.max_iter):
                # Update A and B alternately
                a = np.linalg.inv(x @ x.T) @ x @ y.T @ b.T @ np.linalg.inv(b @ b.T)
                b = np.linalg.inv(a.T @ x @ x.T @ a) @ a.T @ x @ y.T

                # Compute reconstruction loss
                y_pred = b.T @ a.T @ x
                loss = np.mean((y - y_pred) ** 2)

                # Check convergence
                if abs(prev_loss - loss) / prev_loss < self.threshold:
                    break

                prev_loss = loss

            buffer_A.append(a)
            buffer_B.append(b)
        
        self.A = np.array(buffer_A)
        self.B = np.array(buffer_B)

        return self.A, self.B

    def predict(self, X):
        """
        Predict outputs using learned A and B factors.
        """
        return np.transpose(self.B, axes=(0, 2, 1)) @ np.transpose(self.A, axes=(0, 2, 1)) @ X
    
class LRR(LR):
    """
    Low-Rank Regression model: 
    Performs regression and projects onto a rank-constrained subspace.
    Inherits from LR.
    """
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
    
    def __repr__(self):
        return f"Low Rank Regression (rank={self.rank})"

    def train(self, X, Y):
        """
        Train low-rank regression by truncating singular values after standard LR training.
        """
        self.W = super().train(X, Y)
        U, S, Vt = np.linalg.svd(np.transpose(X, (0, 2, 1)) @ self.W, full_matrices=False)
        Vt_truncated = Vt[:, :self.rank, :]
        P = np.transpose(Vt_truncated, (0, 2, 1)) @ Vt_truncated
        self.W = self.W @ P
        return self.W
    
    def predict(self, X):
        """
        Predict outputs using rank-constrained weights.
        """
        return super().predict(X)

def main():
    """
    Main routine:
    1. Generate synthetic dataset.
    2. Train Linear Regression (LR), Low-Rank Regression (LRR), and ALS models.
    3. Compute average losses for each model across different training sizes.
    4. Plot Average Loss vs Training Size for comparison.
    """
    args = get_args()
    X_trains, X_test, y_trains, y_test = generate_dataset(args)

    # Initialize models
    lr = LR()
    lrr = LRR(args.rank)
    als = ALS(args.mu, args.sigma_1, args.in_dim, args.out_dim, args.rank, args.threshold, args.max_iter, args.reps)
    avg_losses = {}

    models = [lr, lrr, als]

    # Train each model and compute average loss
    for model in models:
        avg_losses[model] = []
        for X_train, y_train in zip(X_trains, y_trains):
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            loss = rmse(y_pred, y_test)
            avg_loss = np.mean(loss)
            avg_losses[model].append(avg_loss)
        avg_losses[model] = np.array(avg_losses[model])

    # Plot results
    styles = ['o-', 's--', '^-.' ]
    for model, losses, style in zip(avg_losses.keys(), avg_losses.values(), styles):
        name = str(model)
        plt.plot(args.train_sizes, losses, style, label=name)

    plt.xlabel("Train size")                        
    plt.ylabel("Average Loss")                      
    plt.title("Average Loss vs Train Size")         
    plt.legend()                                    
    plt.grid(True)                                  
    plt.show()                                      

if __name__ == '__main__':
    main()
