from sklearn.kernel_ridge import KernelRidge
from misc import load_data, run_experiment

def main():
    # Load data using the same misc.py function
    df = load_data()
    
    # Initialize Kernel Ridge model
    kr_model = KernelRidge(
        alpha=1.0,
        kernel='rbf',
        gamma=None
    )
    
    # Run experiment using the same generic functions from misc.py
    mse, model = run_experiment(kr_model, df, "Kernel Ridge Regressor")
    
    # Display average MSE score
    print(f"\n{'='*50}")
    print(f"Kernel Ridge Regressor - Final MSE: {mse:.4f}")
    print(f"{'='*50}")
    
    return mse

if __name__ == "__main__":
    main()
