from sklearn.tree import DecisionTreeRegressor
from misc import load_data, run_experiment

def main():
    # Load data using misc.py function
    df = load_data()
    
    # Initialize Decision Tree model
    dt_model = DecisionTreeRegressor(
        random_state=42,
        max_depth=8,
        min_samples_split=5
    )
    
    # Run experiment using generic functions from misc.py
    mse, model = run_experiment(dt_model, df, "Decision Tree Regressor")
    
    # Display average MSE score
    print(f"\n{'='*50}")
    print(f"Decision Tree Regressor - Final MSE: {mse:.4f}")
    print(f"{'='*50}")
    
    return mse

if __name__ == "__main__":
    main()
