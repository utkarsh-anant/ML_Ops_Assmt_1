
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, run_experiment
import pandas as pd

def main():
    print("\n" + "="*60)
    print("MLOps Assignment 1 - Performance Comparison Report")
    print("="*60)
    
    df = load_data()
    
    print("\nTraining and evaluating models...")
    
    # Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42, max_depth=8, min_samples_split=5)
    dt_mse, _ = run_experiment(dt_model, df, "Decision Tree")
    
    # Kernel Ridge
    kr_model = KernelRidge(alpha=1.0, kernel='rbf', gamma=None)
    kr_mse, _ = run_experiment(kr_model, df, "Kernel Ridge")
    
    # Create comparison table
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    comparison_data = {
        'Model': ['Decision Tree', 'Kernel Ridge'],
        'MSE': [dt_mse, kr_mse]
    }
    
    df_compare = pd.DataFrame(comparison_data)
    print(df_compare.to_string(index=False))
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    if dt_mse < kr_mse:
        print("Decision Tree performs better (lower MSE)")
    else:
        print("Kernel Ridge performs better (lower MSE)")
    print(f"Performance difference: {abs(dt_mse - kr_mse):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
