#!/usr/bin/env python3
"""
Analyze residuals across MaxEnt iterations and extract corresponding epsilon matrices.

This script helps you:
1. Find iterations with the lowest residuals from state.json files
2. Understand the different epsilon file types
3. Extract the epsilon matrix corresponding to your best iterations

Usage:
    python analyze_residuals.py --run-root /path/to/your/maxent_run
"""

import argparse
import json
import numpy as np
from pathlib import Path
import pandas as pd
from utils import format_iter

def load_iteration_state(run_root: Path, iteration: int):
    """Load state.json for a given iteration."""
    state_path = run_root / format_iter(iteration) / "update" / "state.json"
    if not state_path.exists():
        return None
    
    with open(state_path, 'r') as f:
        return json.load(f)

def find_epsilon_files(iter_dir: Path):
    """Find all epsilon-related files in an iteration directory."""
    files = {}
    
    # params/epsilon.npy - The epsilon used for this iteration's simulations
    params_epsilon = iter_dir / "params" / "epsilon.npy"
    if params_epsilon.exists():
        files['params_epsilon'] = params_epsilon
    
    # update/epsilon_tk_*.npy - Versioned epsilon files from Newton updates
    update_dir = iter_dir / "update"
    if update_dir.exists():
        epsilon_tk_files = list(update_dir.glob("epsilon_tk_*.npy"))
        if epsilon_tk_files:
            files['epsilon_tk_files'] = sorted(epsilon_tk_files)
        
        # update/lambda_vec.npy - Upper triangle vectorized form of epsilon
        lambda_vec = update_dir / "lambda_vec.npy"
        if lambda_vec.exists():
            files['lambda_vec'] = lambda_vec
            
        # obs/epsilon_next.npy - The updated epsilon for next iteration
        epsilon_next = iter_dir / "obs" / "epsilon_next.npy"
        if epsilon_next.exists():
            files['epsilon_next'] = epsilon_next
    
    return files

def explain_epsilon_files():
    """Explain the different epsilon file types."""
    print("=== Epsilon File Types Explained ===\n")
    
    print("1. iter_XXX/params/epsilon.npy")
    print("   - The epsilon matrix used for simulations in this iteration")
    print("   - This is what gets loaded as the interaction matrix for MD runs")
    print("   - Copied from previous iteration's epsilon_next.npy\n")
    
    print("2. iter_XXX/update/epsilon_tk_N.npy")
    print("   - Versioned epsilon files created by Newton update process")
    print("   - For iteration i, creates epsilon_tk_{i+1}.npy")
    print("   - epsilon_tk_12.npy was created during iteration 11 → iteration 12")
    print("   - epsilon_tk_13.npy was created during iteration 12 → iteration 13")
    print("   - These represent the 'updated' epsilon for the NEXT iteration\n")
    
    print("3. iter_XXX/update/lambda_vec.npy")
    print("   - Upper triangle vectorized form of the epsilon matrix")
    print("   - Contains same information as epsilon matrix, just different format")
    print("   - Used internally for gradient computations and optimization\n")
    
    print("4. iter_XXX/obs/epsilon_next.npy (if present)")
    print("   - Intermediate file: updated epsilon ready for next iteration")
    print("   - Gets copied to next iteration's params/epsilon.npy\n")

def analyze_residuals(run_root: Path, top_n: int = 10):
    """Analyze residuals across all iterations and find the best ones."""
    run_root = Path(run_root)
    
    # Find all iteration directories
    iter_dirs = sorted([d for d in run_root.iterdir() 
                       if d.is_dir() and d.name.startswith('iter_')])
    
    if not iter_dirs:
        print(f"No iteration directories found in {run_root}")
        return
    
    results = []
    
    print(f"Analyzing {len(iter_dirs)} iterations...\n")
    
    for iter_dir in iter_dirs:
        iter_num = int(iter_dir.name.split('_')[1])
        state = load_iteration_state(run_root, iter_num)
        
        if state is None:
            continue
            
        epsilon_files = find_epsilon_files(iter_dir)
        
        result = {
            'iteration': iter_num,
            'max_abs_residual': state.get('max_abs_residual', float('inf')),
            'l2_residual': state.get('l2_residual', float('inf')),
            'max_param_step': state.get('max_param_step', float('inf')),
            'has_params_epsilon': 'params_epsilon' in epsilon_files,
            'has_lambda_vec': 'lambda_vec' in epsilon_files,
            'epsilon_tk_files': len(epsilon_files.get('epsilon_tk_files', [])),
            'epsilon_files': epsilon_files
        }
        results.append(result)
    
    if not results:
        print("No valid state.json files found!")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Sort by different criteria
    print("=== TOP ITERATIONS BY RESIDUALS ===\n")
    
    print(f"Top {top_n} by Max Absolute Residual:")
    top_max_abs = df.nsmallest(top_n, 'max_abs_residual')
    print(top_max_abs[['iteration', 'max_abs_residual', 'l2_residual', 'max_param_step']].to_string(index=False))
    print()
    
    print(f"Top {top_n} by L2 Residual:")
    top_l2 = df.nsmallest(top_n, 'l2_residual')
    print(top_l2[['iteration', 'max_abs_residual', 'l2_residual', 'max_param_step']].to_string(index=False))
    print()
    
    print(f"Top {top_n} by Parameter Step Size:")
    top_step = df.nsmallest(top_n, 'max_param_step')
    print(top_step[['iteration', 'max_abs_residual', 'l2_residual', 'max_param_step']].to_string(index=False))
    print()
    
    # Find the overall best iteration
    best_iter = df.loc[df['max_abs_residual'].idxmin()]
    print(f"=== BEST ITERATION OVERALL ===")
    print(f"Iteration {best_iter['iteration']} has the lowest max absolute residual: {best_iter['max_abs_residual']:.2e}")
    print()
    
    return df, best_iter

def extract_epsilon(run_root: Path, iteration: int, output_path: str = None):
    """Extract epsilon matrix from a specific iteration."""
    run_root = Path(run_root)
    iter_dir = run_root / format_iter(iteration)
    
    if not iter_dir.exists():
        print(f"Iteration directory {iter_dir} does not exist!")
        return None
    
    epsilon_files = find_epsilon_files(iter_dir)
    
    print(f"=== EPSILON FILES FOR ITERATION {iteration} ===")
    
    if 'params_epsilon' in epsilon_files:
        epsilon_path = epsilon_files['params_epsilon']
        print(f"Found params/epsilon.npy: {epsilon_path}")
        epsilon = np.load(epsilon_path)
        print(f"Shape: {epsilon.shape}")
        print(f"Epsilon matrix used for simulations in iteration {iteration}:")
        print(epsilon)
        print()
        
        if output_path:
            np.save(output_path, epsilon)
            print(f"Saved epsilon matrix to: {output_path}")
        
        return epsilon
    
    elif 'epsilon_tk_files' in epsilon_files:
        # Use the highest numbered epsilon_tk file
        latest_tk = max(epsilon_files['epsilon_tk_files'], 
                       key=lambda x: int(x.name.split('_')[2].split('.')[0]))
        print(f"Found epsilon_tk file: {latest_tk}")
        epsilon = np.load(latest_tk)
        print(f"Shape: {epsilon.shape}")
        print(f"Epsilon matrix from Newton update:")
        print(epsilon)
        print()
        
        if output_path:
            np.save(output_path, epsilon)
            print(f"Saved epsilon matrix to: {output_path}")
        
        return epsilon
    
    else:
        print(f"No epsilon matrix files found for iteration {iteration}")
        return None

def main():
    ap = argparse.ArgumentParser(description="Analyze MaxEnt residuals and extract epsilon matrices")
    ap.add_argument("--run-root", required=True, help="Path to MaxEnt run directory")
    ap.add_argument("--explain", action="store_true", help="Explain different epsilon file types")
    ap.add_argument("--extract-iter", type=int, help="Extract epsilon matrix from specific iteration")
    ap.add_argument("--output", help="Output path for extracted epsilon matrix (.npy file)")
    ap.add_argument("--top-n", type=int, default=10, help="Number of top iterations to show (default: 10)")
    args = ap.parse_args()
    
    run_root = Path(args.run_root).resolve()
    
    if not run_root.exists():
        print(f"Run directory does not exist: {run_root}")
        return 1
    
    if args.explain:
        explain_epsilon_files()
        print()
    
    if args.extract_iter is not None:
        epsilon = extract_epsilon(run_root, args.extract_iter, args.output)
        return 0
    
    # Analyze all iterations
    try:
        df, best_iter = analyze_residuals(run_root, args.top_n)
        
        print("=== RECOMMENDATIONS ===")
        print(f"For lowest residuals, use iteration {int(best_iter['iteration'])}")
        print(f"To extract this epsilon matrix, run:")
        print(f"  python analyze_residuals.py --run-root {run_root} --extract-iter {int(best_iter['iteration'])} --output best_epsilon.npy")
        print()
        
        # Show available epsilon files for best iteration
        best_iter_num = int(best_iter['iteration'])
        epsilon_files = find_epsilon_files(run_root / format_iter(best_iter_num))
        
        print(f"Available epsilon files for iteration {best_iter_num}:")
        for file_type, file_path in epsilon_files.items():
            if file_type == 'epsilon_tk_files':
                for f in file_path:
                    print(f"  - {f}")
            else:
                print(f"  - {file_path}")
                
    except ImportError:
        print("pandas not available - showing basic analysis")
        # Fallback without pandas
        iter_dirs = sorted([d for d in run_root.iterdir() 
                           if d.is_dir() and d.name.startswith('iter_')])
        
        best_residual = float('inf')
        best_iteration = None
        
        for iter_dir in iter_dirs:
            iter_num = int(iter_dir.name.split('_')[1])
            state = load_iteration_state(run_root, iter_num)
            
            if state and state.get('max_abs_residual', float('inf')) < best_residual:
                best_residual = state['max_abs_residual']
                best_iteration = iter_num
        
        if best_iteration is not None:
            print(f"Best iteration: {best_iteration} with residual {best_residual:.2e}")
            print(f"Extract with: python analyze_residuals.py --run-root {run_root} --extract-iter {best_iteration} --output best_epsilon.npy")
    
    return 0

if __name__ == "__main__":
    exit(main())




