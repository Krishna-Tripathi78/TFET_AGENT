from .optimization.tfet_problem import TFETProblem
from .optimization.nsga3 import NSGA3Optimizer
from .data.tfet_data import TFETDataHandler

def run_tfet_optimization():
    try:
        problem = TFETProblem()
        optimizer = NSGA3Optimizer(pop_size=100, n_gen=150)
        data_handler = TFETDataHandler()
        synthetic_data = data_handler.generate_synthetic_dataset(1000)
        print(f"Generated {len(synthetic_data)} synthetic TFET samples")
      
        print("Starting TFET optimization using NSGA-III algorithm...")
        result = optimizer.optimize(problem)
        analysis = optimizer.analyze_results(result)
        
        if analysis:
            print(f"Optimization completed. Found {analysis['pareto_front_size']} Pareto optimal solutions")
            print("Best solution parameters:")
            best = analysis['best_parameters'][0]
            print(f"  Gate Voltage: {best[0]:.3f} V")
            print(f"  Drain Voltage: {best[1]:.3f} V") 
            print(f"  Channel Length: {best[2]*1e9:.1f} nm")
            print(f"  Oxide Thickness: {best[3]*1e9:.1f} nm")
            print(f"  Doping: {best[4]:.2e} cm^-3")
            print("Generating Pareto front visualizations...")
            try:
                fig_3d, fig_analysis = optimizer.visualize_pareto_front(result, "tfet_results")
                print("Visualizations saved: tfet_results_3d_pareto.png, tfet_results_analysis.png")
            except Exception as viz_error:
                print(f"Visualization error (non-critical): {viz_error}")
        
        return result, analysis
        
    except Exception as e:
        print(f"Optimization error: {e}")
        # Return None to trigger fallback data generation
        return None, None

if __name__ == "__main__":
    run_tfet_optimization()