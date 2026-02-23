import sys
import os
sys.path.append('tfet_optimization_agent/src')

def run_basic_optimization():
    """basic optimization run"""
    from tfet_agent.main import run_tfet_optimization
    
    print("running optimization...")
    result, analysis = run_tfet_optimization()
    
    if analysis:
        print("got", analysis['pareto_front_size'], "solutions")
        print("using:", analysis.get('algorithm', 'NSGA-III'))
    
    return result, analysis

def run_advanced_framework():
    """try to run advanced version"""
    try:
        from tfet_agent.advanced_framework import run_advanced_tfet_optimization
        
        print("running advanced version...")
        result, analysis = run_advanced_tfet_optimization()
        
        if analysis:
            print("framework:", analysis.get('framework', 'Advanced'))
            print("ml stuff:", analysis.get('ml_features', {}))
            print("materials:", analysis.get('material_coverage', {}).get('materials_used', []))
        
        return result, analysis
        
    except ImportError as e:
        print("missing dependencies:", e)
        print("need to install: pip install torch scikit-learn")
        return run_basic_optimization()

def run_web_optimization():
    """Test web interface optimization"""
    sys.path.append('tfet_optimization_agent/web_interface')
    from app import app
    
    print("=== Testing Web Interface ===")
    with app.test_client() as client:
        # Test synthetic data
        response = client.post('/api/optimize?source=synthetic')
        if response.status_code == 200:
            data = response.get_json()
            print(f"✅ Web optimization: {data.get('status')}")
            print(f"📊 Solutions: {data.get('results', {}).get('pareto_front_size', 0)}")
        else:
            print("❌ Web optimization failed")

if __name__ == '__main__':
    print("TFET optimization tool")
    print("my semester project")
    print("-" * 30)
    
    # Choose what to run
    choice = input("Choose: (1) Basic NSGA-III (2) Advanced Framework (3) Web Test: ")
    
    if choice == '1':
        run_basic_optimization()
    elif choice == '2':
        run_advanced_framework()
    elif choice == '3':
        run_web_optimization()
    else:
        print("Running all...")
        run_basic_optimization()
        run_advanced_framework()
        run_web_optimization()