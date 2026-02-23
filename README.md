# TFET Optimization Agent

A comprehensive AI-powered web application for multi-objective optimization of Tunnel Field-Effect Transistor (TFET) designs using advanced NSGA-III algorithm with machine learning enhancements.

## Overview

This project implements a sophisticated optimization framework for TFET (Tunnel Field-Effect Transistor) semiconductor device design. It combines evolutionary algorithms (NSGA-III) with machine learning techniques to efficiently explore the design space and find optimal device configurations.

### Key Capabilities

- Multi-objective optimization balancing competing design goals
- Interactive web-based interface for easy access
- Advanced ML models for accelerated optimization
- Comprehensive visualization and analysis tools

## Features

### 🚀 Advanced Optimization

- **NSGA-III Algorithm**: Many-objective evolutionary optimization with reference directions
- **Multi-Objective**: Simultaneous optimization of Natural Length, Vertical E-field, and Ion/Ioff Ratio
- **Knee Point Detection**: Automatic identification of best compromise solutions
- **Convergence Tracking**: Real-time hypervolume indicator monitoring
- **Population Size**: Configurable 50-200 individuals
- **Generations**: Configurable 50-300 iterations

### 🤖 Machine Learning Integration

- **Surrogate Models**: Gaussian Process with Matérn kernel for fast evaluation
- **Advanced ML Models**: XGBoost, LightGBM, CatBoost integration
- **Active Learning**: Uncertainty-based sample selection for model improvement
- **Inverse Design**: Neural network-based design from target specifications
- **Adaptive Retraining**: Continuous model improvement during optimization
- **Ensemble Learning**: Combining multiple models for better accuracy
- **Data Augmentation**: Synthetic data generation for improved training

### 📊 Advanced Analytics

- **3D Pareto Front Visualization**: Interactive Plotly.js plots with rotation/zoom
- **Convergence Analysis**: Generation-by-generation hypervolume tracking
- **Parameter Distribution**: Statistical analysis of optimal solutions
- **Skewness Analysis**: Dataset quality assessment with Q-Q plots
- **Diversity Metrics**: Average distance between individuals tracking
- **Selection Analysis**: Parent selection visualization

### 🔬 Material Coverage

- **Bulk Materials**: Si, Ge, InAs, GaSb support
- **2D Materials**: MoS2, WSe2 integration
- **Heterostructures**: Multi-material device modeling
- **Defect Modeling**: Realistic device performance simulation

### 📁 Data Flexibility

- **CSV Upload**: Custom dataset integration with validation
- **Synthetic Data**: Advanced ML-generated training sets
- **Pre-loaded Datasets**: Aluminum, Copper, Sample TFET data included
- **Validation**: Automatic data quality checking
- **Skewness Analysis**: Dataset statistics and distribution analysis

### 📄 Report Generation

- **PDF Reports**: Complete optimization reports with all graphs
- **Publication Quality**: High-DPI figures for research papers
- **Interactive Visualizations**: Embeddable Plotly charts

### 🎨 Web Interface

- **Responsive Design**: Works on desktop and tablet
- **Real-time Progress**: Optimization status tracking
- **Interactive Plots**: Zoom, pan, and rotate capabilities
- **Export Options**: Download results as PDF

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (tested environment)

### Quick Setup

1. **Clone or download** this repository to your local machine
2. **Run the installer**:
   ```batch
   install_dependencies.bat
   ```
3. **Start the application**:
   ```batch
   run_web_app.bat
   ```
4. **Open your browser** to: http://localhost:5000

## Quick Start

### Option 1: Web Interface

1. Run `run_web_app.bat` or execute `python tfet_optimization_agent/web_interface/app.py`
2. Open browser to http://localhost:5000
3. Choose data source (synthetic or CSV)
4. Click "Run Optimization"
5. View results with interactive 3D visualization

### Option 2: Command Line

Run `python run_advanced_tfet.py` and choose:

- 1: Basic NSGA-III
- 2: Advanced Framework with ML
- 3: Web Interface Test

### Manual Installation

If the automatic installer doesn't work:

```bash
pip install flask==2.3.3
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install scipy==1.11.2
pip install scikit-learn==1.3.0
pip install pymoo==0.6.0.1
pip install werkzeug==2.3.7
```

## Usage

### 1. Home Page

- Overview of system capabilities
- Navigation to optimization and results
- System status monitoring

### 2. Optimization Page

- **Data Source Selection**: Choose between synthetic data or CSV upload
- **CSV Upload**: Upload custom TFET datasets with required columns:
  - `gate_voltage` (V)
  - `drain_voltage` (V)
  - `channel_length` (m)
  - `oxide_thickness` (m)
- **Skewness Analysis**: Statistical analysis of available datasets
- **Parameter Configuration**: Population size and generation settings
- **Run Optimization**: Execute NSGA-III with progress tracking

### 3. Results Page

- **3D Pareto Front**: Interactive visualization with knee point highlighting
- **Convergence Analysis**: Hypervolume indicator progression
- **Objective Trade-offs**: 2D projections of solution space
- **Parameter Distribution**: Statistical analysis of optimal parameters
- **Solutions Table**: Top 10 solutions with detailed parameters
- **Algorithm Information**: Comprehensive optimization details

## File Structure

```
TFET _ AGENT/
├── tfet_optimization_agent/
│   ├── src/
│   │   └── tfet_agent/
│   │       ├── data/                    # Data handling modules
│   │       ├── ml/                      # Machine learning components
│   │       ├── optimization/            # NSGA-III implementation
│   │       ├── visualization/           # Plotting utilities
│   │       ├── advanced_framework.py    # ML-enhanced framework
│   │       └── main.py                  # Core optimization logic
│   └── web_interface/
│       ├── static/                      # CSS, JS, images
│       ├── templates/                   # HTML templates
│       ├── uploads/                     # CSV file storage
│       └── app.py                       # Flask web application
├── requirements.txt                     # Python dependencies
├── install_dependencies.bat             # Automatic installer
├── run_web_app.bat                     # Application launcher
├── run_advanced_tfet.py                # CLI runner
├── augment_data.py                      # Data augmentation script
└── README.md                           # This file
```

## Data Files

The project includes sample datasets:

| File                             | Description                           |
| -------------------------------- | ------------------------------------- |
| aluminum_tfet_data.csv           | Aluminum-based TFET data              |
| aluminum_tfet_data_augmented.csv | Augmented aluminum data (100 samples) |
| copper_tfet_data.csv             | Copper-based TFET data                |
| copper_tfet_data_augmented.csv   | Augmented copper data (100 samples)   |
| sample_tfet_data.csv             | Sample/template data                  |

### Data Augmentation

- `augment_data.py`: Augment a single dataset with ±5% variation
- `augment_all_data.py`: Batch augment all datasets

## API Endpoints

### Core Functionality

- `POST /api/optimize` - Run NSGA-III optimization
- `GET /api/results` - Retrieve optimization results
- `POST /api/upload-csv` - Upload custom datasets
- `GET /api/csv-status` - Check uploaded file status

### Analysis Features

- `GET /api/skewness-analysis` - Statistical dataset analysis
- `GET /api/available-datasets` - List available datasets
- `GET /api/test-csv` - Test CSV processing functionality

## Technical Details

### Optimization Algorithm

- **NSGA-III**: Reference point-based many-objective optimization
- **Population Size**: 50-200 (configurable)
- **Generations**: 50-300 (configurable)
- **Crossover**: Simulated Binary Crossover (SBX) with η=15
- **Mutation**: Polynomial Mutation (PM) with η=20

### Objectives

1. **Natural Length** (minimize): √(εsi × tox / (q × doping))
2. **Vertical E-field** (minimize): Vg / tox
3. **Ion/Ioff Ratio** (maximize): Ion current / Ioff current

### Constraints

- Ion/Ioff ratio > 10^6
- Natural length < 100 nm

### Machine Learning Models

- **Surrogate Models**: Gaussian Process with Matérn kernel
- **Active Learning**: Uncertainty-based sample selection
- **Inverse Design**: Multi-layer perceptron (100-50-25 neurons)

## Troubleshooting

### Common Issues

1. **"Python not found"**
   - Install Python 3.8+ from python.org
   - Add Python to system PATH

2. **"Module not found" errors**
   - Run `install_dependencies.bat`
   - Check internet connection for pip downloads

3. **"Port already in use"**
   - Close other applications using port 5000
   - Or modify port in `app.py`

4. **CSV upload fails**
   - Ensure CSV has required columns: gate_voltage, drain_voltage, channel_length, oxide_thickness
   - Check file size (max 16MB)

5. **Optimization fails**
   - Check system memory (optimization requires ~1GB RAM)
   - Reduce population size if needed

### Performance Tips

- Use synthetic data for fastest results
- Reduce population size for quicker optimization
- Close other applications to free memory
- Use CSV data for domain-specific optimization

## Development

### Adding New Features

1. **New Objectives**: Modify `tfet_problem.py`
2. **New Algorithms**: Extend `nsga3.py`
3. **New Visualizations**: Add to `pareto_plot.py`
4. **New ML Models**: Extend `surrogate_model.py`

### Testing

- Use `/api/test-csv` endpoint for CSV processing tests
- Check browser console for JavaScript errors
- Monitor Flask console for backend errors

## License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure Python 3.8+ is properly installed
4. Check that all required files are present

## Version History

- **v1.0**: Initial release with NSGA-III optimization
- **v1.1**: Added ML enhancements and surrogate models
- **v1.2**: Web interface with interactive visualizations
- **v1.3**: CSV upload and skewness analysis features
- **v1.4**: Complete error handling and user experience improvements

---

**TFET Optimization Agent** - Advanced AI-powered semiconductor device optimization platform.
