// Optimization page JavaScript functionality

class OptimizationController {
    constructor() {
        this.isOptimizing = false;
        this.currentResults = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkInitialState();
    }

    setupEventListeners() {
        // Data source selection
        document.querySelectorAll('input[name="dataSource"]').forEach(radio => {
            radio.addEventListener('change', (e) => this.handleDataSourceChange(e));
        });

        // File upload
        const csvFile = document.getElementById('csvFile');
        if (csvFile) {
            csvFile.addEventListener('change', (e) => this.handleFileUpload(e));
        }

        // Optimization controls
        const runButton = document.getElementById('runOptimization');
        if (runButton) {
            runButton.addEventListener('click', () => this.runOptimization());
        }

        const clearButton = document.getElementById('clearResults');
        if (clearButton) {
            clearButton.addEventListener('click', () => this.clearResults());
        }

        // Skewness analysis
        const analyzeButton = document.getElementById('analyzeSkewness');
        if (analyzeButton) {
            analyzeButton.addEventListener('click', () => this.analyzeSkewness());
        }

        // Range sliders
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            slider.addEventListener('input', (e) => this.updateRangeValue(e.target));
        });
    }

    checkInitialState() {
        this.checkCSVStatus();
        this.loadAvailableDatasets();
    }

    handleDataSourceChange(event) {
        const csvSection = document.getElementById('csvSection');
        if (event.target.value === 'csv') {
            csvSection.style.display = 'block';
        } else {
            csvSection.style.display = 'none';
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file && file.name.endsWith('.csv')) {
            this.uploadCSV(file);
        } else {
            this.showError('Please select a valid CSV file');
        }
    }

    async uploadCSV(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload-csv', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            this.displayUploadResult(data);
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Failed to upload file');
        }
    }

    displayUploadResult(data) {
        const fileInfo = document.getElementById('fileInfo');
        if (data.status === 'success') {
            fileInfo.innerHTML = `
                <strong>✓ File uploaded successfully</strong><br>
                Filename: ${data.filename}<br>
                Rows: ${data.rows}
            `;
            fileInfo.className = 'file-info success';
            fileInfo.style.display = 'block';
        } else {
            fileInfo.innerHTML = `<strong>✗ Error:</strong> ${data.error}`;
            fileInfo.className = 'file-info error';
            fileInfo.style.display = 'block';
        }
    }

    async checkCSVStatus() {
        try {
            const response = await fetch('/api/csv-status');
            const data = await response.json();
            
            if (data.has_csv) {
                const fileInfo = document.getElementById('fileInfo');
                fileInfo.innerHTML = `
                    <strong>✓ CSV file available</strong><br>
                    Filename: ${data.filename}<br>
                    Rows: ${data.rows}
                `;
                fileInfo.className = 'file-info success';
                fileInfo.style.display = 'block';
            }
        } catch (error) {
            console.error('Error checking CSV status:', error);
        }
    }

    async loadAvailableDatasets() {
        try {
            const response = await fetch('/api/available-datasets');
            const data = await response.json();
            
            const select = document.getElementById('datasetSelect');
            if (select && data.datasets) {
                select.innerHTML = '';
                data.datasets.forEach(dataset => {
                    const option = document.createElement('option');
                    option.value = dataset;
                    option.textContent = dataset.charAt(0).toUpperCase() + dataset.slice(1) + ' TFET';
                    select.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Error loading datasets:', error);
        }
    }

    async analyzeSkewness() {
        const dataset = document.getElementById('datasetSelect').value;
        if (!dataset) return;

        try {
            const response = await fetch(`/api/skewness-analysis?dataset=${dataset}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displaySkewnessResults(data);
            } else {
                this.showError(data.error || 'Failed to analyze skewness');
            }
        } catch (error) {
            console.error('Skewness analysis error:', error);
            this.showError('Failed to analyze skewness');
        }
    }

    displaySkewnessResults(data) {
        // Display plot
        const plotDiv = document.getElementById('skewnessPlot');
        plotDiv.innerHTML = `<img src="data:image/png;base64,${data.plot}" style="width: 100%; height: auto; border-radius: 5px;">`;
        
        // Display statistics
        const statsPanel = document.getElementById('skewnessStats');
        const statsGrid = document.getElementById('statsGrid');
        
        let statsHTML = '';
        for (const [param, stats] of Object.entries(data.statistics)) {
            statsHTML += `
                <div class="stat-item">
                    <h5>${param}</h5>
                    <p>Skewness: ${stats.skewness.toFixed(3)}</p>
                    <p>Kurtosis: ${stats.kurtosis.toFixed(3)}</p>
                </div>
            `;
        }
        
        statsGrid.innerHTML = statsHTML;
        statsPanel.style.display = 'block';
    }

    async runOptimization() {
        if (this.isOptimizing) return;

        const dataSource = document.querySelector('input[name="dataSource"]:checked').value;
        
        this.isOptimizing = true;
        this.showOptimizationProgress();
        
        try {
            const response = await fetch(`/api/optimize?source=${dataSource}`, {
                method: 'POST'
            });

            const data = await response.json();
            this.handleOptimizationResult(data);
        } catch (error) {
            console.error('Optimization error:', error);
            this.showError('Optimization failed: ' + error.message);
        } finally {
            this.isOptimizing = false;
            this.hideOptimizationProgress();
        }
    }

    showOptimizationProgress() {
        const statusPanel = document.getElementById('statusPanel');
        const statusMessage = document.getElementById('statusMessage');
        const progressFill = document.getElementById('progressFill');
        const runButton = document.getElementById('runOptimization');

        statusPanel.style.display = 'block';
        statusMessage.textContent = 'Running NSGA-III optimization...';
        runButton.disabled = true;
        runButton.textContent = 'Optimizing...';
        
        // Animate progress
        let progress = 0;
        this.progressInterval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
        }, 200);
    }

    hideOptimizationProgress() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        const progressFill = document.getElementById('progressFill');
        const runButton = document.getElementById('runOptimization');
        
        progressFill.style.width = '100%';
        runButton.disabled = false;
        runButton.textContent = 'Run NSGA-III Optimization';
    }

    handleOptimizationResult(data) {
        const statusMessage = document.getElementById('statusMessage');
        
        if (data.status === 'success') {
            statusMessage.textContent = data.message;
            this.currentResults = data.results;
            this.displayOptimizationSummary(data.results);
        } else {
            statusMessage.textContent = 'Error: ' + data.message;
            this.showError(data.message);
        }
    }

    displayOptimizationSummary(results) {
        const resultsDiv = document.getElementById('optimizationResults');
        const summaryDiv = document.getElementById('resultsSummary');
        
        if (results) {
            summaryDiv.innerHTML = `
                <div class="results-summary">
                    <div class="summary-card">
                        <h4>Algorithm</h4>
                        <p class="metric">${results.algorithm || 'NSGA-III'}</p>
                    </div>
                    <div class="summary-card">
                        <h4>Solutions Found</h4>
                        <p class="metric">${results.pareto_front_size || 0}</p>
                    </div>
                    <div class="summary-card">
                        <h4>Framework</h4>
                        <p class="metric">${results.framework || 'Advanced'}</p>
                    </div>
                    <div class="summary-card">
                        <h4>ML Features</h4>
                        <p class="metric">${results.ml_features ? 'Enabled' : 'Disabled'}</p>
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <a href="/results" class="btn btn-primary">View Detailed Results</a>
                    <button onclick="optimizationController.downloadResults()" class="btn btn-secondary">Download Results</button>
                </div>
            `;
            resultsDiv.style.display = 'block';
        }
    }

    clearResults() {
        const statusPanel = document.getElementById('statusPanel');
        const progressFill = document.getElementById('progressFill');
        
        statusPanel.style.display = 'none';
        progressFill.style.width = '0%';
        this.currentResults = null;
    }

    downloadResults() {
        if (!this.currentResults) {
            this.showError('No results to download');
            return;
        }

        const dataStr = JSON.stringify(this.currentResults, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'tfet_optimization_results.json';
        link.click();
    }

    updateRangeValue(slider) {
        const valueSpan = document.getElementById(slider.id + 'Value');
        if (valueSpan) {
            valueSpan.textContent = slider.value;
        }
    }

    showError(message) {
        // Create or update error message
        let errorDiv = document.getElementById('errorMessage');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'errorMessage';
            errorDiv.className = 'error-message';
            document.querySelector('.optimization-container').prepend(errorDiv);
        }
        
        errorDiv.innerHTML = `
            <strong>Error:</strong> ${message}
            <button onclick="this.parentElement.style.display='none'" style="float: right; background: none; border: none; color: white; cursor: pointer;">×</button>
        `;
        errorDiv.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (errorDiv) errorDiv.style.display = 'none';
        }, 5000);
    }
}

// Initialize when DOM is loaded
let optimizationController;
document.addEventListener('DOMContentLoaded', function() {
    optimizationController = new OptimizationController();
});

// Global functions for backward compatibility
function updateRangeValue(id) {
    if (optimizationController) {
        const slider = document.getElementById(id);
        optimizationController.updateRangeValue(slider);
    }
}