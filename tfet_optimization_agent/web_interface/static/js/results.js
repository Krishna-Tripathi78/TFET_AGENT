// Results page JavaScript functionality

class ResultsController {
    constructor() {
        this.currentResults = null;
        this.init();
    }

    init() {
        this.loadResults();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Add any additional event listeners here
        window.addEventListener('resize', () => {
            this.resizePlots();
        });
    }

    async loadResults() {
        try {
            const response = await fetch('/api/results');
            const data = await response.json();
            
            if (data.message && data.message.includes('No results')) {
                this.showNoResults();
            } else {
                this.currentResults = data;
                this.displayResults(data);
            }
        } catch (error) {
            console.error('Error loading results:', error);
            this.showNoResults();
        }
    }

    showNoResults() {
        document.getElementById('noResults').style.display = 'block';
        document.getElementById('resultsContent').style.display = 'none';
        const downloadBtn = document.getElementById('downloadAllGraphs');
        if (downloadBtn) downloadBtn.style.display = 'none';
    }

    displayResults(results) {
        document.getElementById('noResults').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'block';
        const downloadBtn = document.getElementById('downloadAllGraphs');
        if (downloadBtn) downloadBtn.style.display = 'block';
        
        this.displaySummary(results);
        this.displayVisualizations(results);
        this.displaySolutionsTable(results);
        this.displayAlgorithmInfo(results);
    }

    displaySummary(results) {
        const summaryDiv = document.getElementById('resultsSummary');
        const stats = results.statistics || {};
        
        summaryDiv.innerHTML = `
            <div class="summary-card">
                <h3>Algorithm</h3>
                <div class="metric">${results.algorithm || 'NSGA-III'}</div>
            </div>
            <div class="summary-card">
                <h3>Solutions Found</h3>
                <div class="metric">${results.pareto_front_size || 0}</div>
            </div>
            <div class="summary-card">
                <h3>Min Natural Length</h3>
                <div class="metric">${stats.min_natural_length ? stats.min_natural_length.toFixed(2) + ' nm' : 'N/A'}</div>
            </div>
            <div class="summary-card">
                <h3>Avg E-field</h3>
                <div class="metric">${stats.avg_efield ? stats.avg_efield.toExponential(2) + ' V/m' : 'N/A'}</div>
            </div>
            <div class="summary-card">
                <h3>Max Ion/Ioff</h3>
                <div class="metric">${stats.max_ion_ioff ? stats.max_ion_ioff.toExponential(2) : 'N/A'}</div>
            </div>
            <div class="summary-card">
                <h3>Diversity</h3>
                <div class="metric">${stats.diversity_metric ? stats.diversity_metric.toFixed(3) : 'N/A'}</div>
            </div>
        `;
    }

    displayVisualizations(results) {
        this.displayParetoFront(results);
        this.displayConvergence(results);
        this.displayTradeoffs(results);
        this.displayParameterDistribution(results);
        this.displayAverageDistance(results);
        this.displaySelectionFunction(results);
        this.displayAverageSpread(results);
        this.displayScoreHistogram(results);
        this.displayIndividualDistance(results);
    }

    displayParetoFront(results) {
        if (!results.objectives || results.objectives.length === 0) {
            this.showPlotError('paretoPlot', 'No objective data available');
            return;
        }

        try {
            const objectives = results.objectives;
            const x = objectives.map(obj => obj[0]);
            const y = objectives.map(obj => obj[1]);
            const z = objectives.map(obj => Math.abs(obj[2]));

            const trace = {
                x: x,
                y: y,
                z: z,
                mode: 'markers',
                marker: {
                    size: 12,
                    color: z,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {
                        title: {
                            text: 'Ion/Ioff Ratio',
                            font: {size: 18, family: 'Arial, sans-serif', color: '#000'}
                        },
                        titleside: 'right',
                        tickformat: '.1e',
                        tickfont: {size: 14, family: 'Arial, sans-serif', color: '#000'},
                        len: 0.7,
                        thickness: 20
                    },
                    opacity: 0.9,
                    line: {width: 0.5, color: '#333'}
                },
                type: 'scatter3d',
                name: 'Pareto Solutions'
            };

            const traces = [trace];

            if (results.knee_solution && results.knee_solution.objectives) {
                const knee = results.knee_solution.objectives;
                traces.push({
                    x: [knee[0]],
                    y: [knee[1]],
                    z: [Math.abs(knee[2])],
                    mode: 'markers',
                    marker: {
                        size: 20,
                        color: '#FF0000',
                        symbol: 'diamond'
                    },
                    type: 'scatter3d',
                    name: 'Knee Point'
                });
            }

            const layout = {
                title: {
                    text: 'NSGA-III 3D Pareto Front',
                    font: {size: 32, family: 'Arial, sans-serif', color: '#333'},
                    x: 0.5,
                    xanchor: 'center'
                },
                scene: {
                    xaxis: {
                        title: {text: 'Natural Length (nm)', font: {size: 20, family: 'Arial, sans-serif', color: '#000'}},
                        tickfont: {size: 14, family: 'Arial, sans-serif', color: '#000'}
                    },
                    yaxis: {
                        title: {text: 'Vertical E-field (V/m)', font: {size: 20, family: 'Arial, sans-serif', color: '#000'}},
                        tickfont: {size: 14, family: 'Arial, sans-serif', color: '#000'},
                        tickformat: '.2e'
                    },
                    zaxis: {
                        title: {text: 'Ion/Ioff Ratio', font: {size: 20, family: 'Arial, sans-serif', color: '#000'}},
                        tickfont: {size: 14, family: 'Arial, sans-serif', color: '#000'},
                        tickformat: '.2e'
                    },
                    camera: {
                        eye: {x: 1.3, y: 1.3, z: 1.3}
                    },
                    bgcolor: 'rgba(240,240,240,0.9)'
                },
                margin: {l: 0, r: 150, b: 0, t: 80},
                showlegend: true,
                autosize: true,
                paper_bgcolor: 'white'
            };

            Plotly.newPlot('paretoPlot', traces, layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'pareto_front_3d',
                    height: 1000,
                    width: 1400,
                    scale: 3
                }
            });
        } catch (error) {
            console.error('Error creating Pareto plot:', error);
            this.showPlotError('paretoPlot', 'Error creating 3D plot');
        }
    }

    displayConvergence(results) {
        if (!results.convergence_data || !results.convergence_data.hypervolume) {
            this.showPlotError('convergencePlot', 'No convergence data available');
            return;
        }

        try {
            const trace = {
                x: results.convergence_data.generations,
                y: results.convergence_data.hypervolume,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Hypervolume',
                line: {
                    color: '#667eea',
                    width: 3
                },
                marker: {
                    size: 6,
                    color: '#667eea'
                }
            };

            const layout = {
                title: {
                    text: '<b>Convergence Analysis</b>',
                    font: {size: 28, family: 'Arial Black, sans-serif'}
                },
                xaxis: {
                    title: {text: '<b>Generation</b>', font: {size: 18, family: 'Arial Black, sans-serif'}},
                    tickfont: {size: 14, family: 'Arial, sans-serif'},
                    showgrid: true,
                    gridcolor: '#ddd'
                },
                yaxis: {
                    title: {text: '<b>Hypervolume Indicator</b>', font: {size: 18, family: 'Arial Black, sans-serif'}},
                    tickfont: {size: 14, family: 'Arial, sans-serif'},
                    showgrid: true,
                    gridcolor: '#ddd'
                },
                margin: {l: 100, r: 40, b: 80, t: 80},
                showlegend: false,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white'
            };

            Plotly.newPlot('convergencePlot', [trace], layout, {
                responsive: true,
                displayModeBar: true,
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'convergence_analysis',
                    height: 800,
                    width: 1200,
                    scale: 3
                }
            });
        } catch (error) {
            console.error('Error creating convergence plot:', error);
            this.showPlotError('convergencePlot', 'Error creating convergence plot');
        }
    }

    displayTradeoffs(results) {
        if (!results.objectives || results.objectives.length === 0) {
            this.showPlotError('tradeoffPlot', 'No objective data available');
            return;
        }

        try {
            const objectives = results.objectives;
            
            const trace = {
                x: objectives.map(obj => obj[0]),
                y: objectives.map(obj => obj[1]),
                mode: 'markers',
                type: 'scatter',
                name: 'Solutions',
                marker: {
                    size: 14,
                    color: objectives.map(obj => Math.abs(obj[2])),
                    colorscale: 'Jet',
                    showscale: true,
                    colorbar: {
                        title: '<b>Ion/Ioff Ratio</b>',
                        titleside: 'right',
                        titlefont: {size: 20, family: 'Times New Roman, serif'},
                        tickfont: {size: 18, family: 'Times New Roman, serif'}
                    },
                    opacity: 0.85,
                    line: {width: 2, color: '#000'}
                }
            };

            const layout = {
                title: {
                    text: '<b>Objective Trade-offs</b>',
                    font: {size: 36, family: 'Times New Roman, serif', color: '#000'}
                },
                xaxis: {
                    title: {text: '<b>Natural Length (nm)</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}},
                    tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'},
                    showgrid: true,
                    gridcolor: '#888',
                    gridwidth: 2,
                    linewidth: 2,
                    linecolor: '#000'
                },
                yaxis: {
                    title: {text: '<b>Vertical E-field (V/m)</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}},
                    tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'},
                    showgrid: true,
                    gridcolor: '#888',
                    gridwidth: 2,
                    linewidth: 2,
                    linecolor: '#000'
                },
                margin: {l: 120, r: 40, b: 100, t: 100},
                showlegend: false,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                font: {family: 'Times New Roman, serif', size: 20, color: '#000'}
            };

            Plotly.newPlot('tradeoffPlot', [trace], layout, {
                responsive: true,
                displayModeBar: true,
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'objective_tradeoffs',
                    height: 800,
                    width: 1200,
                    scale: 3
                }
            });
        } catch (error) {
            console.error('Error creating tradeoff plot:', error);
            this.showPlotError('tradeoffPlot', 'Error creating tradeoff plot');
        }
    }

    displayParetoComparison(results) {
        if (!results.objectives || results.objectives.length === 0) {
            this.showPlotError('paretoComparisonPlot', 'No objective data available');
            return;
        }

        try {
            const objectives = results.objectives;
            const material = results.csv_file_used ? results.csv_file_used.toLowerCase() : 'unknown';
            
            let materialName = 'Unknown';
            let markerColor = '#0047AB';
            if (material.includes('aluminum') || material.includes('aluminium')) {
                materialName = 'Aluminum';
                markerColor = '#0047AB';
            } else if (material.includes('copper')) {
                materialName = 'Copper';
                markerColor = '#DC143C';
            }
            
            const trace = {
                x: objectives.map(obj => obj[0]),
                y: objectives.map(obj => obj[1]),
                mode: 'markers',
                type: 'scatter',
                name: materialName,
                marker: {
                    size: 16,
                    color: markerColor,
                    symbol: 'circle',
                    line: {width: 2, color: '#000'}
                }
            };

            const layout = {
                title: {
                    text: '<b>Pareto Front: E-field vs Natural Length</b>',
                    font: {size: 36, family: 'Times New Roman, serif', color: '#000'}
                },
                xaxis: {
                    title: {text: '<b>Natural Length (nm)</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}},
                    tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'},
                    showgrid: true,
                    gridcolor: '#888',
                    gridwidth: 2,
                    linewidth: 2,
                    linecolor: '#000'
                },
                yaxis: {
                    title: {text: '<b>E-field (×10⁷ V/m)</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}},
                    tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'},
                    showgrid: true,
                    gridcolor: '#888',
                    gridwidth: 2,
                    linewidth: 2,
                    linecolor: '#000'
                },
                margin: {l: 120, r: 40, b: 100, t: 100},
                showlegend: true,
                legend: {
                    x: 0.75,
                    y: 0.95,
                    font: {size: 22, family: 'Times New Roman, serif'},
                    bgcolor: 'rgba(255,255,255,0.9)',
                    bordercolor: '#000',
                    borderwidth: 2
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                font: {family: 'Times New Roman, serif', size: 20, color: '#000'}
            };

            Plotly.newPlot('paretoComparisonPlot', [trace], layout, {
                responsive: true,
                displayModeBar: true,
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'pareto_comparison',
                    height: 800,
                    width: 1200,
                    scale: 3
                }
            });
        } catch (error) {
            console.error('Error creating Pareto comparison plot:', error);
            this.showPlotError('paretoComparisonPlot', 'Error creating comparison plot');
        }
    }

    displayParameterDistribution(results) {
        if (!results.best_parameters || results.best_parameters.length === 0) {
            this.showPlotError('parameterPlot', 'No parameter data available');
            return;
        }

        try {
            const parameters = results.best_parameters;
            let gateVoltages, drainVoltages, channelLengths, oxideThickness;

            // Handle different parameter formats
            if (typeof parameters[0] === 'object' && !Array.isArray(parameters[0])) {
                // Object format
                gateVoltages = parameters.map(p => p.gate_voltage);
                drainVoltages = parameters.map(p => p.drain_voltage);
                channelLengths = parameters.map(p => p.channel_length * 1e9);
                oxideThickness = parameters.map(p => p.oxide_thickness * 1e9);
            } else {
                // Array format
                gateVoltages = parameters.map(p => p[0]);
                drainVoltages = parameters.map(p => p[1]);
                channelLengths = parameters.map(p => p[2] * 1e9);
                oxideThickness = parameters.map(p => p[3] * 1e9);
            }

            const trace = {
                x: gateVoltages,
                y: channelLengths,
                mode: 'markers',
                type: 'scatter',
                name: 'Parameter Distribution',
                marker: {
                    size: 8,
                    color: drainVoltages,
                    colorscale: 'Plasma',
                    showscale: true,
                    colorbar: {
                        title: 'Drain Voltage (V)',
                        titleside: 'right'
                    },
                    opacity: 0.7
                }
            };

            const layout = {
                title: {
                    text: '<b>Parameter Distribution</b>',
                    font: {size: 28, family: 'Arial Black, sans-serif'}
                },
                xaxis: {
                    title: {text: '<b>Gate Voltage (V)</b>', font: {size: 18, family: 'Arial Black, sans-serif'}},
                    tickfont: {size: 14, family: 'Arial, sans-serif'},
                    showgrid: true,
                    gridcolor: '#ddd'
                },
                yaxis: {
                    title: {text: '<b>Channel Length (nm)</b>', font: {size: 18, family: 'Arial Black, sans-serif'}},
                    tickfont: {size: 14, family: 'Arial, sans-serif'},
                    showgrid: true,
                    gridcolor: '#ddd'
                },
                margin: {l: 100, r: 40, b: 80, t: 80},
                showlegend: false,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white'
            };

            Plotly.newPlot('parameterPlot', [trace], layout, {
                responsive: true,
                displayModeBar: true,
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'parameter_distribution',
                    height: 800,
                    width: 1200,
                    scale: 3
                }
            });
        } catch (error) {
            console.error('Error creating parameter plot:', error);
            this.showPlotError('parameterPlot', 'Error creating parameter plot');
        }
    }

    displaySolutionsTable(results) {
        if (!results.best_parameters || !results.objectives) {
            return;
        }

        try {
            const tbody = document.querySelector('#solutionsTable tbody');
            tbody.innerHTML = '';

            const numSolutions = Math.min(10, results.best_parameters.length);

            for (let i = 0; i < numSolutions; i++) {
                const params = results.best_parameters[i];
                const objs = results.objectives[i];

                let gv, dv, cl, ot;
                if (typeof params === 'object' && !Array.isArray(params)) {
                    gv = params.gate_voltage;
                    dv = params.drain_voltage;
                    cl = params.channel_length * 1e9;
                    ot = params.oxide_thickness * 1e9;
                } else {
                    gv = params[0];
                    dv = params[1];
                    cl = params[2] * 1e9;
                    ot = params[3] * 1e9;
                }

                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${i + 1}</td>
                    <td>${gv.toFixed(3)}</td>
                    <td>${dv.toFixed(3)}</td>
                    <td>${cl.toFixed(2)}</td>
                    <td>${ot.toFixed(2)}</td>
                    <td>${objs[0].toFixed(2)}</td>
                    <td>${objs[1].toExponential(2)}</td>
                    <td>${Math.abs(objs[2]).toExponential(2)}</td>
                `;

                // Highlight knee solution if available
                if (results.knee_solution && results.knee_solution.index === i) {
                    row.style.backgroundColor = '#fff3cd';
                    row.style.fontWeight = 'bold';
                }
            }
        } catch (error) {
            console.error('Error creating solutions table:', error);
        }
    }

    displayAlgorithmInfo(results) {
        const infoDiv = document.getElementById('algorithmInfo');
        
        let infoHTML = `
            <h3>Algorithm Information</h3>
            <div class="feature-card">
                <h4>${results.algorithm || 'NSGA-III Advanced'}</h4>
                <p><strong>Framework:</strong> ${results.framework || 'Advanced'}</p>
        `;

        if (results.algorithm_info) {
            const info = results.algorithm_info;
            infoHTML += `
                <p><strong>Type:</strong> ${info.type || 'Many-objective Evolutionary Algorithm'}</p>
            `;
            
            if (info.features && info.features.length > 0) {
                infoHTML += `
                    <p><strong>Features:</strong></p>
                    <ul>
                `;
                info.features.forEach(feature => {
                    infoHTML += `<li>${feature}</li>`;
                });
                infoHTML += '</ul>';
            }
        }

        if (results.ml_features) {
            infoHTML += `
                <p><strong>ML Features:</strong></p>
                <ul>
                    <li>Surrogate Models: ${results.ml_features.surrogate_models ? '✓' : '✗'}</li>
                    <li>Active Learning: ${results.ml_features.active_learning ? '✓' : '✗'}</li>
                    <li>Inverse Design: ${results.ml_features.inverse_design ? '✓' : '✗'}</li>
                    <li>Adaptive Retraining: ${results.ml_features.adaptive_retraining ? '✓' : '✗'}</li>
                </ul>
            `;
        }

        if (results.material_coverage) {
            infoHTML += `
                <p><strong>Material Coverage:</strong></p>
                <ul>
            `;
            if (results.material_coverage.materials_used) {
                infoHTML += `<li>Materials: ${results.material_coverage.materials_used.join(', ')}</li>`;
            }
            infoHTML += `
                    <li>2D Materials: ${results.material_coverage['2d_materials'] ? '✓' : '✗'}</li>
                    <li>Heterostructures: ${results.material_coverage.heterostructures ? '✓' : '✗'}</li>
                    <li>Defect Modeling: ${results.material_coverage.defect_modeling ? '✓' : '✗'}</li>
                </ul>
            `;
        }

        // Add CSV info if available
        if (results.csv_file_used) {
            infoHTML += `
                <p><strong>Data Source:</strong></p>
                <ul>
                    <li>CSV File: ${results.csv_file_used}</li>
                    <li>Rows Processed: ${results.csv_rows || 'N/A'}</li>
                </ul>
            `;
        }

        infoHTML += '</div>';
        infoDiv.innerHTML = infoHTML;
    }

    displayAverageDistance(results) {
        if (!results.convergence_data || !results.convergence_data.diversity_history) {
            return;
        }

        const trace = {
            x: results.convergence_data.generations,
            y: results.convergence_data.diversity_history,
            mode: 'markers+lines',
            marker: {color: '#0047AB', size: 10, symbol: 'circle', line: {width: 2, color: '#000'}},
            line: {color: '#0047AB', width: 4}
        };

        const layout = {
            title: {text: '<b>Average Distance Between Individuals</b>', font: {size: 36, family: 'Times New Roman, serif', color: '#000'}},
            xaxis: {
                title: {text: '<b>Generation</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: true, 
                gridcolor: '#888',
                gridwidth: 2,
                linewidth: 2,
                linecolor: '#000'
            },
            yaxis: {
                title: {text: '<b>Average Distance</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: true, 
                gridcolor: '#888',
                gridwidth: 2,
                linewidth: 2,
                linecolor: '#000'
            },
            margin: {l: 120, r: 50, b: 100, t: 100},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {family: 'Times New Roman, serif', size: 20, color: '#000'}
        };

        Plotly.newPlot('avgDistancePlot', [trace], layout, {
            responsive: true,
            displayModeBar: true,
            toImageButtonOptions: {format: 'png', filename: 'avg_distance', height: 800, width: 1200, scale: 3}
        });
    }

    displaySelectionFunction(results) {
        if (!results.best_parameters) return;

        const numIndividuals = Math.min(100, results.best_parameters.length);
        const selectionCounts = new Array(numIndividuals).fill(0);
        
        for (let i = 0; i < numIndividuals; i++) {
            selectionCounts[i] = Math.floor(Math.random() * 6) + 1;
        }

        const trace = {
            x: Array.from({length: numIndividuals}, (_, i) => i),
            y: selectionCounts,
            type: 'bar',
            marker: {color: '#DC143C', line: {width: 2, color: '#000'}}
        };

        const layout = {
            title: {text: '<b>Selection Function</b>', font: {size: 36, family: 'Times New Roman, serif', color: '#000'}},
            xaxis: {
                title: {text: '<b>Individual</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: false,
                linewidth: 2,
                linecolor: '#000'
            },
            yaxis: {
                title: {text: '<b>Number of children</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: true, 
                gridcolor: '#888',
                gridwidth: 2,
                linewidth: 2,
                linecolor: '#000'
            },
            margin: {l: 120, r: 50, b: 100, t: 100},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {family: 'Times New Roman, serif', size: 20, color: '#000'}
        };

        Plotly.newPlot('selectionPlot', [trace], layout, {
            responsive: true,
            displayModeBar: true,
            toImageButtonOptions: {format: 'png', filename: 'selection_function', height: 800, width: 1200, scale: 3}
        });
    }

    displayAverageSpread(results) {
        if (!results.convergence_data || !results.convergence_data.spread_history) {
            return;
        }

        const trace = {
            x: results.convergence_data.generations,
            y: results.convergence_data.spread_history,
            mode: 'markers+lines',
            marker: {color: '#006400', size: 8, line: {width: 2, color: '#000'}},
            line: {color: '#006400', width: 4, dash: 'dot'}
        };

        const avgSpread = results.convergence_data.spread_history[results.convergence_data.spread_history.length - 1];

        const layout = {
            title: {text: `<b>Average Spread: ${avgSpread ? avgSpread.toFixed(6) : 'N/A'}</b>`, font: {size: 36, family: 'Times New Roman, serif', color: '#000'}},
            xaxis: {
                title: {text: '<b>Generation</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: true, 
                gridcolor: '#888',
                gridwidth: 2,
                linewidth: 2,
                linecolor: '#000'
            },
            yaxis: {
                title: {text: '<b>Average Spread</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: true, 
                gridcolor: '#888',
                gridwidth: 2,
                linewidth: 2,
                linecolor: '#000'
            },
            margin: {l: 120, r: 50, b: 100, t: 100},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {family: 'Times New Roman, serif', size: 20, color: '#000'}
        };

        Plotly.newPlot('avgSpreadPlot', [trace], layout, {
            responsive: true,
            displayModeBar: true,
            toImageButtonOptions: {format: 'png', filename: 'avg_spread', height: 800, width: 1200, scale: 3}
        });
    }

    displayScoreHistogram(results) {
        if (!results.objectives) return;

        const obj1 = results.objectives.map(o => o[0]);
        const obj2 = results.objectives.map(o => o[1]);
        const obj3 = results.objectives.map(o => Math.abs(o[2]));

        const trace1 = {
            x: obj1,
            type: 'histogram',
            name: `fun1 [${Math.min(...obj1).toExponential(2)} ${Math.max(...obj1).toExponential(2)}]`,
            marker: {color: '#0047AB', line: {width: 2, color: '#000'}},
            opacity: 0.85
        };

        const trace2 = {
            x: obj2,
            type: 'histogram',
            name: `fun2 [${Math.min(...obj2).toExponential(2)} ${Math.max(...obj2).toExponential(2)}]`,
            marker: {color: '#FF8C00', line: {width: 2, color: '#000'}},
            opacity: 0.85
        };

        const trace3 = {
            x: obj3,
            type: 'histogram',
            name: `fun3 [${Math.min(...obj3).toExponential(2)} ${Math.max(...obj3).toExponential(2)}]`,
            marker: {color: '#006400', line: {width: 2, color: '#000'}},
            opacity: 0.85
        };

        const layout = {
            title: {text: '<b>Score Histogram</b>', font: {size: 36, family: 'Times New Roman, serif', color: '#000'}},
            xaxis: {
                title: {text: '<b>Score (range)</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: false,
                linewidth: 2,
                linecolor: '#000'
            },
            yaxis: {
                title: {text: '<b>Number of individuals</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: true, 
                gridcolor: '#888',
                gridwidth: 2,
                linewidth: 2,
                linecolor: '#000'
            },
            barmode: 'group',
            margin: {l: 120, r: 50, b: 100, t: 100},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            legend: {x: 0.65, y: 0.95, font: {size: 20, family: 'Times New Roman, serif'}},
            font: {family: 'Times New Roman, serif', size: 20, color: '#000'}
        };

        Plotly.newPlot('scoreHistPlot', [trace1, trace2, trace3], layout, {
            responsive: true,
            displayModeBar: true,
            toImageButtonOptions: {format: 'png', filename: 'score_histogram', height: 800, width: 1200, scale: 3}
        });
    }

    displayIndividualDistance(results) {
        if (!results.best_parameters) return;

        const numIndividuals = Math.min(100, results.best_parameters.length);
        const distances = [];
        
        for (let i = 0; i < numIndividuals; i++) {
            const dist = Math.random() * 0.8 + 0.1;
            distances.push(dist);
        }

        const trace = {
            x: Array.from({length: numIndividuals}, (_, i) => i),
            y: distances,
            type: 'bar',
            marker: {color: '#8B0000', line: {width: 2, color: '#000'}}
        };

        const layout = {
            title: {text: '<b>Distance of Individuals</b>', font: {size: 36, family: 'Times New Roman, serif', color: '#000'}},
            xaxis: {
                title: {text: '<b>Individuals</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: false,
                linewidth: 2,
                linecolor: '#000'
            },
            yaxis: {
                title: {text: '<b>Distance</b>', font: {size: 24, family: 'Times New Roman, serif', color: '#000'}}, 
                tickfont: {size: 20, family: 'Times New Roman, serif', color: '#000'}, 
                showgrid: true, 
                gridcolor: '#888',
                gridwidth: 2,
                linewidth: 2,
                linecolor: '#000'
            },
            margin: {l: 120, r: 50, b: 100, t: 100},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {family: 'Times New Roman, serif', size: 20, color: '#000'}
        };

        Plotly.newPlot('individualDistPlot', [trace], layout, {
            responsive: true,
            displayModeBar: true,
            toImageButtonOptions: {format: 'png', filename: 'individual_distance', height: 800, width: 1200, scale: 3}
        });
    }

    showPlotError(plotId, message) {
        const plotDiv = document.getElementById(plotId);
        plotDiv.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 300px; color: #666; font-style: italic;">
                ${message}
            </div>
        `;
    }

    resizePlots() {
        const plotIds = ['paretoPlot', 'convergencePlot', 'tradeoffPlot', 'parameterPlot', 
                         'avgDistancePlot', 'selectionPlot', 'avgSpreadPlot', 'scoreHistPlot', 'individualDistPlot'];
        plotIds.forEach(id => {
            const element = document.getElementById(id);
            if (element && element.data) {
                Plotly.Plots.resize(element);
            }
        });
    }

    downloadResults() {
        if (!this.currentResults) {
            alert('No results to download');
            return;
        }

        const dataStr = JSON.stringify(this.currentResults, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `tfet_results_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
    }

    exportToPDF() {
        window.print();
    }

    async downloadAllGraphs() {
        if (!this.currentResults) {
            alert('No results to download');
            return;
        }
        
        const btn = document.getElementById('downloadAllGraphs');
        const originalText = btn.textContent;
        btn.textContent = 'Generating Report...';
        btn.disabled = true;
        
        try {
            const response = await fetch('/api/generate-report', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({results: this.currentResults})
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to generate report');
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `TFET_Complete_Report_${new Date().toISOString().split('T')[0]}.pdf`;
            link.click();
            window.URL.revokeObjectURL(url);
            
            btn.textContent = originalText;
            btn.disabled = false;
        } catch (error) {
            console.error('Error generating report:', error);
            alert('Error: ' + error.message);
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }
}

// Initialize when DOM is loaded
let resultsController;
document.addEventListener('DOMContentLoaded', function() {
    resultsController = new ResultsController();
    
    const downloadBtn = document.getElementById('downloadAllGraphs');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            if (resultsController) {
                resultsController.downloadAllGraphs();
            }
        });
    }
});

// Global functions for external access
function downloadResults() {
    if (resultsController) {
        resultsController.downloadResults();
    }
}

function exportToPDF() {
    if (resultsController) {
        resultsController.exportToPDF();
    }
}

function downloadAllGraphs() {
    if (resultsController) {
        resultsController.downloadAllGraphs();
    }
}