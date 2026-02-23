import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class ParetoFrontVisualizer:
    
    def __init__(self):
        plt.style.use('default')
        # Publication-quality settings for research papers
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['savefig.format'] = 'png'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['font.size'] = 24
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 28
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 24
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['legend.fontsize'] = 20
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.edgecolor'] = 'black'
        plt.rcParams['legend.fancybox'] = False
        plt.rcParams['grid.linewidth'] = 1.5
        plt.rcParams['lines.linewidth'] = 3
        plt.rcParams['lines.markersize'] = 10
        
    def plot_3d_pareto(self, objectives, title="3D Pareto Front"):
        """Plot 3D Pareto front for TFET objectives"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if objectives.shape[1] >= 3:
            scatter = ax.scatter(objectives[:, 0], objectives[:, 1], -objectives[:, 2],
                               c=-objectives[:, 2], cmap='jet', s=100, alpha=0.9, 
                               edgecolors='black', linewidth=1.5, depthshade=True)
            
            ax.set_xlabel('Natural Length (nm)', fontweight='bold', fontsize=24, labelpad=15)
            ax.set_ylabel('Vertical E-field (V/m)', fontweight='bold', fontsize=24, labelpad=15)
            ax.set_zlabel('Ion/Ioff Ratio', fontweight='bold', fontsize=24, labelpad=15)
            ax.set_title(title, fontweight='bold', fontsize=28, pad=25)
            ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
            ax.grid(True, alpha=0.3, linewidth=1.5)
            
            cbar = plt.colorbar(scatter, shrink=0.7, aspect=15, pad=0.1)
            cbar.set_label('Ion/Ioff Ratio', fontweight='bold', fontsize=24, labelpad=15)
            cbar.ax.tick_params(labelsize=18, width=2, length=6)
            cbar.outline.set_linewidth(2)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_pareto_with_knee(self, objectives, knee_idx=None, title="3D Pareto Front with Knee Point"):
        """Plot 3D Pareto front with highlighted knee point"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if objectives.shape[1] >= 3:
            scatter = ax.scatter(objectives[:, 0], objectives[:, 1], -objectives[:, 2],
                               c=-objectives[:, 2], cmap='jet', s=120, alpha=0.9, 
                               edgecolors='black', linewidth=1.5, depthshade=True)
            
            if knee_idx is not None and knee_idx < len(objectives):
                ax.scatter(objectives[knee_idx, 0], objectives[knee_idx, 1], -objectives[knee_idx, 2],
                          c='#FF0000', s=500, marker='*', label='Knee Point', 
                          edgecolors='#000000', linewidth=3, zorder=10)
                ax.legend(fontsize=22, fontweight='bold', loc='upper right', 
                         frameon=True, edgecolor='black', fancybox=False, shadow=False)
            
            ax.set_xlabel('Natural Length (nm)', fontweight='bold', fontsize=24, labelpad=15)
            ax.set_ylabel('Vertical E-field (V/m)', fontweight='bold', fontsize=24, labelpad=15)
            ax.set_zlabel('Ion/Ioff Ratio', fontweight='bold', fontsize=24, labelpad=15)
            ax.set_title(title, fontweight='bold', fontsize=28, pad=25)
            ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
            ax.grid(True, alpha=0.3, linewidth=1.5)
            
            cbar = plt.colorbar(scatter, shrink=0.7, aspect=15, pad=0.1)
            cbar.set_label('Ion/Ioff Ratio', fontweight='bold', fontsize=24, labelpad=15)
            cbar.ax.tick_params(labelsize=18, width=2, length=6)
            cbar.outline.set_linewidth(2)
        
        plt.tight_layout()
        return fig
    
    def plot_convergence_analysis(self, convergence_history):
        """Plot convergence analysis"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if convergence_history:
            generations = list(range(1, len(convergence_history) + 1))
            ax.plot(generations, convergence_history, color='#0047AB', linewidth=4, 
                   marker='o', markersize=12, markerfacecolor='#8B0000', 
                   markeredgecolor='black', markeredgewidth=2.5)
            ax.set_xlabel('Generation', fontweight='bold', fontsize=24, labelpad=10)
            ax.set_ylabel('Hypervolume Indicator', fontweight='bold', fontsize=24, labelpad=10)
            ax.set_title('Convergence Analysis', fontweight='bold', fontsize=28, pad=20)
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5, color='gray')
            ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        else:
            ax.text(0.5, 0.5, 'No convergence data available', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=24, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_advanced_tfet_analysis(self, result, analysis=None):
        """Complete advanced TFET analysis with multiple plots"""
        if result is None or result.F is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No optimization results available', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
            
        objectives = result.F
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 2D projections with high-contrast colors
        ax1.scatter(objectives[:, 0], objectives[:, 1], c='#DC143C', alpha=0.85, s=150, 
                   edgecolors='black', linewidth=2)
        ax1.set_xlabel('Natural Length (nm)', fontweight='bold', fontsize=24, labelpad=10)
        ax1.set_ylabel('Vertical E-field (V/m)', fontweight='bold', fontsize=24, labelpad=10)
        ax1.set_title('Natural Length vs Vertical E-field', fontweight='bold', fontsize=28, pad=15)
        ax1.grid(True, alpha=0.4, linestyle='--', linewidth=1.5, color='gray')
        ax1.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
        for spine in ax1.spines.values():
            spine.set_linewidth(2)
        
        ax2.scatter(objectives[:, 0], -objectives[:, 2], c='#006400', alpha=0.85, s=150, 
                   edgecolors='black', linewidth=2)
        ax2.set_xlabel('Natural Length (nm)', fontweight='bold', fontsize=24, labelpad=10)
        ax2.set_ylabel('Ion/Ioff Ratio', fontweight='bold', fontsize=24, labelpad=10)
        ax2.set_title('Natural Length vs Ion/Ioff Ratio', fontweight='bold', fontsize=28, pad=15)
        ax2.grid(True, alpha=0.4, linestyle='--', linewidth=1.5, color='gray')
        ax2.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
        for spine in ax2.spines.values():
            spine.set_linewidth(2)
        
        ax3.scatter(objectives[:, 1], -objectives[:, 2], c='#FF8C00', alpha=0.85, s=150, 
                   edgecolors='black', linewidth=2)
        ax3.set_xlabel('Vertical E-field (V/m)', fontweight='bold', fontsize=24, labelpad=10)
        ax3.set_ylabel('Ion/Ioff Ratio', fontweight='bold', fontsize=24, labelpad=10)
        ax3.set_title('Vertical E-field vs Ion/Ioff Ratio', fontweight='bold', fontsize=28, pad=15)
        ax3.grid(True, alpha=0.4, linestyle='--', linewidth=1.5, color='gray')
        ax3.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
        for spine in ax3.spines.values():
            spine.set_linewidth(2)
        
        # Statistics with better formatting
        ax4.text(0.1, 0.85, f'Solutions: {len(objectives)}', transform=ax4.transAxes, fontsize=24, fontweight='bold')
        ax4.text(0.1, 0.75, f'Avg Natural Length: {np.mean(objectives[:, 0]):.2f} nm', transform=ax4.transAxes, fontsize=20)
        ax4.text(0.1, 0.65, f'Avg E-field: {np.mean(objectives[:, 1]):.2e} V/m', transform=ax4.transAxes, fontsize=20)
        ax4.text(0.1, 0.55, f'Max Ion/Ioff: {np.max(-objectives[:, 2]):.2e}', transform=ax4.transAxes, fontsize=20)
        
        if analysis and 'statistics' in analysis:
            stats = analysis['statistics']
            ax4.text(0.1, 0.45, f'Diversity: {stats.get("diversity_metric", 0):.3f}', transform=ax4.transAxes, fontsize=20)
        
        ax4.text(0.1, 0.35, 'Algorithm: NSGA-III', transform=ax4.transAxes, fontsize=20, style='italic')
        ax4.text(0.1, 0.25, 'Framework: Advanced ML', transform=ax4.transAxes, fontsize=20, style='italic')
        
        ax4.set_title('Optimization Statistics', fontweight='bold', fontsize=28, pad=15)
        ax4.axis('off')
        ax4.patch.set_facecolor('#F8F9FA')
        ax4.patch.set_alpha(0.8)
        
        plt.tight_layout()
        return fig
    
    def plot_tfet_pareto_analysis(self, result):
        """Complete TFET Pareto front analysis (backward compatibility)"""
        return self.plot_advanced_tfet_analysis(result)
