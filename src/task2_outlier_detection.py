"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}
        
    def _numeric_data(self) -> pd.DataFrame:
        """Return only the numeric columns of the working dataset."""
        return self.data.select_dtypes(include=[np.number])

    def _multivariate_mask(self, flags: np.ndarray) -> pd.DataFrame:
        """Wrap row-level outlier flags (Isolation Forest, LOF) into a one-column mask."""
        return pd.DataFrame({'is_outlier': np.asarray(flags, dtype=bool)}, index=self.data.index)

    def _get_mask(self, method: str) -> pd.DataFrame:
        """
        Return the stored mask of a detection method, running it with defaults if needed.

        Raises:
            ValueError: If the method name is unknown.
        """
        detectors = {
            'zscore': self.detect_outliers_zscore,
            'iqr': self.detect_outliers_iqr,
            'isolation_forest': self.detect_outliers_isolation_forest,
            'lof': self.detect_outliers_lof,
        }
        if method not in detectors:
            raise ValueError(f"Unknown outlier method '{method}'. Choose from: {list(detectors)}")
        if method not in self.outlier_masks:
            detectors[method]()
        return self.outlier_masks[method]

    @staticmethod
    def _row_flags(mask: pd.DataFrame, columns: List[str] = None) -> pd.Series:
        """Collapse a mask to one flag per row, optionally looking only at some columns."""
        selected = [col for col in (columns or []) if col in mask.columns]
        return mask[selected].any(axis=1) if selected else mask.any(axis=1)

    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Parameters:
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric = self._numeric_data()
        std = numeric.std(ddof=0).replace(0, np.nan)  # constant columns can have no outliers
        z_scores = (numeric - numeric.mean()) / std

        mask = z_scores.abs() > threshold  # NaN comparisons evaluate to False
        self.outlier_masks['zscore'] = mask
        return mask
    
    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
            multiplier (float): IQR multiplier for boundary calculation
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric = self._numeric_data()
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        mask = (numeric < lower) | (numeric > upper)
        self.outlier_masks['iqr'] = mask
        return mask
    
    def detect_outliers_isolation_forest(self, contamination: float = 0.1, 
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric = self._numeric_data()
        if numeric.shape[1] == 0:
            raise ValueError("No numeric columns available for outlier detection")

        features = numeric.fillna(numeric.median())
        model = IsolationForest(contamination=contamination, random_state=random_state)
        flags = model.fit_predict(features) == -1  # -1 marks an outlier

        mask = self._multivariate_mask(flags)
        self.outlier_masks['isolation_forest'] = mask
        return mask
    
    def detect_outliers_lof(self, n_neighbors: int = 20, 
                           contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.
        
        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        numeric = self._numeric_data()
        if numeric.shape[1] == 0:
            raise ValueError("No numeric columns available for outlier detection")

        # LOF is distance based, so features must be on a comparable scale
        features = StandardScaler().fit_transform(numeric.fillna(numeric.median()))
        neighbors = max(1, min(n_neighbors, len(features) - 1))
        model = LocalOutlierFactor(n_neighbors=neighbors, contamination=contamination)
        flags = model.fit_predict(features) == -1  # -1 marks an outlier

        mask = self._multivariate_mask(flags)
        self.outlier_masks['lof'] = mask
        return mask
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.
        
        Returns:
            DataFrame with outlier counts for each method
        """
        if not self.outlier_masks:
            return pd.DataFrame()

        features = list(self._numeric_data().columns)
        n_rows = len(self.data)
        summary = {}

        for method, mask in self.outlier_masks.items():
            row_flags = mask.any(axis=1)
            column_wise = set(features).issubset(mask.columns)
            counts = {
                # Isolation Forest / LOF flag whole rows, so they have no per-feature counts
                feature: (int(mask[feature].sum()) if column_wise else np.nan)
                for feature in features
            }
            counts['rows_flagged'] = int(row_flags.sum())
            counts['percent_flagged'] = float(row_flags.sum()) * 100.0 / n_rows if n_rows else 0.0
            summary[method] = counts

        return pd.DataFrame(summary)
    
    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters:
            method (str): Detection method to use
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        mask = self._get_mask(method)
        cleaned = self.data.loc[~mask.any(axis=1)].copy()

        self.treated_datasets[f'removed_{method}'] = cleaned
        return cleaned
    
    def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.
        
        Parameters:
            method (str): Detection method to use
            multiplier (float): IQR multiplier for bounds
            
        Returns:
            pd.DataFrame: Dataset with capped outliers
        """
        numeric = self._numeric_data()

        if method == 'iqr':
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
        elif method == 'zscore':
            # For the Z-score method `multiplier` is the number of standard deviations
            lower = numeric.mean() - multiplier * numeric.std()
            upper = numeric.mean() + multiplier * numeric.std()
        else:
            raise ValueError(f"Unknown capping method '{method}'. Use 'iqr' or 'zscore'.")

        capped = self.data.copy()
        capped[numeric.columns] = numeric.clip(lower=lower, upper=upper, axis=1)

        self.treated_datasets['capped'] = capped
        return capped
    
    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.
        
        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """
        transformed = self.data.copy()
        numeric = self._numeric_data()

        for col in numeric.columns:
            values = numeric[col]
            min_value = values.min()
            # log1p needs values > -1, so shift columns with negatives to start at 0
            if min_value < 0:
                values = values - min_value
            transformed[col] = np.log1p(values)

        self.treated_datasets['log_transform'] = transformed
        return transformed
    
    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.
        
        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """
        transformed = self.data.copy()
        numeric = self._numeric_data()

        # Yeo-Johnson (unlike Box-Cox) accepts zero and negative values; constant columns carry no signal
        varying = [col for col in numeric.columns if numeric[col].nunique(dropna=True) > 1]
        if varying:
            transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            transformed[varying] = transformer.fit_transform(numeric[varying])

        self.treated_datasets['boxcox_transform'] = transformed
        return transformed
    
    def evaluate_model_performance(self, target_col: str, 
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.
        
        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)
            
        Returns:
            Dictionary with performance metrics
        """
        if dataset_name == 'original':
            dataset = self.data
        elif dataset_name in self.treated_datasets:
            dataset = self.treated_datasets[dataset_name]
        else:
            available = ['original'] + list(self.treated_datasets)
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")

        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        numeric = dataset.select_dtypes(include=[np.number]).dropna()
        features = numeric.drop(columns=[target_col])
        target = numeric[target_col]

        folds = min(5, len(numeric))
        if features.shape[1] == 0 or folds < 2:
            raise ValueError("Not enough numeric features or samples to evaluate a model")

        scores = cross_val_score(LinearRegression(), features, target, cv=folds, scoring='r2')

        return {
            'r2_score': float(scores.mean()),
            'r2_std': float(scores.std()),
            'mean_cv_score': float(scores.mean()),
            'n_samples': int(len(numeric)),
        }
    
    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return

        n_rows = (len(numeric_cols) + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for ax, col in zip(axes, numeric_cols):
            values = self.original_data[col].dropna()
            ax.boxplot(values, patch_artist=True,
                       boxprops=dict(facecolor='skyblue'),
                       medianprops=dict(color='darkblue', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='red', markeredgecolor='red', markersize=5))

            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            n_outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
            ax.set_title(f'{col} ({n_outliers} outliers)', fontweight='bold')
            ax.set_ylabel('Value')
            ax.set_xticks([])
            ax.grid(axis='y', alpha=0.3)

        for ax in axes[len(numeric_cols):]:
            ax.axis('off')

        plt.suptitle('Box Plots: Outlier Detection', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return

        n_rows = (len(numeric_cols) + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for ax, col in zip(axes, numeric_cols):
            values = self.data[col].dropna()
            sns.violinplot(y=values, ax=ax, color='lightblue', inner='quartile')
            ax.axhline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {values.mean():.2f}')
            ax.axhline(values.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {values.median():.2f}')
            ax.set_title(col, fontweight='bold')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        for ax in axes[len(numeric_cols):]:
            ax.axis('off')

        plt.suptitle('Violin Plots: Distribution Density', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_outliers_scatter(self, x_col: str, y_col: str, 
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Create scatter plots showing outliers detected by different methods.
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """
        for col in (x_col, y_col):
            if col not in self.data.columns:
                print(f"Column '{col}' not found")
                return

        methods = [('zscore', 'Z-Score'), ('iqr', 'IQR'), ('isolation_forest', 'Isolation Forest')]
        fig, axes = plt.subplots(1, len(methods), figsize=figsize)

        for ax, (method, title) in zip(axes, methods):
            flags = self._row_flags(self._get_mask(method), [x_col, y_col])
            normal = self.data[~flags]
            outliers = self.data[flags]

            ax.scatter(normal[x_col], normal[y_col], c='steelblue', alpha=0.6, s=25, label='Normal')
            ax.scatter(outliers[x_col], outliers[y_col], c='red', marker='x', s=50, label='Outlier')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'{title} ({int(flags.sum())} outliers)', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle(f'Outlier Detection: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.
        
        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """
        from mpl_toolkits.mplot3d import Axes3D

        for col in (x_col, y_col, z_col):
            if col not in self.data.columns:
                print(f"Column '{col}' not found")
                return

        if method not in self.outlier_masks:
            print(f"Outlier mask for method '{method}' not found. Run the detection first.")
            return

        flags = self._row_flags(self.outlier_masks[method], [x_col, y_col, z_col])
        normal = self.data[~flags]
        outliers = self.data[flags]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(normal[x_col], normal[y_col], normal[z_col], c='steelblue', alpha=0.5, s=20, label='Normal')
        ax.scatter(outliers[x_col], outliers[y_col], outliers[z_col], c='red', marker='x', s=60, label='Outlier')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'3D Outliers ({method.replace("_", " ").title()}): {int(flags.sum())} detected',
                     fontweight='bold')
        ax.legend()

        plt.tight_layout()
        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
