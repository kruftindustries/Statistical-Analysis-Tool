import sys
import csv
import math
from typing import List, Dict
from statistics import mean, median, stdev, variance
from collections import defaultdict
import openpyxl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QComboBox, QLabel, QTableView, QTabWidget, QListWidget,
    QGridLayout, QSpinBox, QScrollArea, QMessageBox, QCheckBox, QDoubleSpinBox,
    QGroupBox, QSlider
)
from PySide6.QtCore import Qt, QAbstractTableModel, QPointF
from PySide6.QtCharts import (
    QChart, QChartView, QScatterSeries, QValueAxis, QLineSeries, QBarSeries,
    QBarSet, QBarCategoryAxis, QSplineSeries
)
from PySide6.QtGui import QPen, QColor, QPainter, QVector3D
from PySide6.QtDataVisualization import (
    Q3DScatter, QScatter3DSeries, Q3DCamera, QAbstract3DGraph, QAbstract3DSeries, QValue3DAxis, QScatterDataProxy
)


class Matrix:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0] * cols for _ in range(rows)]
    
    def __getitem__(self, key):
        i, j = key
        
        return self.data[i][j]
    
    def __setitem__(self, key, value):
        i, j = key
        self.data[i][j] = value
    
    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        return result
    
    def multiply(self, other):
        # First matrix columns must equal second matrix rows
        if self.cols != other.rows:
            raise ValueError(f"Matrix dimensions don't match for multiplication: "
                            f"({self.rows}x{self.cols}) * ({other.rows}x{other.cols})")
        
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                sum_val = 0
                for k in range(self.cols):
                    sum_val += self[i, k] * other[k, j]
                result[i, j] = sum_val
        return result
    
    def inverse(self):
        if self.rows != self.cols:
            raise ValueError("Matrix must be square for inverse")
        
        n = self.rows
        augmented = [[0.0] * (2 * n) for _ in range(n)]
        
        # Create augmented matrix
        for i in range(n):
            for j in range(n):
                augmented[i][j] = self[i, j]
            augmented[i][i + n] = 1.0
        
        # Gaussian elimination
        for i in range(n):
            pivot = augmented[i][i]
            if abs(pivot) < 1e-10:
                raise ValueError("Matrix is singular")
            
            for j in range(2 * n):
                augmented[i][j] /= pivot
            
            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    for j in range(2 * n):
                        augmented[k][j] -= factor * augmented[i][j]
        
        # Extract inverse
        result = Matrix(n, n)
        for i in range(n):
            for j in range(n):
                result[i, j] = augmented[i][j + n]
        
        return result

class DataSet:
    def __init__(self):
        self.headers: List[str] = []
        self.data: List[List] = []
        self.numeric_columns: Dict[int, bool] = {}

    def load_file(self, filename: str):
        self.headers = []
        self.data = []
        self.numeric_columns = {}

        if filename.endswith('.csv'):
            self._load_csv(filename)
        else:
            self._load_xlsx(filename)

    def _load_csv(self, filename: str):
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            self.headers = next(csv_reader)
            self._process_data(csv_reader)

    def _load_xlsx(self, filename: str):
        workbook = openpyxl.load_workbook(filename)
        sheet = workbook.active
        
        # Get headers from first row
        self.headers = [str(cell.value) for cell in next(sheet.rows)]
        
        # Convert remaining rows to list
        data_rows = []
        for row in sheet.iter_rows(min_row=2):
            data_rows.append([cell.value for cell in row])
        
        self._process_data(data_rows)

    def _process_data(self, data_rows):
        # Initialize numeric column detection
        for i in range(len(self.headers)):
            self.numeric_columns[i] = True
        
        # Process data and detect numeric columns
        for row in data_rows:
            processed_row = []
            for i, value in enumerate(row):
                try:
                    processed_row.append(float(value))
                except (ValueError, TypeError):
                    processed_row.append(value)
                    self.numeric_columns[i] = False
            self.data.append(processed_row)

    def get_column_data(self, column_index: int) -> List:
        return [row[column_index] for row in self.data]

    def get_numeric_columns(self) -> List[str]:
        return [header for i, header in enumerate(self.headers) 
                if self.numeric_columns[i]]

    def calculate_statistics(self, column_index: int) -> Dict:
        values = [float(x) for x in self.get_column_data(column_index)]
        stats = {
            'mean': mean(values),
            'median': median(values),
            'std_dev': stdev(values) if len(values) > 1 else 0,
            'variance': variance(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values)
        }
        return stats

    def calculate_correlation(self, col1: int, col2: int) -> float:
        x = [float(val) for val in self.get_column_data(col1)]
        y = [float(val) for val in self.get_column_data(col2)]
        
        if len(x) != len(y):
            return 0
        
        n = len(x)
        mean_x = mean(x)
        mean_y = mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = (sum((x[i] - mean_x) ** 2 for i in range(n)) * 
                      sum((y[i] - mean_y) ** 2 for i in range(n))) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0

    def multiple_regression(self, dependent_idx: int, independent_idxs: List[int]) -> Dict:
        # Get data
        y = [float(val) for val in self.get_column_data(dependent_idx)]
        X = [[1.0] + [float(row[idx]) for idx in independent_idxs] 
             for row in self.data]
        
        # Create matrices
        X_matrix = Matrix(len(X), len(X[0]))
        y_matrix = Matrix(len(y), 1)
        
        for i in range(len(X)):
            for j in range(len(X[0])):
                X_matrix[i, j] = X[i][j]
            y_matrix[i, 0] = y[i]
        
        # Calculate coefficients: β = (X'X)^(-1)X'y
        X_transpose = X_matrix.transpose()
        XtX = X_transpose.multiply(X_matrix)
        XtX_inv = XtX.inverse()
        Xty = X_transpose.multiply(y_matrix)
        beta = XtX_inv.multiply(Xty)
        
        # Calculate R-squared
        y_pred = X_matrix.multiply(beta)
        y_mean = mean(y)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((y[i] - y_pred[i, 0]) ** 2 for i in range(len(y)))
        r_squared = 1 - (ss_res / ss_tot)
        
        # Prepare results
        coefficients = [beta[i, 0] for i in range(beta.rows)]
        variables = ['intercept'] + [self.headers[idx] for idx in independent_idxs]
        
        return {
            'coefficients': dict(zip(variables, coefficients)),
            'r_squared': r_squared
        }

    def cluster_analysis(self, column_indices: List[int], n_clusters: int) -> Dict:
        # K-means clustering
        data_points = [[float(row[idx]) for idx in column_indices] 
                      for row in self.data]
        
        # Initialize centroids randomly
        centroids = data_points[:n_clusters]
        old_centroids = None
        clusters = [0] * len(data_points)
        
        # Iterate until convergence
        while old_centroids != centroids:
            old_centroids = [c[:] for c in centroids]
            
            # Assign points to nearest centroid
            for i, point in enumerate(data_points):
                min_dist = float('inf')
                for j, centroid in enumerate(centroids):
                    dist = sum((p - c) ** 2 for p, c in zip(point, centroid))
                    if dist < min_dist:
                        min_dist = dist
                        clusters[i] = j
            
            # Update centroids
            for i in range(n_clusters):
                cluster_points = [p for j, p in enumerate(data_points) 
                                if clusters[j] == i]
                if cluster_points:
                    centroids[i] = [mean(dim) for dim in zip(*cluster_points)]
        
        return {
            'clusters': clusters,
            'centroids': centroids
        }

    def factor_analysis(self, column_indices: List[int], n_factors: int) -> Dict:
        # Simple implementation of principal component analysis
        # Standardize data
        data = [[float(row[idx]) for idx in column_indices] 
                for row in self.data]
        
        means = [mean(col) for col in zip(*data)]
        stds = [stdev(col) if len(set(col)) > 1 else 1.0 
                for col in zip(*data)]
        
        standardized = [[((val - means[j]) / stds[j]) 
                        for j, val in enumerate(row)] 
                       for row in data]
        
        # Calculate correlation matrix
        n_vars = len(column_indices)
        corr_matrix = Matrix(n_vars, n_vars)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    col_i = [row[i] for row in standardized]
                    col_j = [row[j] for row in standardized]
                    corr_matrix[i, j] = sum(x * y for x, y 
                                          in zip(col_i, col_j)) / (len(col_i) - 1)
        
        # Simple power iteration to find principal components
        factors = []
        for _ in range(n_factors):
            vector = [1.0] * n_vars
            for _ in range(100):  # Number of iterations
                new_vector = [0.0] * n_vars
                for i in range(n_vars):
                    for j in range(n_vars):
                        new_vector[i] += corr_matrix[i, j] * vector[j]
                
                # Normalize
                magnitude = sum(x * x for x in new_vector) ** 0.5
                vector = [x / magnitude for x in new_vector]
            
            factors.append(vector)
        
        return {
            'loadings': factors,
            'variables': [self.headers[idx] for idx in column_indices]
        }

class DataTableModel(QAbstractTableModel):
    def __init__(self, dataset: DataSet):
        super().__init__()
        self.dataset = dataset

    def rowCount(self, index):
        return len(self.dataset.data)

    def columnCount(self, index):
        return len(self.dataset.headers)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self.dataset.data[index.row()][index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self.dataset.headers[section])
            if orientation == Qt.Vertical:
                return str(section + 1)
        return None

class NormalDistributionPlot(QWidget):
    def __init__(self, dataset: DataSet, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QHBoxLayout()
        self.variable_combo = QComboBox()
        controls.addWidget(QLabel("Variable:"))
        controls.addWidget(self.variable_combo)
        
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.create_plot)
        controls.addWidget(self.plot_button)
        
        layout.addLayout(controls)
        
        # Chart view
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(self.chart_view.renderHints())
        layout.addWidget(self.chart_view)
        
        # Statistics label
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)
        
        self.update_controls()

    def update_controls(self):
        self.variable_combo.clear()
        self.variable_combo.addItems(self.dataset.get_numeric_columns())

    def create_plot(self):
        if not self.variable_combo.currentText():
            return
            
        var_idx = self.dataset.headers.index(self.variable_combo.currentText())
        data = [float(x) for x in self.dataset.get_column_data(var_idx)]
        
        # Calculate normal distribution parameters
        mu = mean(data)
        sigma = stdev(data)
        
        # Create chart
        chart = QChart()
        
        # Create histogram series
        hist_series = QBarSeries()
        n_bins = 30
        min_val, max_val = min(data), max(data)
        bin_width = (max_val - min_val) / n_bins
        bins = [0] * n_bins
        
        for value in data:
            bin_idx = min(int((value - min_val) / bin_width), n_bins - 1)
            bins[bin_idx] += 1
        
        # Normalize histogram
        total = sum(bins)
        bins = [x / (total * bin_width) for x in bins]
        
        bar_set = QBarSet("Frequency")
        for count in bins:
            bar_set.append(count)
        hist_series.append(bar_set)
        chart.addSeries(hist_series)
        
        # Create normal distribution curve
        curve_series = QSplineSeries()
        curve_series.setName("Normal Distribution")
        
        # Generate points for the curve
        n_points = 100
        x_range = max_val - min_val
        for i in range(n_points):
            x = min_val + (i * x_range / (n_points - 1))
            y = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(x - mu)**2 / (2 * sigma**2))
            curve_series.append(x, y)
        
        chart.addSeries(curve_series)
        
        # Set up axes
        axis_x = QValueAxis()
        axis_y = QValueAxis()
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        
        hist_series.attachAxis(axis_x)
        hist_series.attachAxis(axis_y)
        curve_series.attachAxis(axis_x)
        curve_series.attachAxis(axis_y)
        
        self.chart_view.setChart(chart)
        
        # Update statistics
        stats = self.dataset.calculate_statistics(var_idx)
        stats_text = (f"Mean: {stats['mean']:.2f}\n"
                     f"Median: {stats['median']:.2f}\n"
                     f"Std Dev: {stats['std_dev']:.2f}\n"
                     f"Skewness: {self.calculate_skewness(data):.2f}\n"
                     f"Kurtosis: {self.calculate_kurtosis(data):.2f}")
        self.stats_label.setText(stats_text)

    def calculate_skewness(self, data):
        mu = mean(data)
        sigma = stdev(data)
        n = len(data)
        return (sum((x - mu)**3 for x in data) / n) / (sigma**3)

    def calculate_kurtosis(self, data):
        mu = mean(data)
        sigma = stdev(data)
        n = len(data)
        return (sum((x - mu)**4 for x in data) / n) / (sigma**4) - 3

class HistogramWithFitTab(QWidget):
    def __init__(self, dataset: DataSet, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        layout = QVBoxLayout(self)
        
        # Controls remain the same
        controls = QHBoxLayout()
        
        self.variable_combo = QComboBox()
        controls.addWidget(QLabel("Variable:"))
        controls.addWidget(self.variable_combo)
        
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 100)
        self.bins_spin.setValue(30)
        controls.addWidget(QLabel("Bins:"))
        controls.addWidget(self.bins_spin)
        
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 10.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        controls.addWidget(QLabel("Curve Scale:"))
        controls.addWidget(self.scale_spin)
        
        self.show_curve = QCheckBox("Show Bell Curve")
        self.show_curve.setChecked(True)
        controls.addWidget(self.show_curve)
        
        self.plot_button = QPushButton("Create Plot")
        self.plot_button.clicked.connect(self.create_plot)
        controls.addWidget(self.plot_button)
        
        layout.addLayout(controls)
        
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(self.chart_view.renderHints())
        layout.addWidget(self.chart_view)
        
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)
        
        self.update_controls()

    def update_controls(self):
        self.variable_combo.clear()
        self.variable_combo.addItems(self.dataset.get_numeric_columns())

    def create_plot(self):
        if not self.variable_combo.currentText():
            return
            
        var_idx = self.dataset.headers.index(self.variable_combo.currentText())
        data = [float(x) for x in self.dataset.get_column_data(var_idx)]
        
        # Calculate statistics
        mu = mean(data)
        sigma = stdev(data)
        
        # Create chart
        chart = QChart()
        chart.setTitle(f"Histogram with Normal Distribution - {self.variable_combo.currentText()}")
        
        # Create histogram series
        n_bins = self.bins_spin.value()
        min_val, max_val = min(data), max(data)
        bin_width = (max_val - min_val) / n_bins
        
        # Create histogram using numpy for accurate binning
        import numpy as np
        hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create bar series for histogram
        hist_series = QBarSeries()
        bar_set = QBarSet("Frequency")
        
        # Set the bar positions using a value axis
        for h in hist:
            bar_set.append(h)
        hist_series.append(bar_set)
        
        # Set the categories for correct x-axis positioning
        categories = []
        for center in bin_centers:
            categories.append(str(center))
        
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        
        chart.addSeries(hist_series)
        chart.addAxis(axis_x, Qt.AlignBottom)
        hist_series.attachAxis(axis_x)
        
        # Variable to store maximum y value
        max_y = max(hist)
        
        # Create normal distribution curve if enabled
        if self.show_curve.isChecked():
            curve_series = QLineSeries()
            curve_series.setName("Normal Distribution")
            
            # Generate points for the curve
            n_points = 200
            x_range = max_val - min_val
            x_padding = x_range * 0.2
            x_min = min_val - x_padding
            x_max = max_val + x_padding
            curve_scale = self.scale_spin.value()
            
            # Create points for the normal curve
            curve_y_vals = []
            for i in range(n_points):
                x = x_min + (i * (x_max - x_min) / (n_points - 1))
                # Calculate the normal distribution PDF
                y = (1 / (sigma * math.sqrt(2 * math.pi))) * \
                    math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
                # Scale to match histogram height
                y = y * curve_scale
                curve_y_vals.append(y)
                curve_series.append(x, y)
            
            max_y = max(max_y, max(curve_y_vals))
            
            pen = QPen(QColor("red"))
            pen.setWidth(2)
            curve_series.setPen(pen)
            chart.addSeries(curve_series)
            
            # Create value axis for curve
            value_axis_x = QValueAxis()
            value_axis_x.setRange(x_min, x_max)
            chart.addAxis(value_axis_x, Qt.AlignBottom)
            curve_series.attachAxis(value_axis_x)
            
            # Hide the category axis labels since we're using value axis
            axis_x.hide()
        
        # Set up y axis
        axis_y = QValueAxis()
        axis_y.setTitleText("Density")
        axis_y.setRange(0, max_y * 1.1)  # Add 10% padding
        
        chart.addAxis(axis_y, Qt.AlignLeft)
        hist_series.attachAxis(axis_y)
        if self.show_curve.isChecked():
            curve_series.attachAxis(axis_y)
        
        self.chart_view.setChart(chart)
        
        # Statistics calculation remains the same
        try:
            skewness = self.calculate_skewness(data)
            kurtosis = self.calculate_kurtosis(data)
            n = len(data)
            stats_text = (
                f"Distribution Statistics:\n"
                f"Mean: {mu:.2f}\n"
                f"Median: {median(data):.2f}\n"
                f"Std Dev: {sigma:.2f}\n"
                f"Skewness: {skewness:.2f}\n"
                f"Kurtosis: {kurtosis:.2f}\n"
                f"Sample Size: {n}\n"
                f"Standard Error: {sigma/math.sqrt(n):.2f}\n"
                f"95% Confidence Interval: [{mu - 1.96*sigma/math.sqrt(n):.2f}, "
                f"{mu + 1.96*sigma/math.sqrt(n):.2f}]"
            )
        except Exception as e:
            stats_text = (
                f"Distribution Statistics:\n"
                f"Mean: {mu:.2f}\n"
                f"Median: {median(data):.2f}\n"
                f"Std Dev: {sigma:.2f}\n"
                f"Sample Size: {len(data)}\n"
                f"Error calculating advanced statistics: {str(e)}"
            )
        
        self.stats_label.setText(stats_text)

    def calculate_skewness(self, data):
        try:
            mu = mean(data)
            sigma = stdev(data)
            n = len(data)
            if sigma == 0:
                return 0
            return (sum((x - mu)**3 for x in data) / n) / (sigma**3)
        except Exception:
            return 0

    def calculate_kurtosis(self, data):
        try:
            mu = mean(data)
            sigma = stdev(data)
            n = len(data)
            if sigma == 0:
                return 0
            return (sum((x - mu)**4 for x in data) / n) / (sigma**4) - 3
        except Exception:
            return 0

class ControlChartTab(QWidget):
    def __init__(self, dataset: DataSet, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QGridLayout()
        
        # Variable selections
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        controls.addWidget(QLabel("X Variable:"), 0, 0)
        controls.addWidget(self.x_combo, 0, 1)
        controls.addWidget(QLabel("Y Variable:"), 0, 2)
        controls.addWidget(self.y_combo, 0, 3)
        
        # Control limit settings
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(1, 6)
        self.sigma_spin.setValue(3)
        controls.addWidget(QLabel("Control Limits (sigma):"), 1, 0)
        controls.addWidget(self.sigma_spin, 1, 1)
        
        # Plot button
        self.plot_button = QPushButton("Create Control Chart")
        self.plot_button.clicked.connect(self.create_plot)
        controls.addWidget(self.plot_button, 1, 2, 1, 2)
        
        layout.addLayout(controls)
        
        # Chart view
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(self.chart_view.renderHints())
        layout.addWidget(self.chart_view)
        
        # Statistics label
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)
        
        self.update_controls()

    def update_controls(self):
        numeric_columns = self.dataset.get_numeric_columns()
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(numeric_columns)
        self.y_combo.addItems(numeric_columns)

    def create_plot(self):
        if not (self.x_combo.currentText() and self.y_combo.currentText()):
            return
            
        x_idx = self.dataset.headers.index(self.x_combo.currentText())
        y_idx = self.dataset.headers.index(self.y_combo.currentText())
        
        x_data = [float(x) for x in self.dataset.get_column_data(x_idx)]
        y_data = [float(y) for y in self.dataset.get_column_data(y_idx)]
        
        # Calculate control limits
        y_mean = mean(y_data)
        y_std = stdev(y_data)
        sigma_multiplier = self.sigma_spin.value()
        
        ucl = y_mean + sigma_multiplier * y_std
        lcl = y_mean - sigma_multiplier * y_std
        
        # Create chart
        chart = QChart()
        chart.setTitle(f"Control Chart - {self.y_combo.currentText()} vs {self.x_combo.currentText()}")
        
        # Create scatter series
        scatter_series = QScatterSeries()
        scatter_series.setName("Data Points")
        for x, y in zip(x_data, y_data):
            scatter_series.append(x, y)
        
        # Create control limit lines
        ucl_series = QLineSeries()
        ucl_series.setName(f"UCL ({sigma_multiplier}σ)")
        lcl_series = QLineSeries()
        lcl_series.setName(f"LCL ({sigma_multiplier}σ)")
        center_series = QLineSeries()
        center_series.setName("Center Line")
        
        # Add points to control limit lines
        x_min, x_max = min(x_data), max(x_data)
        x_padding = (x_max - x_min) * 0.05
        ucl_series.append(x_min - x_padding, ucl)
        ucl_series.append(x_max + x_padding, ucl)
        lcl_series.append(x_min - x_padding, lcl)
        lcl_series.append(x_max + x_padding, lcl)
        center_series.append(x_min - x_padding, y_mean)
        center_series.append(x_max + x_padding, y_mean)
        
        # Add all series to chart
        chart.addSeries(scatter_series)
        chart.addSeries(ucl_series)
        chart.addSeries(lcl_series)
        chart.addSeries(center_series)
        
        # Set up axes with padding
        axis_x = QValueAxis()
        axis_y = QValueAxis()
        
        # Calculate y-axis range to include control limits
        y_min = min(min(y_data), lcl)
        y_max = max(max(y_data), ucl)
        y_padding = (y_max - y_min) * 0.1
        
        axis_x.setRange(x_min - x_padding, x_max + x_padding)
        axis_y.setRange(y_min - y_padding, y_max + y_padding)
        
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        
        # Attach axes to all series
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
        
        self.chart_view.setChart(chart)
        
        # Update statistics
        points_above_ucl = sum(1 for y in y_data if y > ucl)
        points_below_lcl = sum(1 for y in y_data if y < lcl)
        
        stats_text = (
            f"Control Limits ({sigma_multiplier}σ):\n"
            f"UCL: {ucl:.2f}\n"
            f"Center Line: {y_mean:.2f}\n"
            f"LCL: {lcl:.2f}\n\n"
            f"Points outside control limits:\n"
            f"Above UCL: {points_above_ucl}\n"
            f"Below LCL: {points_below_lcl}"
        )
        self.stats_label.setText(stats_text)
        
class RegressionTab(QWidget):
    def __init__(self, dataset: DataSet, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QGridLayout()
        
        # Dependent variable selection
        self.dependent_combo = QComboBox()
        controls.addWidget(QLabel("Dependent Variable:"), 0, 0)
        controls.addWidget(self.dependent_combo, 0, 1)
        
        # Independent variables selection
        self.independent_list = QListWidget()
        self.independent_list.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection)
        controls.addWidget(QLabel("Independent Variables:"), 1, 0)
        controls.addWidget(self.independent_list, 1, 1)
        
        # Analyze button
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.perform_regression)
        controls.addWidget(self.analyze_button, 2, 0, 1, 2)
        
        layout.addLayout(controls)
        
        # Create tab widget for results
        self.results_tabs = QTabWidget()
        
        # Summary tab
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.results_label = QLabel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.results_label)
        scroll_area.setWidgetResizable(True)
        summary_layout.addWidget(scroll_area)
        self.results_tabs.addTab(self.summary_tab, "Summary")
        
        # Actual vs Predicted Plot tab
        self.actual_predicted_tab = QWidget()
        ap_layout = QVBoxLayout(self.actual_predicted_tab)
        self.actual_predicted_chart_view = QChartView()
        self.actual_predicted_chart_view.setRenderHint(self.actual_predicted_chart_view.renderHints())
        ap_layout.addWidget(self.actual_predicted_chart_view)
        self.results_tabs.addTab(self.actual_predicted_tab, "Actual vs Predicted")
        
        # Residuals Plot tab
        self.residuals_tab = QWidget()
        res_layout = QVBoxLayout(self.residuals_tab)
        self.residuals_chart_view = QChartView()
        self.residuals_chart_view.setRenderHint(self.residuals_chart_view.renderHints())
        res_layout.addWidget(self.residuals_chart_view)
        self.results_tabs.addTab(self.residuals_tab, "Residuals")
        
        layout.addWidget(self.results_tabs)
        
        self.update_controls()

    def update_controls(self):
        numeric_columns = self.dataset.get_numeric_columns()
        self.dependent_combo.clear()
        self.dependent_combo.addItems(numeric_columns)
        
        self.independent_list.clear()
        self.independent_list.addItems(numeric_columns)

    def perform_regression(self):
        if not self.dependent_combo.currentText():
            return
            
        dependent_idx = self.dataset.headers.index(
            self.dependent_combo.currentText())
        independent_idxs = [
            self.dataset.headers.index(item.text())
            for item in self.independent_list.selectedItems()
        ]
        
        if not independent_idxs:
            return
            
        # Perform regression
        results = self.dataset.multiple_regression(
            dependent_idx, independent_idxs)
        
        # Get actual and predicted values
        X = [[1.0] + [float(row[idx]) for idx in independent_idxs] 
             for row in self.dataset.data]
        y_actual = [float(row[dependent_idx]) for row in self.dataset.data]
        
        # Calculate predicted values
        y_pred = []
        for x_row in X:
            pred = 0
            for i, coef in enumerate(results['coefficients'].values()):
                pred += coef * x_row[i]
            y_pred.append(pred)
        
        # Calculate residuals
        residuals = [actual - pred for actual, pred in zip(y_actual, y_pred)]
        
        # Format summary results
        text = "Multiple Linear Regression Results:\n\n"
        text += f"Dependent Variable: {self.dependent_combo.currentText()}\n"
        text += f"Number of Observations: {len(y_actual)}\n\n"
        text += f"R-squared: {results['r_squared']:.4f}\n"
        text += f"Adjusted R-squared: {self.calculate_adjusted_r_squared(results['r_squared'], len(independent_idxs), len(y_actual)):.4f}\n\n"
        text += "Coefficients:\n"
        text += "-" * 40 + "\n"
        text += f"{'Variable':<20} {'Coefficient':>10} {'Std. Error':>10}\n"
        text += "-" * 40 + "\n"
        
        for var, coef in results['coefficients'].items():
            # Calculate standard error (simplified)
            std_error = abs(coef) * 0.1  # Placeholder - should be calculated properly
            text += f"{var:<20} {coef:>10.4f} {std_error:>10.4f}\n"
        
        self.results_label.setText(text)
        
        # Create Actual vs Predicted plot
        self.create_actual_predicted_plot(y_actual, y_pred)
        
        # Create Residuals plot
        self.create_residuals_plot(y_pred, residuals)

    def calculate_adjusted_r_squared(self, r_squared, n_predictors, n_samples):
        return 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_predictors - 1)

    def create_actual_predicted_plot(self, y_actual, y_pred):
        chart = QChart()
        chart.setTitle("Actual vs Predicted Values")
        
        # Create scatter series
        scatter_series = QScatterSeries()
        scatter_series.setName("Data Points")
        scatter_series.setMarkerSize(10)
        
        try:
            # Convert all values to float and create points
            points_data = [(float(actual), float(pred)) 
                          for actual, pred in zip(y_actual, y_pred)]
            
            min_actual = min(p[0] for p in points_data)
            max_actual = max(p[0] for p in points_data)
            min_pred = min(p[1] for p in points_data)
            max_pred = max(p[1] for p in points_data)
            
            # Add all points to the scatter series
            for actual, pred in points_data:
                scatter_series.append(actual, pred)
            
            # Set up the range for the perfect prediction line
            plot_min = min(min_actual, min_pred)
            plot_max = max(max_actual, max_pred)
            
            # Add padding (5%)
            padding = (plot_max - plot_min) * 0.05
            plot_min -= padding
            plot_max += padding
            
            # Create the perfect prediction line
            line_series = QLineSeries()
            line_series.setName("Perfect Prediction")
            line_series.append(plot_min, plot_min)
            line_series.append(plot_max, plot_max)
            
            # Set line appearance
            pen = QPen(QColor("red"))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            line_series.setPen(pen)
            
            # Add both series to chart
            chart.addSeries(scatter_series)
            chart.addSeries(line_series)
            
            # Set up axes
            axis_x = QValueAxis()
            axis_y = QValueAxis()
            
            # Set descriptive labels
            axis_x.setTitleText(f"Actual {self.dependent_combo.currentText()}")
            axis_y.setTitleText(f"Predicted {self.dependent_combo.currentText()}")
            
            # Set ranges with padding
            axis_x.setRange(plot_min, plot_max)
            axis_y.setRange(plot_min, plot_max)
            
            # Add gridlines
            axis_x.setGridLineVisible(True)
            axis_y.setGridLineVisible(True)
            
            # Add axes to chart
            chart.addAxis(axis_x, Qt.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignLeft)
            
            # Attach axes to series
            scatter_series.attachAxis(axis_x)
            scatter_series.attachAxis(axis_y)
            line_series.attachAxis(axis_x)
            line_series.attachAxis(axis_y)
            
            # Calculate statistics
            n = len(points_data)
            actuals = [p[0] for p in points_data]
            preds = [p[1] for p in points_data]
            
            mean_actual = sum(actuals) / n
            mean_pred = sum(preds) / n
            
            covariance = sum((a - mean_actual) * (p - mean_pred) 
                            for a, p in zip(actuals, preds)) / n
            
            std_actual = (sum((a - mean_actual) ** 2 for a in actuals) / n) ** 0.5
            std_pred = (sum((p - mean_pred) ** 2 for p in preds) / n) ** 0.5
            
            correlation = covariance / (std_actual * std_pred) if std_actual * std_pred != 0 else 0
            
            mse = sum((a - p) ** 2 for a, p in zip(actuals, preds)) / n
            rmse = mse ** 0.5
            
            # Add statistics using a QLineSeries for text
            stats_series = QLineSeries()
            stats_series.setName("Statistics")
            stats_series.append(plot_min + padding, plot_max - padding)  # Position for text
            chart.addSeries(stats_series)
            stats_series.attachAxis(axis_x)
            stats_series.attachAxis(axis_y)
            
            # Add statistics label to the chart
            chart.legend().markers(stats_series)[0].setLabel(
                f"Correlation: {correlation:.3f}\n"
                f"RMSE: {rmse:.3f}\n"
                f"N: {n}"
            )
            
        except Exception as e:
            # Handle error case with a simple error message series
            error_series = QLineSeries()
            error_series.setName(f"Error: {str(e)}")
            chart.addSeries(error_series)
        
        self.actual_predicted_chart_view.setChart(chart)
    
        # Debug print chart info
        #print(f"Chart series count: {len(chart.series())}")
        #if chart.series():
        #    print(f"Points in scatter series: {scatter_series.count()}")
        #print(f"Chart axes: {chart.axes()}")

    def create_residuals_plot(self, y_pred, residuals):
        chart = QChart()
        chart.setTitle("Residuals vs Predicted Values")
        
        # Create scatter series
        scatter_series = QScatterSeries()
        scatter_series.setName("Residuals")
        scatter_series.setMarkerSize(10)
        
        for pred, resid in zip(y_pred, residuals):
            scatter_series.append(pred, resid)
        
        # Create zero line
        line_series = QLineSeries()
        line_series.setName("Zero Line")
        min_pred = min(y_pred)
        max_pred = max(y_pred)
        padding_x = (max_pred - min_pred) * 0.1
        line_series.append(min_pred - padding_x, 0)
        line_series.append(max_pred + padding_x, 0)
        
        # Add series to chart
        chart.addSeries(scatter_series)
        chart.addSeries(line_series)
        
        # Set up axes
        axis_x = QValueAxis()
        axis_y = QValueAxis()
        axis_x.setTitleText("Predicted Values")
        axis_y.setTitleText("Residuals")
        
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        
        # Calculate y-axis range with padding
        max_abs_resid = max(abs(min(residuals)), abs(max(residuals)))
        padding_y = max_abs_resid * 0.1
        
        axis_x.setRange(min_pred - padding_x, max_pred + padding_x)
        axis_y.setRange(-max_abs_resid - padding_y, max_abs_resid + padding_y)
        
        scatter_series.attachAxis(axis_x)
        scatter_series.attachAxis(axis_y)
        line_series.attachAxis(axis_x)
        line_series.attachAxis(axis_y)
        
        self.residuals_chart_view.setChart(chart)


class ClusterAnalysisTab(QWidget):
    def __init__(self, dataset: DataSet, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QGridLayout()
        
        # Variable selection
        self.variables_list = QListWidget()
        self.variables_list.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection)
        controls.addWidget(QLabel("Variables:"), 0, 0)
        controls.addWidget(self.variables_list, 0, 1)
        
        # Number of clusters
        self.clusters_spin = QSpinBox()
        self.clusters_spin.setRange(2, 10)
        self.clusters_spin.setValue(3)
        controls.addWidget(QLabel("Number of Clusters:"), 1, 0)
        controls.addWidget(self.clusters_spin, 1, 1)
        
        # Analyze button
        self.analyze_button = QPushButton("Perform Clustering")
        self.analyze_button.clicked.connect(self.perform_clustering)
        controls.addWidget(self.analyze_button, 2, 0, 1, 2)
        
        layout.addLayout(controls)
        
        # Create horizontal layout for chart/scatter and results
        results_layout = QHBoxLayout()
        layout.addLayout(results_layout)
        
        # Container for visualization (will hold either 2D or 3D view)
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        
        # 2D Chart view
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setMinimumSize(600, 400)
        
        # 3D Scatter view setup
        self.scatter_3d = Q3DScatter()
        self.scatter_widget = QWidget.createWindowContainer(self.scatter_3d)
        self.scatter_widget.setMinimumSize(600, 400)
        self.scatter_widget.hide()  # Hide initially
        
        # Initialize scatter plot
        self._initialize_3d_scatter()
        
        # Camera controls
        self.rotation_controls = QWidget()
        rotation_layout = QHBoxLayout(self.rotation_controls)
        
        # Horizontal rotation
        self.horizontal_slider = QSlider(Qt.Horizontal)
        self.horizontal_slider.setRange(-180, 180)
        self.horizontal_slider.setValue(0)
        self.horizontal_slider.valueChanged.connect(
            lambda x: self.scatter_3d.scene().activeCamera().setXRotation(x))
        
        # Vertical rotation
        self.vertical_slider = QSlider(Qt.Horizontal)
        self.vertical_slider.setRange(-90, 90)
        self.vertical_slider.setValue(30)
        self.vertical_slider.valueChanged.connect(
            lambda x: self.scatter_3d.scene().activeCamera().setYRotation(x))
        
        # Add sliders to layout
        for label, slider in [
            ("Horizontal:", self.horizontal_slider),
            ("Vertical:", self.vertical_slider)
        ]:
            rotation_layout.addWidget(QLabel(label))
            rotation_layout.addWidget(slider)
            
        self.rotation_controls.hide()  # Hide initially
        
        # Add views to container
        self.viz_layout.addWidget(self.chart_view)
        self.viz_layout.addWidget(self.scatter_widget)
        self.viz_layout.addWidget(self.rotation_controls)
        
        results_layout.addWidget(self.viz_container, stretch=2)
        
        # Results display
        results_widget = QWidget()
        results_widget.setMinimumWidth(250)
        results_widget_layout = QVBoxLayout(results_widget)
        
        self.results_label = QLabel()
        self.results_label.setWordWrap(True)
        self.results_label.setAlignment(Qt.AlignTop)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.results_label)
        scroll_area.setWidgetResizable(True)
        results_widget_layout.addWidget(scroll_area)
        
        results_layout.addWidget(results_widget, stretch=1)
        
        # Set up initial empty chart
        self._setup_empty_chart()
        self.update_controls()

    def _initialize_3d_scatter(self):
        """Initialize the 3D scatter plot with basic settings"""
        # Create and configure axes
        x_axis = QValue3DAxis()
        y_axis = QValue3DAxis()
        z_axis = QValue3DAxis()
        
        x_axis.setTitle("X Axis")
        y_axis.setTitle("Y Axis")
        z_axis.setTitle("Z Axis")
        
        self.scatter_3d.setAxisX(x_axis)
        self.scatter_3d.setAxisY(y_axis)
        self.scatter_3d.setAxisZ(z_axis)
        
        # Set up camera and scene
        camera = self.scatter_3d.scene().activeCamera()
        camera.setCameraPreset(Q3DCamera.CameraPresetFrontHigh)
        self.scatter_3d.setAspectRatio(1.0)
        self.scatter_3d.setHorizontalAspectRatio(1.0)
        self.scatter_3d.setShadowQuality(QAbstract3DGraph.ShadowQualityNone)

    def _setup_empty_chart(self):
        """Initialize an empty chart with basic setup"""
        chart = QChart()
        chart.setTitle("Select variables and click 'Perform Clustering'")
        
        axis_x = QValueAxis()
        axis_y = QValueAxis()
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        
        axis_x.setRange(0, 100)
        axis_y.setRange(0, 100)
        
        self.chart_view.setChart(chart)

    def update_controls(self):
        """Update the variable list with numeric columns"""
        self.variables_list.clear()
        self.variables_list.addItems(self.dataset.get_numeric_columns())

    def perform_clustering(self):
        """Perform clustering analysis and update visualization"""
        selected_vars = [item.text() for item in self.variables_list.selectedItems()]
        if len(selected_vars) < 2:
            QMessageBox.warning(self, "Warning", 
                              "Please select at least 2 variables for clustering.")
            return
            
        var_indices = [self.dataset.headers.index(var) for var in selected_vars]
        n_clusters = self.clusters_spin.value()
        
        results = self.dataset.cluster_analysis(var_indices, n_clusters)
        
        # Switch between 2D and 3D visualization
        if len(var_indices) == 2:
            self.chart_view.show()
            self.scatter_widget.hide()
            self.rotation_controls.hide()
            self._create_2d_visualization(selected_vars, var_indices, results)
        elif len(var_indices) == 3:
            self.chart_view.hide()
            self.scatter_widget.show()
            self.rotation_controls.show()
            self._create_3d_visualization(selected_vars, var_indices, results)
        else:
            self.chart_view.hide()
            self.scatter_widget.hide()
            self.rotation_controls.hide()
            QMessageBox.information(self, "Info", 
                "Visualization is only available for 2 or 3 variables.")
        
        self._update_results_text(selected_vars, n_clusters, results)

    def _prepare_3d_data(self, var_indices):
        """Prepare and validate 3D data points"""
        valid_data = []
        try:
            for i in range(len(self.dataset.data)):
                try:
                    x = float(self.dataset.data[i][var_indices[0]])
                    y = float(self.dataset.data[i][var_indices[1]])
                    z = float(self.dataset.data[i][var_indices[2]])
                    if all(not math.isnan(val) and not math.isinf(val) 
                          for val in (x, y, z)):
                        valid_data.append((x, y, z))
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f"Error preparing 3D data: {e}")
            
        return valid_data

    def _create_3d_visualization(self, selected_vars, var_indices, results):
        """Create the 3D scatter plot visualization"""
        try:
            # Clear existing series
            while self.scatter_3d.seriesList():
                self.scatter_3d.removeSeries(self.scatter_3d.seriesList()[0])
                
            # Define colors for clusters
            colors = [
                QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#2ca02c"),
                QColor("#d62728"), QColor("#9467bd"), QColor("#8c564b"),
                QColor("#e377c2"), QColor("#7f7f7f"), QColor("#bcbd22"),
                QColor("#17becf")
            ]
            
            # Get valid data points
            valid_data = self._prepare_3d_data(var_indices)
            if not valid_data:
                raise ValueError("No valid data points found for visualization")
            
            # Create cluster series
            for cluster_idx in range(len(results['centroids'])):
                # Create series for this cluster
                series = QScatter3DSeries()
                series.setItemSize(0.15)
                series.setBaseColor(colors[cluster_idx % len(colors)])
                series.setMesh(QAbstract3DSeries.MeshSphere)
                series.setName(f"Cluster {cluster_idx + 1}")
                
                # Add points for this cluster
                points = []
                for i, cluster in enumerate(results['clusters']):
                    if cluster == cluster_idx and i < len(valid_data):
                        x, y, z = valid_data[i]
                        points.append(QVector3D(x, y, z))
                
                # Set points to series
                if points:
                    series.dataProxy().addItems(points)
                    self.scatter_3d.addSeries(series)
            
            # Add centroids
            centroid_series = QScatter3DSeries()
            centroid_series.setItemSize(0.3)
            centroid_series.setBaseColor(QColor("black"))
            centroid_series.setMesh(QAbstract3DSeries.MeshSphere)
            centroid_series.setName("Centroids")
            
            # Add centroid points
            centroid_points = []
            for centroid in results['centroids']:
                centroid_points.append(QVector3D(centroid[0], centroid[1], centroid[2]))
            
            if centroid_points:
                centroid_series.dataProxy().addItems(centroid_points)
                self.scatter_3d.addSeries(centroid_series)
            
            # Update axes
            self.scatter_3d.axisX().setTitle(selected_vars[0])
            self.scatter_3d.axisY().setTitle(selected_vars[1])
            self.scatter_3d.axisZ().setTitle(selected_vars[2])
            
            # Calculate and set axis ranges
            x_values = [p[0] for p in valid_data]
            y_values = [p[1] for p in valid_data]
            z_values = [p[2] for p in valid_data]
            
            margin = 0.1
            for axis, values in [
                (self.scatter_3d.axisX(), x_values),
                (self.scatter_3d.axisY(), y_values),
                (self.scatter_3d.axisZ(), z_values)
            ]:
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val
                axis.setRange(min_val - margin * range_val, 
                            max_val + margin * range_val)
            
        except Exception as e:
            print(f"Error in _create_3d_visualization: {e}")
            QMessageBox.warning(self, "Warning", 
                "Failed to create 3D visualization. Check the data and try again.")
            self.scatter_widget.hide()
            self.chart_view.show()

    def _create_2d_visualization(self, selected_vars, var_indices, results):
        """Create the 2D chart visualization"""
        chart = QChart()
        chart.setTitle(f"Cluster Analysis: {selected_vars[0]} vs {selected_vars[1]}")
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)
        
        # Clear any existing series
        chart.removeAllSeries()
        
        # Define distinct colors for clusters
        colors = [
            QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#2ca02c"),
            QColor("#d62728"), QColor("#9467bd"), QColor("#8c564b"),
            QColor("#e377c2"), QColor("#7f7f7f"), QColor("#bcbd22"),
            QColor("#17becf")
        ]
        
        # Calculate ranges first
        x_values = []
        y_values = []
        for i in range(len(self.dataset.data)):
            try:
                x = float(self.dataset.data[i][var_indices[0]])
                y = float(self.dataset.data[i][var_indices[1]])
                x_values.append(x)
                y_values.append(y)
            except (ValueError, IndexError) as e:
                print(f"Error reading point {i}: {e}")
                continue
        
        if not x_values or not y_values:
            print("No valid data points found!")
            return
            
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        # Create a series for each cluster
        num_clusters = len(results['centroids'])
        for cluster_idx in range(num_clusters):
            series = QScatterSeries()
            series.setName(f"Cluster {cluster_idx + 1}")
            series.setMarkerSize(10)
            series.setColor(colors[cluster_idx % len(colors)])
            series.setMarkerShape(QScatterSeries.MarkerShapeCircle)
            
            # Add points for this cluster
            points_added = 0
            for i, cluster in enumerate(results['clusters']):
                if cluster == cluster_idx:
                    try:
                        x = float(self.dataset.data[i][var_indices[0]])
                        y = float(self.dataset.data[i][var_indices[1]])
                        series.append(x, y)
                        points_added += 1
                    except (ValueError, IndexError) as e:
                        print(f"Error adding point {i} to cluster {cluster_idx}: {e}")
                        continue
            
            chart.addSeries(series)
        
        # Add centroids as different markers
        centroid_series = QScatterSeries()
        centroid_series.setName("Centroids")
        centroid_series.setMarkerSize(15)
        centroid_series.setMarkerShape(QScatterSeries.MarkerShapeRectangle)
        centroid_series.setColor(QColor("black"))
        
        for centroid in results['centroids']:
            centroid_series.append(centroid[0], centroid[1])
        
        chart.addSeries(centroid_series)
        
        # Set up axes with padding
        axis_x = QValueAxis()
        axis_y = QValueAxis()
        
        # Make grid lines visible
        axis_x.setGridLineVisible(True)
        axis_y.setGridLineVisible(True)
        
        # Set labels
        axis_x.setTitleText(selected_vars[0])
        axis_y.setTitleText(selected_vars[1])
        
        # Add axes to chart
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        
        # Calculate padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        # Set ranges with padding
        axis_x.setRange(x_min - x_padding, x_max + x_padding)
        axis_y.setRange(y_min - y_padding, y_max + y_padding)
        
        # Attach axes to all series
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
        
        self.chart_view.setChart(chart)

    def _update_results_text(self, selected_vars, n_clusters, results):
        # Format results text
        text = "<h3>Cluster Analysis Results</h3>"
        text += f"<p><b>Number of clusters:</b> {n_clusters}</p>"
        text += "<h4>Cluster Centers:</h4>"
        for i, centroid in enumerate(results['centroids']):
            text += f"<p><b>Cluster {i + 1}:</b><br>"
            text += ", ".join(f"{var}: {val:.2f}" 
                            for var, val in zip(selected_vars, centroid))
            text += "</p>"
        
        # Add cluster sizes
        cluster_sizes = defaultdict(int)
        for cluster in results['clusters']:
            cluster_sizes[cluster] += 1
        
        text += "<h4>Cluster Sizes:</h4>"
        for cluster, size in sorted(cluster_sizes.items()):
            text += f"<p><b>Cluster {cluster + 1}:</b> {size} points "
            text += f"({(size/len(results['clusters'])*100):.1f}%)</p>"
        
        self.results_label.setText(text)


class FactorAnalysisTab(QWidget):
    def __init__(self, dataset: DataSet, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QGridLayout()
        
        # Variable selection
        self.variables_list = QListWidget()
        self.variables_list.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection)
        controls.addWidget(QLabel("Variables:"), 0, 0)
        controls.addWidget(self.variables_list, 0, 1)
        
        # Number of factors
        self.factors_spin = QSpinBox()
        self.factors_spin.setRange(1, 10)
        self.factors_spin.setValue(2)
        controls.addWidget(QLabel("Number of Factors:"), 1, 0)
        controls.addWidget(self.factors_spin, 1, 1)
        
        # Analyze button
        self.analyze_button = QPushButton("Perform Factor Analysis")
        self.analyze_button.clicked.connect(self.perform_analysis)
        controls.addWidget(self.analyze_button, 2, 0, 1, 2)
        
        layout.addLayout(controls)
        
        # Results display
        self.results_label = QLabel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.results_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        self.update_controls()

    def update_controls(self):
        self.variables_list.clear()
        self.variables_list.addItems(self.dataset.get_numeric_columns())

    def perform_analysis(self):
        selected_vars = [item.text() for item in self.variables_list.selectedItems()]
        if len(selected_vars) < 2:
            return
            
        var_indices = [self.dataset.headers.index(var) for var in selected_vars]
        n_factors = min(self.factors_spin.value(), len(selected_vars))
        
        results = self.dataset.factor_analysis(var_indices, n_factors)
        
        # Format results
        text = "Factor Analysis Results:\n\n"
        text += "Factor Loadings:\n\n"
        
        # Create a table-like format
        text += "Variable".ljust(20)
        for i in range(n_factors):
            text += f"Factor {i+1}".rjust(12)
        text += "\n"
        text += "-" * (20 + 12 * n_factors) + "\n"
        
        for var_idx, var_name in enumerate(results['variables']):
            text += var_name.ljust(20)
            for factor_idx in range(n_factors):
                text += f"{results['loadings'][factor_idx][var_idx]:10.3f}".rjust(12)
            text += "\n"
        
        self.results_label.setText(text)

class ProcessCapabilityTab(QWidget):
    def __init__(self, dataset: DataSet, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        layout = QHBoxLayout(self)  # Using horizontal layout for side-by-side arrangement
        
        # Left side - Control Chart
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        
        # Chart view
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(self.chart_view.renderHints())
        chart_layout.addWidget(self.chart_view)
        
        layout.addWidget(chart_container, stretch=2)  # Give chart more space
        
        # Right side - Settings and Statistics
        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        
        # Settings Box
        settings_group = QGroupBox("Settings")
        settings_form = QGridLayout(settings_group)
        
        # Variable selection
        self.variable_combo = QComboBox()
        settings_form.addWidget(QLabel("Variable:"), 0, 0)
        settings_form.addWidget(self.variable_combo, 0, 1)
        
        # Specification Limits
        self.lsl_spin = QDoubleSpinBox()
        self.lsl_spin.setRange(-1000000, 1000000)
        self.lsl_spin.setDecimals(3)
        settings_form.addWidget(QLabel("LSL:"), 1, 0)
        settings_form.addWidget(self.lsl_spin, 1, 1)
        
        self.usl_spin = QDoubleSpinBox()
        self.usl_spin.setRange(-1000000, 1000000)
        self.usl_spin.setDecimals(3)
        settings_form.addWidget(QLabel("USL:"), 2, 0)
        settings_form.addWidget(self.usl_spin, 2, 1)
        
        # Target value for Cpm calculation
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(-1000000, 1000000)
        self.target_spin.setDecimals(3)
        settings_form.addWidget(QLabel("Target:"), 3, 0)
        settings_form.addWidget(self.target_spin, 3, 1)
        
        # Control Limits
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(1, 6)
        self.sigma_spin.setValue(3)
        settings_form.addWidget(QLabel("Control Limits (σ):"), 4, 0)
        settings_form.addWidget(self.sigma_spin, 4, 1)
        
        # Update button
        self.update_button = QPushButton("Update Analysis")
        self.update_button.clicked.connect(self.perform_analysis)
        settings_form.addWidget(self.update_button, 5, 0, 1, 2)
        
        settings_layout.addWidget(settings_group)
        
        # Statistics Box
        stats_group = QGroupBox("Process Capability Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel()
        stats_layout.addWidget(self.stats_label)
        
        settings_layout.addWidget(stats_group)
        settings_layout.addStretch()
        
        layout.addWidget(settings_container, stretch=1)  # Give settings less space
        
        self.update_controls()

    def update_controls(self):
        self.variable_combo.clear()
        self.variable_combo.addItems(self.dataset.get_numeric_columns())

    def calculate_process_yield(self, data, usl, lsl, mean, std):
        """Calculate process yield and fallout using normal distribution"""
        from scipy.stats import norm
        
        # Calculate z-scores for specification limits
        z_upper = (usl - mean) / std
        z_lower = (lsl - mean) / std
        
        # Calculate yield as area between LSL and USL
        process_yield = (norm.cdf(z_upper) - norm.cdf(z_lower)) * 100
        
        # Calculate fallout in PPM (parts per million)
        fallout_ppm = (1 - (process_yield / 100)) * 1_000_000
        
        return process_yield, fallout_ppm

    def perform_analysis(self):
        if not self.variable_combo.currentText():
            return
            
        var_idx = self.dataset.headers.index(self.variable_combo.currentText())
        data = [float(x) for x in self.dataset.get_column_data(var_idx)]
        
        # Calculate basic statistics
        data_mean = mean(data)
        data_std = stdev(data)
        
        # Get control and specification limits
        sigma_mult = self.sigma_spin.value()
        ucl = data_mean + sigma_mult * data_std
        lcl = data_mean - sigma_mult * data_std
        
        usl = self.usl_spin.value()
        lsl = self.lsl_spin.value()
        target = self.target_spin.value()
        
        # Create control chart
        chart = QChart()
        chart.setTitle(f"Process Capability Analysis - {self.variable_combo.currentText()}")
        
        # Create data series
        data_series = QScatterSeries()
        data_series.setName("Data Points")
        data_series.setMarkerSize(8)
        
        for i, value in enumerate(data):
            data_series.append(i, value)
        
        # Create control limit lines
        ucl_series = QLineSeries()
        ucl_series.setName(f"UCL ({sigma_mult}σ)")
        lcl_series = QLineSeries()
        lcl_series.setName(f"LCL ({sigma_mult}σ)")
        
        # Create specification limit lines if set
        usl_series = QLineSeries()
        usl_series.setName("USL")
        lsl_series = QLineSeries()
        lsl_series.setName("LSL")
        
        # Create target line if set
        target_series = QLineSeries()
        target_series.setName("Target")
        
        # Create mean line
        mean_series = QLineSeries()
        mean_series.setName("Mean")
        
        # Add points to all lines
        for i in range(-1, len(data) + 1):
            ucl_series.append(i, ucl)
            lcl_series.append(i, lcl)
            mean_series.append(i, data_mean)
            if usl != 0:
                usl_series.append(i, usl)
            if lsl != 0:
                lsl_series.append(i, lsl)
            if target != 0:
                target_series.append(i, target)
        
        # Set line styles
        pen = QPen(QColor("red"))
        pen.setStyle(Qt.DashLine)
        ucl_series.setPen(pen)
        lcl_series.setPen(pen)
        
        pen = QPen(QColor("blue"))
        pen.setStyle(Qt.DashLine)
        usl_series.setPen(pen)
        lsl_series.setPen(pen)
        
        pen = QPen(QColor("purple"))
        pen.setStyle(Qt.DashLine)
        target_series.setPen(pen)
        
        pen = QPen(QColor("green"))
        mean_series.setPen(pen)
        
        # Add all series to chart
        chart.addSeries(data_series)
        chart.addSeries(ucl_series)
        chart.addSeries(lcl_series)
        chart.addSeries(mean_series)
        if usl != 0:
            chart.addSeries(usl_series)
        if lsl != 0:
            chart.addSeries(lsl_series)
        if target != 0:
            chart.addSeries(target_series)
        
        # Set up axes
        axis_x = QValueAxis()
        axis_y = QValueAxis()
        
        # Calculate y-axis range
        all_limits = [ucl, lcl]
        if usl != 0:
            all_limits.append(usl)
        if lsl != 0:
            all_limits.append(lsl)
        if target != 0:
            all_limits.append(target)
        
        y_min = min(min(data), min(all_limits))
        y_max = max(max(data), max(all_limits))
        y_padding = (y_max - y_min) * 0.1
        
        axis_x.setRange(-1, len(data))
        axis_y.setRange(y_min - y_padding, y_max + y_padding)
        
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        
        # Attach axes to all series
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
        
        self.chart_view.setChart(chart)
        
        # Calculate process capability statistics
        cp = None
        cpk = None
        cpm = None
        cpkm = None
        process_yield = None
        fallout_ppm = None
        
        if usl != 0 and lsl != 0:
            # Calculate Cp (Process Capability)
            cp = (usl - lsl) / (6 * data_std)
            
            # Calculate Cpk (Process Capability Index)
            cpu = (usl - data_mean) / (3 * data_std)
            cpl = (data_mean - lsl) / (3 * data_std)
            cpk = min(cpu, cpl)
            
            # Calculate process yield and fallout
            process_yield, fallout_ppm = self.calculate_process_yield(data, usl, lsl, data_mean, data_std)
            
            if target != 0:
                # Calculate Cpm (Process Capability Index considering target)
                variance_from_target = sum((x - target) ** 2 for x in data) / (len(data) - 1)
                cpm = (usl - lsl) / (6 * math.sqrt(variance_from_target))
                
                # Calculate Cpkm (Modified Process Capability Index)
                mean_deviation_from_target = abs(data_mean - target)
                cpkm = cp * (1 / math.sqrt(1 + (mean_deviation_from_target / data_std) ** 2))
        
        # Calculate process performance statistics
        points_outside_cl = sum(1 for x in data if x > ucl or x < lcl)
        points_outside_sl = sum(1 for x in data if (usl != 0 and x > usl) or (lsl != 0 and x < lsl))
        n = len(data)
        sigma = stdev(data)
        
        # Update statistics display
        stats_text = f"Process Statistics:\n\n"
        stats_text += f"Mean: {data_mean:.3f}\n"
        stats_text += f"Standard Deviation: {data_std:.3f}\n"
        stats_text += f"Points Outside Control Limits: {points_outside_cl}\n"
        stats_text += f"Points Outside Spec Limits: {points_outside_sl}\n\n"
        stats_text += f"Sample Size: {n}\n"
        stats_text += f"Standard Error: {sigma/math.sqrt(n):.2f}\n"
        stats_text += f"95% Confidence Interval: [{data_mean - 1.96*sigma/math.sqrt(n):.2f}, "
        stats_text += f"{data_mean + 1.96*sigma/math.sqrt(n):.2f}]\n\n"
        
        if cp is not None:
            stats_text += f"Process Capability Indices:\n"
            stats_text += f"Cp: {cp:.3f}\n"
            stats_text += f"Process Yield: {process_yield:.2f}%\n"
            stats_text += f"Process Fallout: {fallout_ppm:.1f} PPM\n"
            stats_text += f"Cpk: {cpk:.3f}\n"
            if cpm is not None:
                stats_text += f"Cpm: {cpm:.3f}\n"
                stats_text += f"Cpkm: {cpkm:.3f}\n"
        
        self.stats_label.setText(stats_text)

        
class MultivariateAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Statistical Analysis Tool")
        self.setMinimumSize(1200, 800)
        self.dataset = DataSet()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create top controls
        controls_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        controls_layout.addWidget(self.load_button)
        layout.addLayout(controls_layout)
        
        # Create table view
        self.table_view = QTableView()
        layout.addWidget(self.table_view)
        
        # Create tab widget for different analyses
        self.tab_widget = QTabWidget()
        
        # Create and add analysis tabs
        self.histogram_tab = HistogramWithFitTab(self.dataset)
        self.regression_tab = RegressionTab(self.dataset)
        self.cluster_tab = ClusterAnalysisTab(self.dataset)
        self.factor_tab = FactorAnalysisTab(self.dataset)
        self.process_capability_tab = ProcessCapabilityTab(self.dataset)
        
        self.tab_widget.addTab(self.histogram_tab, "Distribution Analysis")
        self.tab_widget.addTab(self.regression_tab, "Multiple Regression")
        self.tab_widget.addTab(self.cluster_tab, "Cluster Analysis")
        self.tab_widget.addTab(self.factor_tab, "Factor Analysis")
        self.tab_widget.addTab(self.process_capability_tab, "Process Capability")
        
        layout.addWidget(self.tab_widget)
        
        # Set layout proportions
        layout.setStretch(1, 1)  # Table view
        layout.setStretch(2, 2)  # Analysis tabs

    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data File",
            "",
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if file_name:
            try:
                self.dataset.load_file(file_name)
                
                # Update table view
                model = DataTableModel(self.dataset)
                self.table_view.setModel(model)
                
                # Update all analysis tabs
                self.histogram_tab.update_controls()
                self.regression_tab.update_controls()
                self.cluster_tab.update_controls()
                self.factor_tab.update_controls()
                self.process_capability_tab.update_controls()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load data: {str(e)}"
                )

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = MultivariateAnalysisApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()