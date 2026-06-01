# -*- coding: utf-8 -*-
import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from scipy.stats import chi2_contingency

class DataExplorer:
    """
    An integrated suite for data auditing, cleaning, 
    and multi-dimensional visualization optimized for local desktop environments.
    """
    def __init__(self):
        self.dataset = None
        self.num_vars = []
        self.cat_vars = []

    # --- Section 1: Ingestion ---
    def load_local_csv(self):
        """Opens a native desktop file picker to select and load a CSV file."""
        import tkinter as tk
        from tkinter import filedialog
        
        print("Opening file selector window...")
        root = tk.Tk()
        root.withdraw()  # Hide the main background window of tkinter
        root.attributes('-topmost', True)  # Bring the file dialog to the front
        
        file_path = filedialog.askopenfilename(
            title="Select a CSV File for Analysis",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            print("File selection cancelled.")
            return

        # Define strings to be treated as NaN immediately
        null_markers = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', 'nan']
        self.dataset = pd.read_csv(file_path, na_values=null_markers)
        print(f"Data source successfully loaded from: {os.path.basename(file_path)}")
        self._refresh_metadata()

    def _refresh_metadata(self):
        """Internal helper to sync column types and identify numeric/categorical splits."""
        if self.dataset is None: return

        for col in self.dataset.columns:
            # Attempt numeric conversion for object columns
            if self.dataset[col].dtype == 'object':
                attempt = pd.to_numeric(self.dataset[col], errors='coerce')
                if not attempt.isna().all():
                    self.dataset[col] = attempt

        self.num_vars = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_vars = self.dataset.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Section 2: Audit & Cleaning ---
    def audit_structure(self):
        """Prints a high-level overview of the data architecture."""
        if self.dataset is None: 
            print("No active dataset.")
            return
        
        print(f"\nInventory: {self.dataset.shape[0]} rows | {self.dataset.shape[1]} features")
        print(f"Numeric Map: {self.num_vars}")
        print(f"Categorical Map: {self.cat_vars}\n")
        print("--- First 10 Rows Preview ---")
        print(self.dataset.head(10))

    def impute_nulls(self, method='median', val=None):
        """Applies imputation logic across the dataframe based on data types."""
        if self.dataset is None: return

        for col in self.dataset.columns:
            if self.dataset[col].isnull().any():
                if col in self.num_vars:
                    if method == 'mean': self.dataset[col].fillna(self.dataset[col].mean(), inplace=True)
                    elif method == 'median': self.dataset[col].fillna(self.dataset[col].median(), inplace=True)
                    elif method == 'mode': self.dataset[col].fillna(self.dataset[col].mode()[0], inplace=True)
                    elif method == 'fixed': self.dataset[col].fillna(val, inplace=True)
                else:
                    # Categorical logic: default to most frequent or 'Missing'
                    freq = self.dataset[col].mode()
                    fill_token = freq[0] if not freq.empty else "N/A"
                    self.dataset[col].fillna(fill_token if method != 'fixed' else val, inplace=True)
        print(f"Data missingness addressed using '{method}' strategy.")

    def drop_redundancy(self):
        """Identifies and purges duplicate records."""
        if self.dataset is not None:
            count = self.dataset.duplicated().sum()
            self.dataset.drop_duplicates(inplace=True)
            print(f"Redundancy check complete: {count} exact duplicates removed.")

    def manage_outliers(self, targets=None, purge=True):
        """Uses Interquartile Range (IQR) to detect/remove extreme values."""
        if self.dataset is None: return
        scan_list = targets if targets else self.num_vars
        
        indices_to_cull = []
        for col in scan_list:
            q_low, q_high = self.dataset[col].quantile([0.25, 0.75])
            iqr = q_high - q_low
            floor, ceil = q_low - 1.5 * iqr, q_high + 1.5 * iqr
            
            mask = (self.dataset[col] < floor) | (self.dataset[col] > ceil)
            found = self.dataset.index[mask].tolist()
            
            if not purge:
                print(f"Column '{col}': {len(found)} outliers flagged.")
            else:
                indices_to_cull.extend(found)

        if purge and indices_to_cull:
            unique_culls = list(set(indices_to_cull))
            self.dataset.drop(index=unique_culls, inplace=True)
            print(f"Engine purged {len(unique_culls)} total outlier rows across flagged features.")

    # --- Section 3: ML Preprocessing ---
    def scale_features(self, mode='standard'):
        """Applies numeric scaling (standard, minmax, or robust)."""
        if self.dataset is None or not self.num_vars: return pd.DataFrame()
        
        engine = {'minmax': MinMaxScaler(), 'robust': RobustScaler()}.get(mode, StandardScaler())
        processed = engine.fit_transform(self.dataset[self.num_vars])
        return pd.DataFrame(processed, columns=self.num_vars, index=self.dataset.index)

    def encode_features(self, mode='onehot'):
        """Converts text-based features into numeric vectors."""
        if self.dataset is None or not self.cat_vars: return pd.DataFrame()

        if mode == 'onehot':
            xfactor = OneHotEncoder(sparse_output=False, drop='first')
            vect = xfactor.fit_transform(self.dataset[self.cat_vars].astype(str))
            labels = xfactor.get_feature_names_out(self.cat_vars)
            return pd.DataFrame(vect, columns=labels, index=self.dataset.index)
        else:
            xfactor = OrdinalEncoder()
            vect = xfactor.fit_transform(self.dataset[self.cat_vars].astype(str))
            df_v = pd.DataFrame(vect, columns=self.cat_vars, index=self.dataset.index)
            if mode == 'uniform':
                df_v = (df_v - df_v.min()) / (df_v.max() - df_v.min() + 1e-7)
            return df_v

    def prepare_ml_frame(self, s_mode='standard', e_mode='onehot'):
        """Combines scaled and encoded frames into one training-ready set."""
        return pd.concat([self.scale_features(s_mode), self.encode_features(e_mode)], axis=1)

    # --- Section 4: Visual Analytics ---
    def visualize_univariate(self, columns):
        """Generates a detailed 3-way profile for numeric features."""
        for c in columns:
            if c not in self.num_vars: continue
            stage = make_subplots(rows=1, cols=3, subplot_titles=('Distribution', 'Timeline/Order', 'Density'))
            stage.add_trace(go.Violin(x=self.dataset[c], box_visible=True, name=c), row=1, col=1)
            stage.add_trace(go.Scatter(y=self.dataset[c], mode='markers', opacity=0.5), row=1, col=2)
            stage.add_trace(go.Histogram(x=self.dataset[c]), row=1, col=3)
            stage.update_layout(height=400, title=f"Statistical Profile: {c}", showlegend=False)
            stage.show()

    def visualize_bivariate(self, a, b):
        """Intelligently picks a plot based on the pair's data types."""
        if a not in self.dataset.columns or b not in self.dataset.columns: return
        
        is_a_num, is_b_num = a in self.num_vars, b in self.num_vars

        if is_a_num and is_b_num:
            canvas = px.scatter(self.dataset, x=a, y=b, trendline="ols", title=f"Regression: {a} vs {b}")
        elif not is_a_num and not is_b_num:
            canvas = px.density_heatmap(self.dataset, x=a, y=b, text_auto=True, title=f"Intersection: {a} & {b}")
        else:
            cat, num = (a, b) if not is_a_num else (b, a)
            canvas = px.box(self.dataset, x=cat, y=num, points="outliers", title=f"Comparison: {num} by {cat}")
        canvas.show()

    def generate_heatmap(self):
        """Computes a unified correlation matrix for all data types."""
        if self.dataset is None: return
        fields = self.dataset.columns
        size = len(fields)
        grid = pd.DataFrame(np.zeros((size, size)), index=fields, columns=fields)

        for i in fields:
            for j in fields:
                if i == j: grid.loc[i, j] = 1.0
                elif i in self.num_vars and j in self.num_vars:
                    grid.loc[i, j] = self.dataset[i].corr(self.dataset[j])
                elif i in self.cat_vars and j in self.cat_vars:
                    # Categorical vs Categorical (Cramér's V)
                    ctab = pd.crosstab(self.dataset[i], self.dataset[j])
                    if ctab.size > 0:
                        chi2 = chi2_contingency(ctab)[0]
                        obs = ctab.sum().sum()
                        r, k = ctab.shape
                        grid.loc[i, j] = np.sqrt((chi2/obs) / min(k-1, r-1)) if min(k-1, r-1) > 0 else 0
                else:
                    # Mixed Type Proxy
                    try:
                        n_col, c_col = (i, j) if i in self.num_vars else (j, i)
                        grid.loc[i, j] = self.dataset[n_col].corr(self.dataset[c_col].astype('category').cat.codes)
                    except:
                        grid.loc[i, j] = 0.0

        px.imshow(grid, text_auto=".2f", color_continuous_scale='Viridis', title="Cross-Type Correlation Heatmap").show()

class ChartFactory:
    """Utility for building and injecting HTML-based Plotly components."""
    @staticmethod
    def render(component):
        """For local desktop environments, opens up the generated standalone HTML graph in your browser."""
        if component.get("valid"):
            import webbrowser
            # Creates a temp local html file and boots it up in default browser
            temp_file = "temp_plotly_chart.html"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(component["payload"])
            webbrowser.open(os.path.abspath(temp_file))
            print("Chart rendered successfully in your default web browser!")
        else:
            print(f"Visual Error: {component.get('log')}")

    def create_pie(self, df, label_col, val_col, name="Pie Chart"):
        try:
            fig = px.pie(df, names=label_col, values=val_col, hole=0.3, title=name)
            # Tweak to include the standard full HTML wrappers for standalone desktop view context
            return {"valid": True, "payload": fig.to_html(include_plotlyjs='cdn', full_html=True)}
        except Exception as e:
            return {"valid": False, "log": str(e)}

# --- EXECUTION FLOW ---
if __name__ == "__main__":
    core = DataExplorer()
    ui = ChartFactory()

    print("--- 1. LOADING DATA ---")
    # Alternately, you could run core.load_local_csv() to test your desktop UI file pop-up!
    # Here we default to the web URL for test safety:
    core.dataset = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    core._refresh_metadata()

    print("\n--- 2. AUDITING & RESTRUCTURING ---")
    core.audit_structure()

    print("\n--- 3. CLEANING PIPELINE ---")
    core.impute_nulls(method='median')
    core.drop_redundancy()
    core.manage_outliers(targets=['Fare'], purge=False)

    print("\n--- 4. FEATURE ENGINE OUTPUT PREVIEW ---")
    ml_ready_df = core.prepare_ml_frame(s_mode='robust', e_mode='onehot')
    print(ml_ready_df.head(5))

    print("\n--- 5. INTERACTIVE ANALYTICS WINDOWS ---")
    print("Launching browser windows with your figures...")
    core.visualize_univariate(['Age', 'Fare'])
    core.visualize_bivariate('Survived', 'Age')
    core.generate_heatmap()

    print("\n--- 6. CUSTOM MODULAR COMPONENT GENERATION ---")
    pie_res = ui.create_pie(core.dataset, 'Sex', 'PassengerId', 'Gender Distribution')
    ui.render(pie_res)