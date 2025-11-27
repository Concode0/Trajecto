import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py
import numpy as np
import os

# Initialize Panel extension for Plotly
pn.extension('plotly')

# Define the directory where data files are stored
DATA_DIR = './data'

def get_h5_files():
    """Returns a list of .h5 files in the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return []
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]

class Visualizer:
    """A Panel-based visualizer for HDF5 file data."""
    def __init__(self):
        self.h5_files = get_h5_files()

        if not self.h5_files:
            self.view = pn.Column("No HDF5 files found in the 'data' directory.")
            return

        # Create a dropdown to select an H5 file
        self.file_selector = pn.widgets.Select(name='Select H5 File', options=self.h5_files)

        # Create a pane to display the Plotly figure
        self.plot_pane = pn.pane.Plotly(go.Figure(), sizing_mode='stretch_width')

        # Watch for changes in the file selector's value and update the plot
        self.file_selector.param.watch(self._update_plot, 'value')

        # If there are files, select the first one and trigger the plot update
        self.file_selector.value = self.h5_files[0]
        self._update_plot(None)  # Initial plot update

        self.view = pn.Column(
            self.file_selector,
            self.plot_pane,
            sizing_mode='stretch_width'
        )

    def _update_plot(self, event):
        """Callback to update the plot when the selected file changes."""
        file_path = os.path.join(DATA_DIR, self.file_selector.value)

        if not self.file_selector.value:
            self.plot_pane.object = go.Figure()
            return

        with h5py.File(file_path, 'r') as f:
            samples = list(f.keys())
            if not samples:
                self.plot_pane.object = go.Figure(layout_title_text="No samples in this file.")
                return

            # Create subplots, one for each sample
            fig = make_subplots(rows=len(samples), cols=1, subplot_titles=samples)

            for i, sample_name in enumerate(samples):
                data = f[sample_name]

                # Check for ground truth or sensor data and create an appropriate plot
                if 'x' in data and 'y' in data and 'z' in data:
                    x, y, z = data['x'][:], data['y'][:], data['z'][:]
                    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Trajectory'), row=i+1, col=1)
                elif 'Ax' in data and 'Ay' in data and 'Az' in data:
                    accel_x, accel_y, accel_z = data['Ax'][:], data['Ay'][:], data['Az'][:]
                    time = np.arange(len(accel_x))
                    fig.add_trace(go.Scatter(x=time, y=accel_x, name='Ax'), row=i+1, col=1)
                    fig.add_trace(go.Scatter(x=time, y=accel_y, name='Ay'), row=i+1, col=1)
                    fig.add_trace(go.Scatter(x=time, y=accel_z, name='Az'), row=i+1, col=1)

            fig.update_layout(height=300 * len(samples), title_text=f"Samples in {self.file_selector.value}")
            self.plot_pane.object = fig

def main():
    """Main function to create and serve the dashboard."""
    visualizer = Visualizer()
    dashboard = visualizer.view
    dashboard.servable(title="Inku Data Visualizer")

if __name__ == "__main__":
    main()
    # To run, use `panel serve visualize.py --show`