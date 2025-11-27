import panel as pn
import plotly.graph_objects as go
import h5py
import numpy as np
import os

pn.extension('plotly')

DATA_DIR = './data'

def get_h5_files():
    return [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]

class Visualizer:
    def __init__(self):
        self.h5_files = get_h5_files()
        self.file_selector = pn.widgets.Select(name='Select H5 File', options=self.h5_files)
        self.sample_selector = pn.widgets.Select(name='Select Sample')
        self.plot_pane = pn.pane.Plotly(go.Figure())

        self.file_selector.param.watch(self._update_samples, 'value')
        self.sample_selector.param.watch(self._update_plot, 'value')

        if self.h5_files:
            self.file_selector.value = self.h5_files[0]

    def _update_samples(self, event):
        file_path = os.path.join(DATA_DIR, self.file_selector.value)
        with h5py.File(file_path, 'r') as f:
            samples = list(f.keys())
            self.sample_selector.options = samples

    def _update_plot(self, event):
        file_path = os.path.join(DATA_DIR, self.file_selector.value)
        sample_name = self.sample_selector.value

        if not sample_name:
            self.plot_pane.object = go.Figure()
            return

        with h5py.File(file_path, 'r') as f:
            if 'Groud_Truth' in self.file_selector.value:
                data = f[sample_name]
                x, y, z = data['x'][:], data['y'][:], data['z'][:]
                fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines')])
                fig.update_layout(title=f'Ground Truth: {sample_name}')
            else:
                data = f[sample_name]
                accel_x, accel_y, accel_z = data['Ax'][:], data['Ay'][:], data['Az'][:]
                time = np.arange(len(accel_x))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time, y=accel_x, name='Ax'))
                fig.add_trace(go.Scatter(x=time, y=accel_y, name='Ay'))
                fig.add_trace(go.Scatter(x=time, y=accel_z, name='Az'))
                fig.update_layout(title=f'Sensor Data: {sample_name}')

            self.plot_pane.object = fig

    def view(self):
        return pn.Column(
            self.file_selector,
            self.sample_selector,
            self.plot_pane
        )

if __name__ == "__main__":
    visualizer = Visualizer()
    dashboard = visualizer.view()
    dashboard.servable(title="Inku Data Visualizer")
    # To run, use `panel serve visualize.py --show`