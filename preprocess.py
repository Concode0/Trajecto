import numpy as np
import pandas as pd
import h5py
import os

class DataPreprocessor:
    def __init__(self, output_dir='./preprocessed_data'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _interpolate_missing_data(self, df):
        """
        Interpolates missing values in the DataFrame using linear interpolation.
        Assumes data is time-series and columns are numerical.
        """
        return df.interpolate(method='linear', limit_direction='both')

    def _synchronize_data(self, df_list, time_column='timestamp'):
        """
        Synchronizes multiple DataFrames based on a common time column.
        Currently performs a simple merge based on the nearest timestamp.
        Further refinement might be needed based on specific synchronization requirements.
        """
        if not df_list:
            return pd.DataFrame()

        # For simplicity, let's assume we are synchronizing dataframes that should be merged
        # A more complex synchronization might involve resampling or more advanced alignment
        # For now, let's just return the first dataframe, assuming it's the primary
        # and other data will be aligned during feature engineering or specific processing
        return df_list[0] # Placeholder: Needs actual implementation based on data sources

    def preprocess_file(self, h5_file_path):
        """
        Preprocesses a single HDF5 file.
        Reads data, interpolates missing values, and saves the processed data.
        """
        processed_data = {}
        file_name = os.path.basename(h5_file_path)
        output_file_path = os.path.join(self.output_dir, f"preprocessed_{file_name}")

        with h5py.File(h5_file_path, 'r') as f_in:
            with h5py.File(output_file_path, 'w') as f_out:
                for sample_name in f_in.keys():
                    sample_group = f_in[sample_name]
                    
                    # Convert to pandas DataFrame for easier interpolation
                    # Assuming all datasets in the sample_group are numerical and can form a DataFrame
                    data_dict = {key: sample_group[key][:] for key in sample_group.keys()}
                    df = pd.DataFrame(data_dict)

                    # Interpolate missing data
                    df_interpolated = self._interpolate_missing_data(df)

                    # Save processed data back to HDF5
                    processed_sample_group = f_out.create_group(sample_name)
                    for col in df_interpolated.columns:
                        processed_sample_group.create_dataset(col, data=df_interpolated[col].values)
            
            # Here, if there were multiple dataframes to synchronize within one HDF5 file,
            # _synchronize_data would be called before saving.
            # For now, it's applied per sample.

        print(f"Processed and saved: {output_file_path}")
        return output_file_path

    def preprocess_directory(self, input_dir='./data'):
        """
        Preprocesses all HDF5 files in a given directory.
        """
        h5_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.h5')]
        
        for h5_file_path in h5_files:
            self.preprocess_file(h5_file_path)

if __name__ == '__main__':
    # Example Usage:
    # 1. Create dummy data for testing (if not already present)
    dummy_data_dir = './data'
    os.makedirs(dummy_data_dir, exist_ok=True)
    
    # Create a dummy HDF5 file with some missing data
    dummy_h5_path = os.path.join(dummy_data_dir, 'dummy_sensor_data.h5')
    with h5py.File(dummy_h5_path, 'w') as f:
        sample1 = f.create_group('sample1')
        sample1.create_dataset('Ax', data=np.array([1.0, 2.0, np.nan, 4.0, 5.0]))
        sample1.create_dataset('Ay', data=np.array([5.0, np.nan, 3.0, 2.0, 1.0]))
        sample1.create_dataset('Az', data=np.array([0.1, 0.2, 0.3, np.nan, 0.5]))

        sample2 = f.create_group('sample2')
        sample2.create_dataset('Ax', data=np.array([10.0, 11.0, 12.0, np.nan, 14.0]))
        sample2.create_dataset('Ay', data=np.array([np.nan, 16.0, 17.0, 18.0, 19.0]))
        sample2.create_dataset('Az', data=np.array([1.1, 1.2, np.nan, 1.4, 1.5]))

    print(f"Created dummy HDF5 file: {dummy_h5_path}")

    # 2. Instantiate the preprocessor and preprocess the dummy file
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_directory(input_dir=dummy_data_dir)

    # 3. Verify the preprocessed data
    preprocessed_dummy_h5_path = os.path.join(preprocessor.output_dir, 'preprocessed_dummy_sensor_data.h5')
    if os.path.exists(preprocessed_dummy_h5_path):
        print(f"\nVerifying preprocessed data in {preprocessed_dummy_h5_path}:")
        with h5py.File(preprocessed_dummy_h5_path, 'r') as f:
            for sample_name in f.keys():
                print(f"  Sample: {sample_name}")
                sample_group = f[sample_name]
                for key in sample_group.keys():
                    data = sample_group[key][:]
                    print(f"    {key}: {data} (Contains NaN: {np.isnan(data).any()})")
    else:
        print(f"\nPreprocessed file not found: {preprocessed_dummy_h5_path}")
