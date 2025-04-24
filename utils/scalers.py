from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

class FeatureScaler:
    def __init__(self, method='standard'):
        """
        Initialize the scaler with either 'standard' or 'minmax' scaling methods.

        Parameters:
        - method (str): Type of scaling method ('standard' or 'minmax')
        """
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

    def fit(self, all_data):
        """
        Compute the scaling parameters based on the method selected during initialization
        and fit the scaler to all node features.

        Parameters:
        - all_data (list of np.ndarray): Each array is a node feature matrix of shape (num_atoms, num_features)
        """
        all_features = np.vstack(all_data)  # Combine all feature matrices vertically
        self.scaler.fit(all_features)

    def transform(self, data):
        """
        Apply the scaling transformation to the data.

        Parameters:
        - data (list of np.ndarray): Data to be scaled

        Returns:
        - list of np.ndarray: Scaled data
        """
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """
        Reverse the scaling transformation (useful for interpreting model outputs).

        Parameters:
        - data (list of np.ndarray): Scaled data to be reversed

        Returns:
        - list of np.ndarray: Original scale data
        """
        return self.scaler.inverse_transform(data)
