import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import pmlb  # Fixed typo from 'from pmlb import pmlb' to avoid module namespace conflicts
from pymfe.mfe import MFE
from collections import Counter

class DataHandler:
    """
    Handles data generation (synthetic), loading (PMLB), and meta-feature extraction.
    """
    def __init__(self, synthetic_data: bool, dataset_path: str = None):
        self.synthetic_data = synthetic_data
        self.dataset_path = dataset_path

    def get_datasets(self, n_samples=500, n_classes=3, proximity=0.5, impurity=0.1, 
                     feature_names=None, random_state=42):
        """
        Public interface to get data. 
        Now accepts 'proximity' and 'impurity' instead of raw means/covariances.
        """
        if self.synthetic_data:
            return self._generate_synthetic_2d(
                n_samples=n_samples,
                n_classes=n_classes,
                proximity=proximity,
                impurity=impurity,
                feature_names=feature_names,
                random_state=random_state
            )
        else:
            return self._load_data_from_file(self.dataset_path)

    @staticmethod
    def _generate_synthetic_2d(n_samples=500, n_classes=2, proximity=0.0, impurity=0.0, 
                               feature_names=None, random_state=None):
        """
        Generates synthetic data using sklearn.make_blobs with control over 
        cluster separation (proximity) and label noise (impurity).
        """
        if feature_names is None:
            feature_names = ["x1", "x2"]
        
        # 1. Handle Random State
        rng = np.random.RandomState(random_state)
        
        # 2. Define Proximity Logic
        max_separation_radius = 2.5
        current_radius = max_separation_radius * (1.0 - proximity)
        
        # Generate centers on a circle with the calculated radius
        angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
        centers = np.column_stack([
            current_radius * np.cos(angles),
            current_radius * np.sin(angles)
        ])

        # 3. Generate Blobs (Using sklearn)
        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            n_features=2,
            cluster_std=1.0, 
            shuffle=True,
            random_state=rng
        )

        # 4. Define Impurity Logic (Label Noise)
        if impurity > 0.0:
            n_noise = int(n_samples * impurity)
            noise_indices = rng.choice(n_samples, size=n_noise, replace=False)
            y[noise_indices] = rng.randint(0, n_classes, size=n_noise)

        # 5. formatting
        df = pd.DataFrame(X, columns=feature_names)
        df["label"] = [f"class_{label}" for label in y]
        
        return df

    @staticmethod
    def _load_meta_features(dataset, dataset_path, dataset_name):
        X = dataset.drop(columns=['target'])
        y = dataset['target']
        X_np = X.values
        y_np = y.values
        
        data_type_counts = dict(Counter(X.dtypes.astype(str)))
        numeric_features = X.select_dtypes(include=np.number)
        missing_ratio = dataset.isnull().sum().sum() / (dataset.shape[0] * dataset.shape[1])

        meta_features = {
            'dataset_path': dataset_path,
            'name': dataset_name,
            'size': dataset.shape[0],
            'n_features': X.shape[1],
            'target_classes': y.nunique(),
            'n_numeric_features': numeric_features.shape[1],
            'n_categorical_features': X.select_dtypes(include=['object', 'category']).shape[1],
            'data_types_counts': data_type_counts,
            'missing_values_ratio': missing_ratio,
        }

        mfe = MFE(groups=['general', 'statistical'], summary=["mean", "sd", "min", "max"], random_state=42)
        mfe.fit(X_np, y_np)
        ft_names, ft_values = mfe.extract(suppress_warnings=True)

        for name, value in zip(ft_names, ft_values):
            clean_name = name.replace("-", "_")
            meta_features[clean_name] = value

        return meta_features

    @staticmethod
    def _load_data_from_file(data_path):
        if not os.path.exists('final_combined_meta_features.csv'):
            print("Loading PMLB datasets...")
            DATASET_DIR = './pmlb_datasets'
            os.makedirs(DATASET_DIR, exist_ok=True)
            
            # FIXED: Correct usage of pmlb dataset names attribute
            classification_datasets = pmlb.classification_dataset_names
            print(f"Found {len(classification_datasets)} classification datasets.")

            # Slice the dataset array if you want to test quickly, e.g., classification_datasets[:5]
            dataset_info = []
            for name in classification_datasets:
                dataset_path_local = os.path.join(DATASET_DIR, f'{name}.csv')
                if os.path.exists(dataset_path_local):
                    try:
                        dataset_info.append({'dataset_path': dataset_path_local, 'name': name})
                        print(f"Loaded '{name}' from local file")
                    except Exception as e:
                        print(f"Could not load dataset '{name}' from local file. Reason: {e}.")
                else:
                    try:
                        dataset = pmlb.fetch_data(name, return_X_y=False)
                        dataset.to_csv(dataset_path_local, index=False)
                        dataset_info.append({'dataset_path': dataset_path_local, 'name': name})
                        print(f"Successfully loaded and saved '{name}'.")
                    except Exception as e:
                        print(f"Could not load dataset '{name}'. Reason: {e}")

            real_datasets_batch = []
            saved_batch_filenames = []
            file_counter = 1
            total_datasets = len(dataset_info)

            print(f"Starting to process {total_datasets} datasets in batches of 5...")

            for i, dataset in enumerate(dataset_info, 1):
                try:
                    print(f"Processing dataset {i}/{total_datasets}: {dataset['name']}...")
                    loaded_dataset = pd.read_csv(dataset['dataset_path'])
                    processed_meta_features = DataHandler._load_meta_features(loaded_dataset, dataset['dataset_path'], dataset['name'])
                    real_datasets_batch.append(processed_meta_features)

                    if (i % 5 == 0) or (i == total_datasets):
                        if not real_datasets_batch:
                            continue
                        print(f"--- Saving batch {file_counter} ---")
                        batch_df = pd.DataFrame.from_records(real_datasets_batch)
                        filename = f'meta_features_batch_{file_counter}.csv'
                        batch_df.to_csv(filename, index=False)
                        print(f"Successfully saved {filename}")
                        saved_batch_filenames.append(filename)
                        real_datasets_batch = []
                        file_counter += 1

                except Exception as e:
                    print(f"WARNING: Error processing {dataset['name']}: {e}. Skipping.")
                    continue

            print("\n--- Combining all saved batch files... ---")
            if not saved_batch_filenames:
                final_combined_dataset = pd.DataFrame()
            else:
                all_batch_dfs = []
                for f in saved_batch_filenames:
                    try:
                        df = pd.read_csv(f)
                        all_batch_dfs.append(df)
                    except Exception as e:
                        print(f"Error loading {f}: {e}")
                
                if all_batch_dfs:
                    final_combined_dataset = pd.concat(all_batch_dfs, ignore_index=True)
                    final_combined_dataset.to_csv('final_combined_meta_features.csv', index=False)
                else:
                    final_combined_dataset = pd.DataFrame()
        else:
            final_combined_dataset = pd.read_csv('final_combined_meta_features.csv')
            print("Read local dataset.")

        return final_combined_dataset

    @staticmethod
    def _generate_complex_synthetic(cluster_configs: list, n_samples=1000, n_features=3, 
                                     n_classes=2, random_state=42):
        rng = np.random.RandomState(random_state)
        X_parts = []
        y_parts = []
        
        samples_per_cluster = n_samples // len(cluster_configs)

        for config in cluster_configs:
            label = config.get('label', 0)
            center = np.array(config.get('center', [0] * n_features))
            
            if config.get('shape') == 'elongated':
                cov = np.eye(n_features)
                axis = config.get('main_axis', 0)
                stretch_val = config.get('width', config.get('stretch', 5.0))
                cov[axis, axis] = stretch_val
            else:
                spread = config.get('spread', 1.0)
                cov = np.eye(n_features) * (spread ** 2)

            points = rng.multivariate_normal(mean=center, cov=cov, size=samples_per_cluster)

            cluster_labels = np.full(samples_per_cluster, label)
            impurity = config.get('impurity', 0.0)
            if impurity > 0:
                n_noise = int(samples_per_cluster * impurity)
                noise_indices = rng.choice(samples_per_cluster, n_noise, replace=False)
                cluster_labels[noise_indices] = rng.randint(0, n_classes, size=n_noise)

            X_parts.append(points)
            y_parts.append(cluster_labels)

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_features)])
        df["label"] = [f"class_{l}" for l in y]
        
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)