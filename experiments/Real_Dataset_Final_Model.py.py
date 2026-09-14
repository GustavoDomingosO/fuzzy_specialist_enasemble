from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

from Final_Models.PipelineEight_BASE import FinalFuzzyEnsembleClassifier
from models.PipelineSimplest import SimplestClassifier
from models.PipelineBaseline import BaselineClassifier

import os
import time

def pipeline_kfold_real_data(all_results, df):

    X = df.drop(columns=['target'])
    y = df['target']
    
    # 2. Setup 5-Fold Cross-Validation
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=100)
    
    # 3. Execution Loop
    for class_files in [SimplestClassifier,
                            BaselineClassifier]:
        
        fold_accuracies = [] # Reset for every model
        fold_auc = []

        # START TIMER
        start_time = time.perf_counter()

        print(f"\n--- Accuracies for model {class_files.__name__}")
        for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y), 1):

            # Split data for this fold
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            X_train_fold = pd.DataFrame(X_train_fold, columns=X.columns)
            X_val_fold = pd.DataFrame(X_val_fold, columns=X.columns)

            # Instantiate classifier
            classifier = class_files(
                detailed_logs=False,
                random_state=100
            )

            try:
                classifier.fit(X_train_fold, y_train_fold)

                acc, auc = classifier.score(
                    X_val_fold,
                    y_val_fold
                )

                print(f"Fold {fold_idx}: Accuracy = {acc:.4f}")
                fold_accuracies.append(acc)
                fold_auc.append(auc)

            except Exception as e:
                print(f"Fold {fold_idx} Failed: {e}")
        # Final Aggregation
        if fold_accuracies:

            processing_time = time.perf_counter() - start_time

            all_results.append({
                            "Model": class_files.__name__,
                            "Mean_Accuracy": np.mean(fold_accuracies),
                            "STD_Accuracy": np.std(fold_accuracies),
                            "Mean_AUC": np.mean(fold_auc),
                            "STD_AUC":  np.std(fold_auc),
                            "Processing_Time_Seconds": processing_time,
                        })
            
    return all_results

if __name__ == "__main__":
    folder_path = Path('pmlb_datasets')
    
    # 1. Initialize the list OUTSIDE the loop to collect ALL results
    all_results = []

    # Loop through all files ending in .csv
    for file_path in folder_path.glob('*.csv'):
        print(f"\nProcessing dataset: {file_path.name}")
        
        # Load the dataset
        df = pd.read_csv(file_path)

        if df.shape[0] > 5000:
            print(f"Skipping dataset {file_path.name} due to size constraints.")
            continue

        # 2. Pass the cumulative list to the function
        # Note: Added a 'dataset_name' logic inside or outside to track data
        temp_results = []
        try:
            temp_results = pipeline_kfold_real_data(temp_results, df)
        except Exception as e:
            print(f"Error processing dataset {file_path.name}: {e}")
            continue
        
        # Add the dataset name to each result row for clarity
        for result in temp_results:
            result['Dataset'] = file_path.name
            all_results.append(result)

    # 3. After the loop finishes, convert the full list to a DataFrame
    if all_results:
        df_results = pd.DataFrame(all_results)

        # Get current time for a unique filename
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_results_not_prepross{current_time}.csv"

        # Reorder columns (added Dataset at the front for organization)
        cols = ["Dataset", "Model",  "Mean_Accuracy","STD_Accuracy","Mean_AUC","STD_AUC","Processing_Time_Seconds"]
        df_results = df_results[cols]

        # Save once
        df_results.to_csv(filename, index=False)
        print(f"\n--- Process Complete ---")
        print(f"All results saved to: {filename}")