from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from Final_Models.PipelineEight_BASE import FinalFuzzyEnsembleClassifier
from Final_Models.PipelineEight_C_MODELS import KModelsFuzzyEnsembleClassifier
from Final_Models.PipelineEight_SOFT_VOTING import SoftVotingFuzzyEnsembleClassifier
from Final_Models.PipelineEight_SAMPLES_WITH_DATASET_SIZE import SamplesWithDatasetSizeFuzzyEnsembleClassifier
from Final_Models.PipelineEight_HARD_VOTING import HardVotingFuzzyEnsembleClassifier
from Final_Models.PipelineEight_NoClassProportionAdjust import NoProportionFuzzyEnsembleClassifier
from Final_Models.PipelineEight_100_Time_K_Models import SampleSizeTimesKFuzzyEnsembleClassifier
from Final_Models.PipelineEight_NORMALIZADO import NormalizedFuzzyEnsembleClassifier
from Final_Models.PipelineEight_SMOTE_SCALER import SmoteScalerFuzzyEnsembleClassifier
from Final_Models.PipelineEight_SMOTE import SmoteFuzzyEnsembleClassifier


from DataHandler import DataHandler
import os
import time

def run_synthetic_pipeline_kfold_for_proximity(all_results):

    accuracies_mean = {}

    for proximity_val in np.arange(0, 1.05, 0.05):
        # 1. Setup Data Handler for Synthetic Data
        print("--- Generating Synthetic Data ---")
        data_handler = DataHandler(synthetic_data=True)
        
        # Generate 500 samples, 3 classes
        df_dataset = data_handler.get_datasets(            
                n_samples=500, 
                n_classes=2, 
                proximity=proximity_val, 
                impurity=0.0, 
                feature_names=['feature_1', 'feature_2'], 
                random_state=None )
        
        # Split features and target
        X = df_dataset.drop(columns=['label'])
        y = df_dataset['label']
        
        # 2. Setup 10-Fold Cross-Validation
        k_folds = 10
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        print(f"\n--- Starting {k_folds}-Fold Cross-Validation ---")

        # 3. Execution Loop
        for class_files in [NormalizedFuzzyEnsembleClassifier,
                            SmoteScalerFuzzyEnsembleClassifier,
                            SmoteFuzzyEnsembleClassifier,
                            SampleSizeTimesKFuzzyEnsembleClassifier,
                            NoProportionFuzzyEnsembleClassifier,
                            HardVotingFuzzyEnsembleClassifier,
                            SamplesWithDatasetSizeFuzzyEnsembleClassifier,
                            SoftVotingFuzzyEnsembleClassifier,
                            KModelsFuzzyEnsembleClassifier,
                            FinalFuzzyEnsembleClassifier]:
            print(f"\n--- Accuracies for model {class_files.__name__} and a Proximity of {proximity_val:.2f}")
            
            
            fold_accuracies = [] 
        
            start_time = time.time() #INÍCIO DO CRONÔMETRO para o modelo atual
            
            
            for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y), 1):
                # Split data for this fold
                X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
                y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
                
                # Instantiate a FRESH classifier for each fold to ensure no data leakage
                
                classifier = class_files(detailed_logs=False, random_state=42)
                
                try:
                    # FIT: Preprocessing -> Best K -> Clustering -> Sampling -> Training
                    classifier.fit(X_train_fold, y_train_fold)
                    
                    # PREDICT/SCORE: Preprocessing -> Voting -> Evaluation
                    acc, auc = classifier.score(X_val_fold, y_val_fold)
                    
                    print(f"Fold {fold_idx}: Accuracy = {acc:.4f}")
                    fold_accuracies.append(acc)
                    
                except Exception as e:
                    print(f"Fold {fold_idx} Failed: {e}")

            end_time = time.time() # FIM DO CRONÔMETRO
            execution_time = end_time - start_time # Tempo total em segundos
            
            # Final Aggregation
            if fold_accuracies:
                avg_acc = np.mean(fold_accuracies)

                all_results.append({
                                "Model": class_files.__name__,
                                "Proximity": round(proximity_val, 2),
                                "Impurity": 0.0,       # Fixed at 0 for this experiment
                                "Accuracy": avg_acc,    # The identifier/output column
                                "Execution_Time_Seconds": round(execution_time, 2)
                            })


                accuracies_mean[proximity_val] = avg_acc
        
    # Plot a line graph of Proximity vs. Average Accuracy

    return all_results

def run_synthetic_pipeline_kfold_for_impurity(all_results):

    accuracies_mean = {}

    for impurity_val in np.arange(0, 1.05, 0.05):
        # 1. Setup Data Handler for Synthetic Data
        print("--- Generating Synthetic Data ---")
        data_handler = DataHandler(synthetic_data=True)
        
        # Generate 500 samples, 3 classes
        df_dataset = data_handler.get_datasets(            
                n_samples=500, 
                n_classes=2, 
                proximity=0.0, 
                impurity=impurity_val, 
                feature_names=['feature_1', 'feature_2'], 
                random_state=None )
        
        # Split features and target
        X = df_dataset.drop(columns=['label'])
        y = df_dataset['label']
        
        # 2. Setup 10-Fold Cross-Validation
        k_folds = 10
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        print(f"\n--- Starting {k_folds}-Fold Cross-Validation ---")
        # 3. Execution Loop
        for class_files in [NormalizedFuzzyEnsembleClassifier,
                            SmoteScalerFuzzyEnsembleClassifier,
                            SmoteFuzzyEnsembleClassifier,
                            SampleSizeTimesKFuzzyEnsembleClassifier,
                            NoProportionFuzzyEnsembleClassifier,
                            HardVotingFuzzyEnsembleClassifier,
                            SamplesWithDatasetSizeFuzzyEnsembleClassifier,
                            SoftVotingFuzzyEnsembleClassifier,
                            KModelsFuzzyEnsembleClassifier,
                            FinalFuzzyEnsembleClassifier]:
            
            fold_accuracies = [] 
        
            start_time = time.time() # INÍCIO DO CRONÔMETRO para o modelo atual
            
            print(f"\n--- Accuracies for model {class_files.__name__} and a Impurity of {impurity_val:.2f}")
            for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y), 1):
                # Split data for this fold
                X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
                y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
                
                # Instantiate a FRESH classifier for each fold to ensure no data leakage
                classifier = class_files(detailed_logs=False, random_state=42)
                
                try:
                    # FIT: Preprocessing -> Best K -> Clustering -> Sampling -> Training
                    classifier.fit(X_train_fold, y_train_fold)
                    
                    # PREDICT/SCORE: Preprocessing -> Voting -> Evaluation
                    acc, auc = classifier.score(X_val_fold, y_val_fold)
                    
                    print(f"Fold {fold_idx}: Accuracy = {acc:.4f}")
                    fold_accuracies.append(acc)
                    
                except Exception as e:
                    print(f"Fold {fold_idx} Failed: {e}")

            end_time = time.time() # FIM DO CRONÔMETRO
            execution_time = end_time - start_time # Tempo total em segundos
            
            # 4. Final Aggregation
            if fold_accuracies:
                avg_acc = np.mean(fold_accuracies)

                all_results.append({
                    "Model": class_files.__name__,
                    "Proximity": 0.0,
                    "Impurity": round(impurity_val, 2), 
                    "Accuracy": avg_acc,    # The identifier/output column
                    "Execution_Time_Seconds": round(execution_time, 2)
                })

                accuracies_mean[impurity_val] = avg_acc
    return all_results

if __name__ == "__main__":
    all_results = []
    all_results = run_synthetic_pipeline_kfold_for_impurity(all_results)
    all_results = run_synthetic_pipeline_kfold_for_proximity(all_results)
    
    # Convert the list to a DataFrame and save
    df_results = pd.DataFrame(all_results)

    # Get current time for unique filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the filename
    filename = f"Final_Models/test_{current_time}.csv"

    # Reorder columns to ensure exact requested format
    df_results = df_results[["Model", "Proximity", "Impurity", "Accuracy", "Execution_Time_Seconds"]]

    # Check if file exists to determine if we need to write the header
    file_exists = os.path.isfile(filename)

    # Save with mode='a' (append); only write header if file does NOT exist
    df_results.to_csv(filename, mode='a', index=False, header=not file_exists)

    print(f"Results appended to {filename}")