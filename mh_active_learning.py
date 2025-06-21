import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mh_optimize as mho



##  svm: Support Vector Machine
##  dt: Decision Tree
##  rf: Random Forest
##  lr: Logistic Regression

MODEL_NAMES = ['rf']  ##  or can be ['svm', 'dt', 'rf'] or ['svm', 'rf', 'lr']
# MODEL_NAMES = ['svm', 'rf', 'lr']  ##  or can be ['svm', 'dt', 'rf'] or ['svm', 'rf', 'lr']
# MODEL_NAMES = ['rf']  ##  or can be ['svm', 'dt', 'rf'] or ['svm', 'rf', 'lr']
# MODEL_NAMES = ['lr']  ##  or can be ['svm', 'dt', 'rf'] or ['svm', 'rf', 'lr']
BETA = 50  ##  Number of pairs to be manually labelled in each iteration
alpha = 1.  ##  Accuracy constraint for the optimization

##  Step 0: Load the dataset
df = pd.read_csv('cifar10_metrics.csv')
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

X_labelled, X_unlabelled, y_labelled, y_unlabelled = train_test_split(X, y, test_size=0.8, random_state=42, stratify=y)

##  Step 1: Train the models on the training set
def createModel(model_name):
    match model_name:
        case 'svm':
            return SVC(probability=True)
        case 'dt':
            return DecisionTreeClassifier()
        case 'rf':
            return RandomForestClassifier()
        case 'lr':
            return LogisticRegression(max_iter=1000)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
        
models = [createModel(model_name) for model_name in MODEL_NAMES]    

##  Step 2: Active learning loop
##  We use the pretrained models to predict the labels of the test set
def active_learning(models, X_labelled, y_labelled, X_unlabelled, y_unlabelled, beta):
    print('Initially labelled samples:', len(X_labelled))
    print('Initially unlabelled samples:', len(X_unlabelled))
    print('Total samples:', len(X_labelled) + len(X_unlabelled))
    print('Beta (manual samples per iteration):', beta)
    n_iterations = len(X_unlabelled) // beta
    final_labels = []  ##  For storing the final labels (those that are manually labelled + those that are automatically labelled)
    manual_effort = len(X_labelled)  ##  Initial manually labeled samples
    print(f"Total iterations: {n_iterations}")

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")
        
        # Step 2.0: Split the labelled into training and optimization sets (stratified)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, opt_idx = next(sss.split(X_labelled, y_labelled))
        X_train, X_opt = X_labelled[train_idx], X_labelled[opt_idx]
        y_train, y_opt_true = y_labelled[train_idx], y_labelled[opt_idx]
        print(f"Training set size: {len(X_train)}, Optimization set size: {len(X_opt)}")

        # Step 2.1: Train the models on the labelled data
        for model in models:
            model.fit(X_train, y_train)
            
        # Step 2.2: Predict the labels of the optimization data and prepare DataFrame for mh_optimize
        pred_cols = []
        df_opt_data = {}
        
        for i, model in enumerate(models):
            y_pred = model.predict(X_opt)
            y_proba = model.predict_proba(X_opt)
            max_proba = np.max(y_proba, axis=1)
            
            df_opt_data[f'p_l_{i}'] = y_pred
            df_opt_data[f'p_theta_{i}'] = max_proba
        
        df_opt_data['y'] = y_opt_true
        df_pred_opt = pd.DataFrame(df_opt_data)

        # Step 2.3: Use mh_optimize to find the best model weights
        effort_value, w_list, x_list = mho.mh_optimize(df_pred_opt, alpha=alpha)
        print(f"Optimization weights: {w_list}")

        # Step 2.4: Predict the labels of the unlabelled data
        y_unlabelled_pred = np.array([model.predict(X_unlabelled) for model in models]).T
        prob_unlabelled = np.array([model.predict_proba(X_unlabelled) for model in models])
        
        # Calculate z values for unlabelled data (check if all models agree)
        if len(models) > 1:
            # Check if all predictions are the same for each sample
            z_unlabelled = np.all(y_unlabelled_pred == y_unlabelled_pred[:, 0:1], axis=1).astype(int)
        else:
            z_unlabelled = np.ones(len(X_unlabelled), dtype=int)
        
        # Get max probabilities for each model
        max_prob_unlabelled = np.array([np.max(prob, axis=1) for prob in prob_unlabelled]).T
        
        # Calculate f values using weights from optimization
        f_unlabelled = np.sum([w_list[i] * max_prob_unlabelled[:, i] for i in range(len(models))], axis=0) - 1

        # Step 2.5: Select beta samples to process in this iteration
        # First, identify which samples can be auto-labeled (f > 0 and all models agree)
        auto_condition = (f_unlabelled > 0) & (z_unlabelled == 1)
        auto_eligible_indices = np.where(auto_condition)[0]
        
        # Process at most beta samples in this iteration
        if len(auto_eligible_indices) >= beta:
            # If we have enough auto-labelable samples, select beta of them (highest f values)
            f_auto_eligible = f_unlabelled[auto_eligible_indices]
            auto_selection = np.argsort(f_auto_eligible)[-beta:]  # Select highest f values
            indices_auto = auto_eligible_indices[auto_selection]
            indices_manually = np.array([])
        else:
            # Auto-label all eligible samples
            indices_auto = auto_eligible_indices
            
            # For remaining quota, select samples for manual labeling (lowest f values)
            remaining_quota = beta - len(indices_auto)
            non_auto_indices = np.where(~auto_condition)[0]
            
            if len(non_auto_indices) > 0 and remaining_quota > 0:
                f_non_auto = f_unlabelled[non_auto_indices]
                n_manual = min(remaining_quota, len(non_auto_indices))
                manual_selection = np.argsort(f_non_auto)[:n_manual]  # Select lowest f values
                indices_manually = non_auto_indices[manual_selection]
            else:
                indices_manually = np.array([])

        # Step 2.6: Add the manually labelled pairs to the labelled set
        if len(indices_manually) > 0:
            # Add manually labeled samples to labeled set
            X_labelled = np.vstack([X_labelled, X_unlabelled[indices_manually]])
            y_labelled = np.hstack([y_labelled, y_unlabelled[indices_manually]])
            # Update manual effort counter
            manual_effort += len(indices_manually)
        
        # Auto-label and store final labels
        if len(indices_auto) > 0:
            # Use consensus prediction for auto-labeled samples
            auto_labels = y_unlabelled_pred[indices_auto, 0]  # Since all models agree
            final_labels.extend(list(zip(indices_auto, auto_labels, y_unlabelled[indices_auto])))
        
        # Remove both auto-labeled and manually labeled samples from unlabeled set
        all_remove_indices = np.concatenate([indices_auto, indices_manually]) if len(indices_manually) > 0 else indices_auto
        if len(all_remove_indices) > 0:
            mask = np.ones(len(X_unlabelled), dtype=bool)
            mask[all_remove_indices] = False
            X_unlabelled = X_unlabelled[mask]
            y_unlabelled = y_unlabelled[mask]
        
        print(f"Auto-labeled: {len(indices_auto)}, Manually labeled: {len(indices_manually)}, Remaining: {len(X_unlabelled)}")
        
        if len(X_unlabelled) == 0:
            break
    
    return final_labels, manual_effort

# Run the active learning experiment
final_labels, manual_effort = active_learning(models, X_labelled, y_labelled, X_unlabelled, y_unlabelled, BETA)

# Store initial labeled data for overall metrics calculation
initial_X_labelled = X_labelled.copy()
initial_y_labelled = y_labelled.copy()

# Calculate total dataset size
total_samples = len(X) 

# Evaluation: Calculate metrics for the entire processed dataset
if final_labels:
    auto_indices, predicted_labels, true_labels = zip(*final_labels)
    
    # Calculate metrics for auto-labeled samples only
    auto_accuracy = accuracy_score(true_labels, predicted_labels)
    auto_precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    auto_recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    auto_f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    print(f"\nEvaluation Results for Auto-labeled Samples:")
    print(f"Total auto-labeled samples: {len(final_labels)}")
    print(f"Auto-label Accuracy: {auto_accuracy:.4f}")
    print(f"Auto-label Precision: {auto_precision:.4f}")
    print(f"Auto-label Recall: {auto_recall:.4f}")
    print(f"Auto-label F1-score: {auto_f1:.4f}")
    
    # Calculate overall metrics for all processed samples (initial + auto-labeled)
    # Initial samples are 100% accurate (human labeled), auto-labeled have calculated accuracy
    total_processed_samples = len(initial_X_labelled) + len(final_labels)
    correct_initial_samples = len(initial_X_labelled)  # All initial samples are correct
    correct_auto_samples = sum([1 for pred, true in zip(predicted_labels, true_labels) if pred == true])
    
    overall_accuracy = (correct_initial_samples + correct_auto_samples) / total_processed_samples
    
    # For precision, recall, F1: combine initial and auto-labeled samples
    all_predicted = list(initial_y_labelled) + list(predicted_labels)
    all_true = list(initial_y_labelled) + list(true_labels)
    
    overall_precision = precision_score(all_true, all_predicted, average='weighted', zero_division=0)
    overall_recall = recall_score(all_true, all_predicted, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_true, all_predicted, average='weighted', zero_division=0)
    
    print(f"\nOverall Evaluation Results (Initial + Auto-labeled):")
    print(f"Total processed samples: {total_processed_samples}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1-score: {overall_f1:.4f}")
    
    print(f"\nOverall Dataset Statistics:")
    print(f"Total dataset size: {total_samples}")
    print(f"Manual effort (human-labeled samples): {manual_effort}")
    print(f"Auto-labeled samples: {len(final_labels)}")
    print(f"Remaining unlabeled samples: {total_samples - manual_effort - len(final_labels)}")
    print(f"Manual effort ratio: {manual_effort / total_samples:.4f}")
    print(f"Auto-labeling ratio: {len(final_labels) / total_samples:.4f}")
    
else:
    print("No samples were auto-labeled.")
    
    # Even if no auto-labeling occurred, we still have the initial labeled samples
    print(f"\nOverall Evaluation Results (Initial samples only):")
    print(f"Total processed samples: {len(initial_X_labelled)}")
    print(f"Overall Accuracy: 1.0000 (initial samples are correctly labeled)")
    print(f"Overall Precision: 1.0000")
    print(f"Overall Recall: 1.0000") 
    print(f"Overall F1-score: 1.0000")
    
    print(f"\nOverall Dataset Statistics:")
    print(f"Total dataset size: {total_samples}")
    print(f"Manual effort (human-labeled samples): {manual_effort}")
    print(f"Manual effort ratio: {manual_effort / total_samples:.4f}")



        
        
