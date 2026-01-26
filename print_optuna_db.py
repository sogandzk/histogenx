import optuna
import pandas as pd

# 1. Load your study
study = optuna.load_study(study_name="Dec_25_2025_phase_2", storage="sqlite:///experiments/optuna.db")
df = study.trials_dataframe()

# 2. Extract the max epoch for each trial (The "History")
# This creates a dictionary of {trial_number: max_epoch}
epoch_counts = {
    trial.number: max(trial.intermediate_values.keys()) + 1 if trial.intermediate_values else 0 
    for trial in study.trials
}

# 3. Add the epoch count to the DataFrame
df['epochs_run'] = df['number'].map(epoch_counts)

# 4. Keep only the columns I need for analysis
analysis_cols = ['number', 'value', 'state', 'epochs_run', 'duration'] + \
                [col for col in df.columns if col.startswith('params_')]

summary = df[analysis_cols]

# 5. Output for me
print("--- START OF DATA ---")
print(summary.to_csv(index=False))
print("--- END OF DATA ---")
