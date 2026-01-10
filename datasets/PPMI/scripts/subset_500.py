from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Path to the current script
script_dir = Path(__file__).parent       # scripts/
ppmi_dir = script_dir.parent             # PPMI/
metadata_dir = ppmi_dir / "metadata"     # metadata/

def create_500_subset():
    """
    Curate 500 subjects from PPMI with:
    - 70/15/15 train/test/val split
    - Keep ALL healthy controls (57)
    - Stratified sampling to maintain diagnosis distribution
    """
    print("--- Creating 500 Subject Subset ---")
    
    # Load data
    df = pd.read_csv(metadata_dir / 'PPMI_Hybrid_Cases.csv')
    
    # Get unique subjects with their diagnosis
    subjects = df.drop_duplicates('PATNO')[['PATNO', 'Diagnosis']]
    
    print(f"\nOriginal distribution ({len(subjects)} subjects):")
    print(subjects['Diagnosis'].value_counts())
    
    # Split by diagnosis
    hc = subjects[subjects['Diagnosis'] == 'Healthy Control']
    pd_subj = subjects[subjects['Diagnosis'] == "Parkinson's Disease"]
    prodromal = subjects[subjects['Diagnosis'] == 'Prodromal']
    
    # Keep ALL healthy controls (57)
    n_hc = len(hc)
    n_remaining = 500 - n_hc  # 443 spots for PD + Prodromal
    
    # Maintain original ratio of PD:Prodromal
    # Original: 246 PD, 431 Prodromal -> ~36.3% PD, ~63.7% Prodromal
    pd_ratio = len(pd_subj) / (len(pd_subj) + len(prodromal))
    
    n_pd_target = int(np.round(n_remaining * pd_ratio))
    n_prodromal_target = n_remaining - n_pd_target
    
    print(f"\nTarget counts for 500 subjects:")
    print(f"  Healthy Control: {n_hc} (all)")
    print(f"  Parkinson's Disease: {n_pd_target}")
    print(f"  Prodromal: {n_prodromal_target}")
    
    # Random sample from PD and Prodromal
    np.random.seed(42)  # For reproducibility
    
    pd_sampled = pd_subj.sample(n=n_pd_target, random_state=42)
    prodromal_sampled = prodromal.sample(n=n_prodromal_target, random_state=42)
    
    # Combine all selected subjects
    selected_subjects = pd.concat([hc, pd_sampled, prodromal_sampled])
    
    print(f"\nSelected 500 subjects distribution:")
    print(selected_subjects['Diagnosis'].value_counts())
    
    # Now split into train/test/val (70/15/15) with stratification
    # First split: train (70%) vs temp (30%)
    train_subj, temp_subj = train_test_split(
        selected_subjects,
        test_size=0.30,
        stratify=selected_subjects['Diagnosis'],
        random_state=42
    )
    
    # Second split: test (50% of 30% = 15%) vs val (50% of 30% = 15%)
    test_subj, val_subj = train_test_split(
        temp_subj,
        test_size=0.50,
        stratify=temp_subj['Diagnosis'],
        random_state=42
    )
    
    # Assign split labels
    train_subj = train_subj.copy()
    test_subj = test_subj.copy()
    val_subj = val_subj.copy()
    
    train_subj['Split'] = 'train'
    test_subj['Split'] = 'test'
    val_subj['Split'] = 'val'
    
    split_info = pd.concat([train_subj, test_subj, val_subj])
    
    # Print split statistics
    print("\n--- Split Statistics ---")
    print(f"Train: {len(train_subj)} subjects ({len(train_subj)/500*100:.1f}%)")
    print(f"Test:  {len(test_subj)} subjects ({len(test_subj)/500*100:.1f}%)")
    print(f"Val:   {len(val_subj)} subjects ({len(val_subj)/500*100:.1f}%)")
    
    print("\n--- Diagnosis Distribution per Split ---")
    for split_name in ['train', 'test', 'val']:
        subset = split_info[split_info['Split'] == split_name]
        print(f"\n{split_name.upper()}:")
        for diag, count in subset['Diagnosis'].value_counts().items():
            pct = count / len(subset) * 100
            print(f"  {diag}: {count} ({pct:.1f}%)")
    
    # Merge split info back to full dataframe (all sessions)
    df_subset = df[df['PATNO'].isin(selected_subjects['PATNO'])].copy()
    df_subset = df_subset.merge(
        split_info[['PATNO', 'Split']], 
        on='PATNO', 
        how='left'
    )
    
    # Reorder columns to put Split early
    cols = df_subset.columns.tolist()
    cols.remove('Split')
    cols.insert(3, 'Split')
    df_subset = df_subset[cols]
    
    # Sort for nice output
    df_subset.sort_values(by=['Split', 'Diagnosis', 'PATNO', 'EVENT_ID'], inplace=True)
    
    # Save
    output_file = metadata_dir / 'PPMI_500_Curated.csv'
    df_subset.to_csv(output_file, index=False)
    
    print(f"\n--- SAVED: {output_file} ---")
    print(f"Total rows (sessions): {len(df_subset)}")
    print(f"Total subjects: {df_subset['PATNO'].nunique()}")
    
    # Also save a subject-level split file for easy reference
    split_ref = split_info[['PATNO', 'Diagnosis', 'Split']].sort_values(['Split', 'Diagnosis', 'PATNO'])
    split_ref.to_csv(metadata_dir / 'PPMI_500_Split_Reference.csv', index=False)
    print(f"Split reference saved to: PPMI_500_Split_Reference.csv")

if __name__ == "__main__":
    create_500_subset()

