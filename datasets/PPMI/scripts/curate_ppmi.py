from pathlib import Path
import pandas as pd
import numpy as np

# Path to the current script
script_dir = Path(__file__).parent  # scripts/
ppmi_dir = script_dir.parent         # PPMI/
metadata_dir = ppmi_dir / "metadata"


def curate_ppmi_hybrid_complete():
    print("--- Starting PPMI Curation (Hybrid Complete Cases) ---")

    # 1. Load the Master Subject List (786 Subjects)
    try:
        df_img = pd.read_csv(metadata_dir / "PPMI_T1+fMRI_gt200TR_valid_subjects.csv")
        # Standardize columns
        if 'Subject' in df_img.columns:
            df_img.rename(columns={'Subject': 'PATNO', 'Visit': 'EVENT_ID'}, inplace=True)
        elif 'Subject ID' in df_img.columns:
            df_img.rename(columns={'Subject ID': 'PATNO', 'Visit': 'EVENT_ID'}, inplace=True)
        
        # Skeleton: Unique Subject + Visit
        master = df_img[['PATNO', 'EVENT_ID']].drop_duplicates()
        demog = df_img[['PATNO', 'Sex']].drop_duplicates()
        
        print(f"Initial Pool: {len(master)} sessions from {master['PATNO'].nunique()} subjects.")
    except FileNotFoundError:
        print("ERROR: Could not find 'metadata_filtered_bold_t1w.csv'.")
        return

    # 2. Load Status
    status = pd.read_csv(metadata_dir / "Participant_Status_26Nov2025.csv")
    df_status = status[['PATNO', 'COHORT_DEFINITION', 'ENROLL_AGE']].rename(columns={'COHORT_DEFINITION': 'Diagnosis'})

    # 3. Load & Process Clinical Data
    # UPDRS III (Motor) + Tremor/PIGD
    updrs3 = pd.read_csv(metadata_dir / "MDS-UPDRS_Part_III_26Nov2025.csv")
    tremor_cols = ['NP3PTRMR', 'NP3PTRML', 'NP3KTRMR', 'NP3KTRML', 'NP3RTARU', 'NP3RTALU', 'NP3RTARL', 'NP3RTALL', 'NP3RTALJ', 'NP3RTCON']
    pigd_cols = ['NP3RISNG', 'NP3GAIT', 'NP3FRZGT', 'NP3PSTBL', 'NP3POSTR']
    updrs3['Tremor_Mean'] = updrs3[tremor_cols].mean(axis=1)
    updrs3['PIGD_Mean'] = updrs3[pigd_cols].mean(axis=1)
    
    df_u3 = updrs3[['PATNO', 'EVENT_ID', 'INFODT', 'NP3TOT', 'NHY', 'Tremor_Mean', 'PIGD_Mean']].copy()
    df_u3.rename(columns={'NP3TOT': 'UPDRS_III', 'NHY': 'HoehnYahr'}, inplace=True)
    df_u3['HoehnYahr'] = df_u3['HoehnYahr'].replace(101.0, np.nan)

    # UPDRS II & I
    updrs2 = pd.read_csv(metadata_dir / "MDS_UPDRS_Part_II__Patient_Questionnaire_26Nov2025.csv")
    df_u2 = updrs2[['PATNO', 'EVENT_ID', 'NP2PTOT']].rename(columns={'NP2PTOT': 'UPDRS_II'})
    
    updrs1_pt = pd.read_csv(metadata_dir / "MDS-UPDRS_Part_I_Patient_Questionnaire_26Nov2025.csv")
    df_u1_pt = updrs1_pt[['PATNO', 'EVENT_ID', 'NP1PTOT']].rename(columns={'NP1PTOT': 'UPDRS_I_Patient'})
    
    updrs1_rater = pd.read_csv(metadata_dir / "MDS-UPDRS_Part_I_26Nov2025.csv")
    df_u1_rater = updrs1_rater[['PATNO', 'EVENT_ID', 'NP1RTOT']].rename(columns={'NP1RTOT': 'UPDRS_I_Rater'})

    # MoCA
    moca = pd.read_csv(metadata_dir / "Montreal_Cognitive_Assessment__MoCA__26Nov2025.csv")
    df_moca = moca[['PATNO', 'EVENT_ID', 'MCATOT']].rename(columns={'MCATOT': 'MoCA'})

    # GDS (Depression)
    gds = pd.read_csv(metadata_dir / "Geriatric_Depression_Scale__Short_Version__26Nov2025.csv")
    rev_cols = ['GDSSATIS', 'GDSGSPIR', 'GDSHAPPY', 'GDSALIVE', 'GDSENRGY']
    gds_sc = gds.copy()
    for c in rev_cols:
        if c in gds_sc.columns: gds_sc[c] = 1 - gds_sc[c]
    gds_cols = [c for c in gds.columns if c.startswith('GDS') and c not in ['REC_ID', 'PATNO', 'EVENT_ID', 'PAG_NAME', 'INFODT', 'ORIG_ENTRY', 'LAST_UPDATE']]
    gds_sc['GDS_Total'] = gds_sc[gds_cols].sum(axis=1)
    df_gds = gds_sc[['PATNO', 'EVENT_ID', 'GDS_Total']]

    # ESS (Sleepiness)
    ess = pd.read_csv(metadata_dir / "Epworth_Sleepiness_Scale_26Nov2025.csv")
    ess_cols = [f'ESS{i}' for i in range(1, 9)]
    ess['ESS_Total'] = ess[ess_cols].sum(axis=1) if all(c in ess.columns for c in ess_cols) else np.nan
    df_ess = ess[['PATNO', 'EVENT_ID', 'ESS_Total']]

    # RBD (REM Sleep)
    rbd = pd.read_csv(metadata_dir / "REM_Sleep_Behavior_Disorder_Questionnaire_26Nov2025.csv")
    rbd_items = ['DRMVIVID', 'DRMAGRAC', 'DRMNOCTB', 'SLPLMBMV', 'SLPINJUR', 'DRMVERBL', 'DRMFIGHT', 'DRMUMV', 'DRMOBJFL', 'MVAWAKEN', 'DRMREMEM', 'SLPDSTRB']
    existing_items = [c for c in rbd_items if c in rbd.columns]
    rbd['RBD_Total_Score'] = rbd[existing_items].sum(axis=1)
    df_rbd = rbd[['PATNO', 'EVENT_ID', 'RBD_Total_Score']]

    # 4. Merge
    print("Merging Data...")
    master = master.merge(df_status, on='PATNO', how='left')
    master = master.merge(demog, on='PATNO', how='left')
    master = master[master['Diagnosis'] != 'SWEDD'] # Drop SWEDD

    master = master.merge(df_u3, on=['PATNO', 'EVENT_ID'], how='left')
    master = master.merge(df_u2, on=['PATNO', 'EVENT_ID'], how='left')
    master = master.merge(df_u1_pt, on=['PATNO', 'EVENT_ID'], how='left')
    master = master.merge(df_u1_rater, on=['PATNO', 'EVENT_ID'], how='left')
    master = master.merge(df_moca, on=['PATNO', 'EVENT_ID'], how='left')
    master = master.merge(df_gds, on=['PATNO', 'EVENT_ID'], how='left')
    master = master.merge(df_ess, on=['PATNO', 'EVENT_ID'], how='left')
    master = master.merge(df_rbd, on=['PATNO', 'EVENT_ID'], how='left')

    # 5. HYBRID FILTERING
    print("Applying Hybrid Filter...")
    
    # Define columns that MUST exist for SICK patients (PD/Prodromal)
    critical_columns = [
        'Diagnosis', 'Sex', 'ENROLL_AGE',
        'UPDRS_III', 'MoCA', 'GDS_Total', 'ESS_Total', 
        'RBD_Total_Score', 'Tremor_Mean', 'PIGD_Mean'
    ]
    
    # Split Dataset
    mask_control = master['Diagnosis'] == 'Healthy Control'
    df_controls = master[mask_control].copy()
    df_others = master[~mask_control].copy()
    
    # A. Filter "Others" STRICTLY
    initial_others = len(df_others)
    df_others_clean = df_others.dropna(subset=critical_columns)
    print(f"Filtered Others (PD/Prodromal): Dropped {initial_others - len(df_others_clean)} incomplete rows.")
    
    # B. Keep "Controls" LENIENTLY
    # We only ensure they have Diagnosis/Age/Sex. We allow missing clinical scores.
    # To prevent issues later, we IMPUTE 'Normal' values (0) for missing scores in controls.
    fill_values = {
        'UPDRS_III': 0, 'GDS_Total': 0, 'ESS_Total': 0, 
        'RBD_Total_Score': 0, 'Tremor_Mean': 0, 'PIGD_Mean': 0
    }
    df_controls.fillna(value=fill_values, inplace=True)
    
    # For MoCA, filling 0 is bad (0 = dementia). Fill with Control Mean or 28 (Normal).
    # Using 28 as a safe default for a Healthy Control missing the test.
    df_controls['MoCA'].fillna(28.0, inplace=True)
    
    print(f"Controls kept: {len(df_controls)} rows (Imputed missing scores with Normal/0).")

    # Recombine
    master_clean = pd.concat([df_others_clean, df_controls])

    # 6. Save
    master_clean.sort_values(by=['Diagnosis', 'PATNO', 'EVENT_ID'], inplace=True)
    output_file = metadata_dir / 'PPMI_Hybrid_Cases.csv'
    master_clean.to_csv(output_file, index=False)

    print("-" * 40)
    print(f"SUCCESS. Saved to {output_file}")
    print(f"Total Unique Subjects: {master_clean['PATNO'].nunique()}")
    print("-" * 40)
    print("Final Diagnosis Distribution:")
    print(master_clean.drop_duplicates('PATNO')['Diagnosis'].value_counts())

if __name__ == "__main__":
    curate_ppmi_hybrid_complete()