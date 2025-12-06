"""
Version 11: Meta-Learning Ensemble with Stacking
==================================================

Advanced ensemble using cross-validation stacking to combine:
- RSF (Random Survival Forest)
- XGBoost Survival
- CoxPH (Cox Proportional Hazards)

Target: C-index > 0.75
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from tqdm import tqdm

from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 80)
print("VERSION 11: META-LEARNING ENSEMBLE WITH STACKING")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"C:\Users\guill\Desktop\Data Challenge QRT\Data-Challenge-Prediction-de-Survie"

# RSF: 90 best features (from V4)
RSF_FEATURES = [
    'BM_BLAST', 'HB', 'PLT', 'vaf_sum', 'mutation_count_total', 'effect_non_synonymous_codon',
    'gene_RUNX1_count', 'WBC', 'cyto_total_anomalies', 'gene_RUNX1_present', 'ANC', 'cyto_normal',
    'MONOCYTES', 'vaf_mean', 'gene_TP53_count', 'cyto_complex', 'gene_TP53_present', 'cyto_loss_count',
    'effect_frameshift_variant', 'gene_NRAS_present', 'gene_NRAS_count', 'CENTER_PV', 'effect_PTD',
    'gene_ASXL1_count', 'cyto_chr7_affected', 'gene_TET2_count', 'gene_SF3B1_present', 'vaf_max',
    'gene_DNMT3A_present', 'gene_U2AF1_present', 'gene_SF3B1_count', 'CENTER_RMCN', 'cyto_chr18_affected',
    'cyto_gain_count', 'gene_U2AF1_count', 'cyto_chr1_affected', 'cyto_chrX_affected', 'cyto_other_count',
    'cyto_transloc_count', 'cyto_chr3_affected', 'gene_NF1_count', 'cyto_chr7_abnormal', 'gene_EZH2_present',
    'cyto_chr6_affected', 'CENTER_TUD', 'gene_SRSF2_count', 'gene_DNMT3A_count', 'gene_PHF6_count',
    'gene_TET2_present', 'CENTER_ROM', 'effect_inframe_codon_gain', 'gene_BCOR_count', 'CENTER_GESMD',
    'gene_CUX1_present', 'cyto_chr4_affected', 'gene_ZRSR2_count', 'gene_ZRSR2_present', 'CENTER_DUS',
    'cyto_chrY_affected', 'CENTER_CGM', 'cyto_del5q', 'cyto_chr15_affected', 'cyto_chr14_affected',
    'gene_NF1_present', 'CENTER_DUTH', 'CENTER_HIAE', 'effect_complex_change_in_transcript',
    'effect_initiator_codon_change', 'cyto_chr3_abnormal', 'cyto_chr19_affected', 'cyto_chr10_affected',
    'CENTER_UMG', 'CENTER_REL', 'CENTER_IHBT', 'CENTER_HMS', 'CENTER_MSK', 'effect_3_prime_UTR_variant',
    'effect_2KB_upstream_variant', 'effect_ITD', 'effect_stop_lost', 'cyto_inv_count',
    'effect_stop_retained_variant', 'effect_inframe_variant', 'effect_synonymous_codon', 'CENTER_VU',
    'CENTER_UOXF', 'CENTER_UOB', 'gene_KRAS_present', 'gene_KRAS_count', 'cyto_has_inv'
]

RSF_PARAMS = {
    'n_estimators': 350,
    'min_samples_split': 27,
    'min_samples_leaf': 5,
    'max_features': 0.2,
    'max_depth': 23,
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 0
}

# XGBoost parameters (from V6.1)
XGB_PARAMS = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'tree_method': 'hist',
    'learning_rate': 0.0744,
    'max_depth': 3,
    'min_child_weight': 9,
    'subsample': 0.6194,
    'colsample_bytree': 0.8872,
    'reg_alpha': 0.0124,
    'reg_lambda': 0.0043,
    'gamma': 0.0600,
}

# CoxPH: 69 features (from V9)
COXPH_FEATURES = [
    'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT', 'CENTER_DUS', 'CENTER_GESMD', 'CENTER_HMS', 'CENTER_IHBT', 
    'CENTER_MUV', 'CENTER_PV', 'CENTER_ROM', 'CENTER_TUD', 'CENTER_UOXF', 'vaf_max', 'effect_PTD', 
    'effect_inframe_codon_loss', 'effect_non_synonymous_codon', 'gene_TET2_present', 'gene_ASXL1_present', 
    'gene_SF3B1_present', 'gene_DNMT3A_present', 'gene_RUNX1_present', 'gene_SRSF2_count', 'gene_TP53_count', 
    'gene_STAG2_present', 'gene_EZH2_count', 'gene_CBL_present', 'gene_NRAS_present', 'gene_ZRSR2_count', 
    'gene_ZRSR2_present', 'gene_CUX1_count', 'gene_PHF6_count', 'gene_KRAS_present', 'cyto_del_count', 
    'cyto_has_transloc', 'cyto_loss_count', 'cyto_has_loss', 'cyto_other_count', 'cyto_normal', 
    'cyto_chr1_affected', 'cyto_chr2_affected', 'cyto_chr4_affected', 'cyto_chr7_affected', 'cyto_chr8_affected', 
    'cyto_chr9_affected', 'cyto_chr11_affected', 'cyto_chr13_affected', 'cyto_chr14_affected', 'cyto_chr17_affected', 
    'cyto_chr18_affected', 'cyto_chr19_affected', 'cyto_chrX_affected', 'cyto_monosomy7', 'cyto_chr3_abnormal', 
    'anc_ratio', 'platelet_to_blast', 'vaf_mutation_burden', 'blast_to_mutation', 'wbc_plt_index', 
    'monocyte_blast_ratio', 'blast_mutation_interaction', 'cyto_mutation_interaction', 'chr7_blast', 
    'vaf_tp53', 'log_wbc', 'log_plt', 'log_blast', 'log_mutation_count'
]

COXPH_PARAMS = {
    'l1_ratio': 0.9025866771499649,
    'fit_baseline_model': True
}

N_FOLDS = 5  # Cross-validation folds

print(f"✓ Configuration loaded")
print(f"  RSF:    {len(RSF_FEATURES)} features")
print(f"  XGBoost: Auto-selection")
print(f"  CoxPH:  {len(COXPH_FEATURES)} features")
print(f"  CV Folds: {N_FOLDS}")
print()

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS (From V10)
# ============================================================================

def create_cytogenetic_features(clinical_df):
    """Extract cytogenetic features from clinical data"""
    cyto_features = pd.DataFrame(index=clinical_df['ID'])
    cyto_col = clinical_df.set_index('ID')['CYTOGENETICS'].fillna('')
    
    # Counts
    cyto_features['cyto_del_count'] = cyto_col.str.count(r'del\(')
    cyto_features['cyto_has_del'] = (cyto_features['cyto_del_count'] > 0).astype(int)
    cyto_features['cyto_transloc_count'] = cyto_col.str.count(r't\(')
    cyto_features['cyto_has_transloc'] = (cyto_features['cyto_transloc_count'] > 0).astype(int)
    cyto_features['cyto_inv_count'] = cyto_col.str.count(r'inv\(')
    cyto_features['cyto_has_inv'] = (cyto_features['cyto_inv_count'] > 0).astype(int)
    cyto_features['cyto_gain_count'] = cyto_col.str.count(r'\+')
    cyto_features['cyto_has_gain'] = (cyto_features['cyto_gain_count'] > 0).astype(int)
    cyto_features['cyto_loss_count'] = cyto_col.str.count(r'-[0-9XY]')
    cyto_features['cyto_has_loss'] = (cyto_features['cyto_loss_count'] > 0).astype(int)
    cyto_features['cyto_other_count'] = cyto_col.str.count(r'add\(|ins\(|dup\(')
    
    # Total anomalies
    cyto_features['cyto_total_anomalies'] = (
        cyto_features['cyto_del_count'] + cyto_features['cyto_transloc_count'] + 
        cyto_features['cyto_inv_count'] + cyto_features['cyto_gain_count'] + 
        cyto_features['cyto_loss_count'] + cyto_features['cyto_other_count']
    )
    
    # Normal and complex
    cyto_features['cyto_normal'] = cyto_col.str.match(r'^46,(xx|xy)(\[\d+\])?$', case=False).astype(int)
    cyto_features['cyto_complex'] = (
        (cyto_features['cyto_total_anomalies'] >= 3) | 
        cyto_col.str.contains('complex', case=False, na=False)
    ).astype(int)
    
    # Chromosome-specific
    chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
    for chrom in chromosomes:
        pattern = rf'(\b|[,\(]){chrom}([,;:\)\[]|[pq])'
        cyto_features[f'cyto_chr{chrom}_affected'] = cyto_col.str.contains(
            pattern, case=False, na=False, regex=True
        ).astype(int)
    
    # Specific abnormalities
    cyto_features['cyto_monosomy7'] = cyto_col.str.contains(r'-7[^0-9]|^45.*-7', case=False, na=False, regex=True).astype(int)
    cyto_features['cyto_trisomy8'] = cyto_col.str.contains(r'\+8[^0-9]|^47.*\+8', case=False, na=False, regex=True).astype(int)
    cyto_features['cyto_del5q'] = cyto_col.str.contains(r'del\(5\)\(q', case=False, na=False, regex=True).astype(int)
    cyto_features['cyto_del20q'] = cyto_col.str.contains(r'del\(20\)\(q', case=False, na=False, regex=True).astype(int)
    cyto_features['cyto_chr3_abnormal'] = cyto_col.str.contains(r'(del|t|inv)\(3[;,:\)]', case=False, na=False, regex=True).astype(int)
    cyto_features['cyto_chr7_abnormal'] = cyto_col.str.contains(r'(del|t|inv)\(7[;,:\)]', case=False, na=False, regex=True).astype(int)
    
    return cyto_features.fillna(0)


def create_molecular_features(molecular_df, patient_ids, top_n_genes=20):
    """Extract molecular features from mutation data"""
    mol_features = pd.DataFrame({'ID': patient_ids})
    
    # Mutation counts
    mutation_counts = molecular_df.groupby('ID').size().to_frame('mutation_count_total')
    mol_features = mol_features.merge(mutation_counts, on='ID', how='left')
    
    # VAF statistics
    vaf_stats = molecular_df.groupby('ID')['VAF'].agg([
        ('vaf_mean', 'mean'), ('vaf_max', 'max'), ('vaf_sum', 'sum')
    ]).reset_index()
    mol_features = mol_features.merge(vaf_stats, on='ID', how='left')
    
    # Effect counts
    effect_counts = molecular_df.groupby(['ID', 'EFFECT']).size().unstack(fill_value=0)
    effect_counts.columns = [f'effect_{col}' for col in effect_counts.columns]
    mol_features = mol_features.merge(effect_counts.reset_index(), on='ID', how='left')
    
    # Gene features
    top_genes_list = molecular_df['GENE'].value_counts().head(top_n_genes).index.tolist()
    for gene in top_genes_list:
        gene_mutations = molecular_df[molecular_df['GENE'] == gene].groupby('ID').size()
        mol_features[f'gene_{gene}_count'] = mol_features['ID'].map(gene_mutations)
        gene_present = molecular_df[molecular_df['GENE'] == gene]['ID'].unique()
        mol_features[f'gene_{gene}_present'] = mol_features['ID'].isin(gene_present).astype(int)
    
    # Fill NaN
    feature_cols = [col for col in mol_features.columns if col != 'ID']
    mol_features[feature_cols] = mol_features[feature_cols].fillna(0)
    
    return mol_features.set_index('ID')


def create_advanced_features(X_df):
    """Create advanced interaction and transformation features"""
    X_adv = X_df.copy()
    
    # Ratios
    X_adv['blast_to_wbc'] = X_adv['BM_BLAST'] / (X_adv['WBC'] + 1)
    X_adv['monocyte_ratio'] = X_adv['MONOCYTES'] / (X_adv['WBC'] + 1)
    X_adv['anc_ratio'] = X_adv['ANC'] / (X_adv['WBC'] + 1)
    X_adv['platelet_to_blast'] = X_adv['PLT'] / (X_adv['BM_BLAST'] + 1)
    X_adv['hb_to_plt'] = X_adv['HB'] / (X_adv['PLT'] + 1)
    
    # VAF interactions
    X_adv['vaf_mutation_burden'] = X_adv['vaf_mean'] * X_adv['mutation_count_total']
    X_adv['vaf_per_mutation'] = X_adv['vaf_sum'] / (X_adv['mutation_count_total'] + 1)
    
    # Cytogenetic risk
    X_adv['cytogenetic_risk_score'] = (
        X_adv['cyto_complex'] * 3 + X_adv['cyto_chr7_affected'] * 2 + 
        X_adv['cyto_del5q'] * 1.5 + X_adv['cyto_loss_count'] * 0.5
    )
    X_adv['blast_cytogenetic_risk'] = X_adv['BM_BLAST'] * X_adv['cytogenetic_risk_score']
    
    # Clinical interactions
    X_adv['blast_to_mutation'] = X_adv['BM_BLAST'] / (X_adv['mutation_count_total'] + 1)
    X_adv['wbc_plt_index'] = X_adv['WBC'] * X_adv['PLT'] / 1000
    X_adv['blast_hb_ratio'] = X_adv['BM_BLAST'] / (X_adv['HB'] + 1)
    X_adv['monocyte_blast_ratio'] = X_adv['MONOCYTES'] / (X_adv['BM_BLAST'] + 1)
    
    # Mutation interactions
    X_adv['mutation_per_vaf'] = X_adv['mutation_count_total'] / (X_adv['vaf_mean'] + 0.01)
    X_adv['cyto_anomaly_density'] = X_adv['cyto_total_anomalies'] / (X_adv['cyto_total_anomalies'].max() + 1)
    X_adv['blast_mutation_interaction'] = X_adv['BM_BLAST'] * X_adv['mutation_count_total']
    X_adv['blast_vaf_interaction'] = X_adv['BM_BLAST'] * X_adv['vaf_mean']
    X_adv['blast_cyto_complex'] = X_adv['BM_BLAST'] * X_adv['cyto_complex']
    
    # Gene-specific interactions
    X_adv['tp53_blast'] = X_adv['gene_TP53_present'] * X_adv['BM_BLAST']
    X_adv['runx1_mutation_burden'] = X_adv['gene_RUNX1_count'] * X_adv['mutation_count_total']
    X_adv['nras_vaf'] = X_adv['gene_NRAS_present'] * X_adv['vaf_mean']
    X_adv['cyto_mutation_interaction'] = X_adv['cyto_total_anomalies'] * X_adv['mutation_count_total']
    X_adv['chr7_blast'] = X_adv['cyto_chr7_affected'] * X_adv['BM_BLAST']
    X_adv['vaf_cyto_burden'] = X_adv['vaf_sum'] * X_adv['cyto_total_anomalies']
    X_adv['vaf_tp53'] = X_adv['vaf_mean'] * X_adv['gene_TP53_present']
    
    # Log transformations
    X_adv['log_wbc'] = np.log1p(X_adv['WBC'])
    X_adv['log_plt'] = np.log1p(X_adv['PLT'])
    X_adv['log_blast'] = np.log1p(X_adv['BM_BLAST'])
    X_adv['log_mutation_count'] = np.log1p(X_adv['mutation_count_total'])
    X_adv['log_vaf_sum'] = np.log1p(X_adv['vaf_sum'])
    
    return X_adv


print("=" * 80)
print("LOADING AND PREPARING DATA")
print("=" * 80)

# Load data
clinical_train = pd.read_csv(f"{DATA_PATH}\\X_train\\clinical_train.csv")
target_train = pd.read_csv(f"{DATA_PATH}\\target_train.csv")
clinical_test = pd.read_csv(f"{DATA_PATH}\\X_test\\clinical_test.csv")
molecular_train = pd.read_csv(f"{DATA_PATH}\\X_train\\molecular_train.csv")
molecular_test = pd.read_csv(f"{DATA_PATH}\\X_test\\molecular_test.csv")

print("✓ Data loaded")

# Create features
cyto_features_train = create_cytogenetic_features(clinical_train)
train_patient_ids = clinical_train['ID'].unique()
mol_features_train = create_molecular_features(molecular_train, train_patient_ids)

# Clean target
target_clean = target_train.dropna(subset=['OS_YEARS', 'OS_STATUS']).copy()
target_clean['OS_STATUS'] = target_clean['OS_STATUS'].astype(bool)
target_clean = target_clean.set_index('ID')

# Clean clinical
clinical_train_clean = clinical_train[clinical_train['ID'].isin(target_clean.index)].copy()
clinical_train_clean = clinical_train_clean.set_index('ID').loc[target_clean.index]

# Build feature matrix
numeric_features = ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT']
X_numeric = clinical_train_clean[numeric_features].copy()
center_encoded = pd.get_dummies(clinical_train_clean['CENTER'], prefix='CENTER', drop_first=True)
X_clinical = pd.concat([X_numeric, center_encoded], axis=1)

mol_features_train_aligned = mol_features_train.reindex(X_clinical.index, fill_value=0)
cyto_features_train_aligned = cyto_features_train.reindex(X_clinical.index, fill_value=0)
X_all_base = pd.concat([X_clinical, mol_features_train_aligned, cyto_features_train_aligned], axis=1)
X_all = create_advanced_features(X_all_base)

# Create survival target
y_surv = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_clean)

print(f"✓ Features created: {X_all.shape[1]} total")
print(f"  Samples: {X_all.shape[0]}")
print()

# ============================================================================
# CROSS-VALIDATION STACKING FRAMEWORK
# ============================================================================

print("=" * 80)
print(f"CROSS-VALIDATION STACKING ({N_FOLDS} FOLDS)")
print("=" * 80)
print()

# Initialize storage for out-of-fold predictions
oof_predictions = pd.DataFrame(index=X_all.index)
oof_predictions['rsf_pred'] = 0.0
oof_predictions['xgb_pred'] = 0.0
oof_predictions['cox_pred'] = 0.0

# Stratified K-Fold
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, target_clean['OS_STATUS'].astype(int)), 1):
    print(f"Fold {fold_idx}/{N_FOLDS}")
    print("-" * 40)
    
    # Split data
    X_train_fold = X_all.iloc[train_idx]
    X_val_fold = X_all.iloc[val_idx]
    y_train_fold = y_surv[train_idx]
    y_val_fold = y_surv[val_idx]
    
    # ========== RSF ==========
    print("  Training RSF...", end=" ")
    available_rsf = [f for f in RSF_FEATURES if f in X_all.columns]
    X_rsf_train = X_train_fold[available_rsf]
    X_rsf_val = X_val_fold[available_rsf]
    
    imputer_rsf = SimpleImputer(strategy='median')
    X_rsf_train_imp = imputer_rsf.fit_transform(X_rsf_train)
    X_rsf_val_imp = imputer_rsf.transform(X_rsf_val)
    
    rsf = RandomSurvivalForest(**RSF_PARAMS)
    rsf.fit(X_rsf_train_imp, y_train_fold)
    rsf_pred = rsf.predict(X_rsf_val_imp)
    
    oof_predictions.loc[X_val_fold.index, 'rsf_pred'] = rsf_pred
    
    c_rsf = concordance_index_censored(
        y_val_fold['OS_STATUS'], y_val_fold['OS_YEARS'], rsf_pred
    )[0]
    print(f"C-index: {c_rsf:.4f}")
    
    # ========== XGBoost ==========
    print("  Training XGBoost...", end=" ")
    imputer_xgb = SimpleImputer(strategy='median')
    X_xgb_train_imp = imputer_xgb.fit_transform(X_train_fold)
    X_xgb_val_imp = imputer_xgb.transform(X_val_fold)
    
    # Create XGBoost labels
    y_train_xgb = y_train_fold['OS_YEARS'].copy()
    y_train_xgb[~y_train_fold['OS_STATUS']] = -y_train_xgb[~y_train_fold['OS_STATUS']]
    
    dtrain = xgb.DMatrix(X_xgb_train_imp, label=y_train_xgb)
    dval = xgb.DMatrix(X_xgb_val_imp)
    
    xgb_model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=100, verbose_eval=False)
    xgb_pred = xgb_model.predict(dval)
    
    oof_predictions.loc[X_val_fold.index, 'xgb_pred'] = xgb_pred
    
    c_xgb = concordance_index_censored(
        y_val_fold['OS_STATUS'], y_val_fold['OS_YEARS'], xgb_pred
    )[0]
    print(f"C-index: {c_xgb:.4f}")
    
    # ========== CoxPH ==========
    print("  Training CoxPH...", end=" ")
    available_coxph = [f for f in COXPH_FEATURES if f in X_all.columns]
    X_cox_train = X_train_fold[available_coxph]
    X_cox_val = X_val_fold[available_coxph]
    
    imputer_cox = SimpleImputer(strategy='median')
    X_cox_train_imp = imputer_cox.fit_transform(X_cox_train)
    X_cox_val_imp = imputer_cox.transform(X_cox_val)
    
    # Variance filter
    selector = VarianceThreshold(threshold=0.01)
    X_cox_train_var = selector.fit_transform(X_cox_train_imp)
    X_cox_val_var = selector.transform(X_cox_val_imp)
    
    # Standardize
    scaler = StandardScaler()
    X_cox_train_scaled = scaler.fit_transform(X_cox_train_var)
    X_cox_val_scaled = scaler.transform(X_cox_val_var)
    
    cox_model = CoxnetSurvivalAnalysis(**COXPH_PARAMS)
    cox_model.fit(X_cox_train_scaled, y_train_fold)
    cox_pred = cox_model.predict(X_cox_val_scaled)
    
    oof_predictions.loc[X_val_fold.index, 'cox_pred'] = cox_pred
    
    c_cox = concordance_index_censored(
        y_val_fold['OS_STATUS'], y_val_fold['OS_YEARS'], cox_pred
    )[0]
    print(f"C-index: {c_cox:.4f}")
    
    # Average ensemble for this fold
    avg_pred = (rsf_pred + xgb_pred + cox_pred) / 3
    c_avg = concordance_index_censored(
        y_val_fold['OS_STATUS'], y_val_fold['OS_YEARS'], avg_pred
    )[0]
    
    fold_scores.append({
        'fold': fold_idx,
        'rsf': c_rsf,
        'xgb': c_xgb,
        'cox': c_cox,
        'avg': c_avg
    })
    
    print(f"  → Average ensemble: {c_avg:.4f}")
    print()

print("=" * 80)
print("CROSS-VALIDATION RESULTS")
print("=" * 80)
print()

scores_df = pd.DataFrame(fold_scores)
print(scores_df.to_string(index=False))
print()
print(f"Mean RSF:     {scores_df['rsf'].mean():.4f} ± {scores_df['rsf'].std():.4f}")
print(f"Mean XGBoost: {scores_df['xgb'].mean():.4f} ± {scores_df['xgb'].std():.4f}")
print(f"Mean CoxPH:   {scores_df['cox'].mean():.4f} ± {scores_df['cox'].std():.4f}")
print(f"Mean Average: {scores_df['avg'].mean():.4f} ± {scores_df['avg'].std():.4f}")
print()

# ============================================================================
# CREATE META-FEATURES
# ============================================================================

print("=" * 80)
print("CREATING META-FEATURES")
print("=" * 80)
print()

X_meta = pd.DataFrame(index=X_all.index)

# Base predictions
X_meta['rsf_pred'] = oof_predictions['rsf_pred']
X_meta['xgb_pred'] = oof_predictions['xgb_pred']
X_meta['cox_pred'] = oof_predictions['cox_pred']

# Prediction statistics
X_meta['pred_mean'] = oof_predictions[['rsf_pred', 'xgb_pred', 'cox_pred']].mean(axis=1)
X_meta['pred_std'] = oof_predictions[['rsf_pred', 'xgb_pred', 'cox_pred']].std(axis=1)
X_meta['pred_min'] = oof_predictions[['rsf_pred', 'xgb_pred', 'cox_pred']].min(axis=1)
X_meta['pred_max'] = oof_predictions[['rsf_pred', 'xgb_pred', 'cox_pred']].max(axis=1)

# Pairwise differences
X_meta['rsf_xgb_diff'] = oof_predictions['rsf_pred'] - oof_predictions['xgb_pred']
X_meta['rsf_cox_diff'] = oof_predictions['rsf_pred'] - oof_predictions['cox_pred']
X_meta['xgb_cox_diff'] = oof_predictions['xgb_pred'] - oof_predictions['cox_pred']

# Pairwise products
X_meta['rsf_xgb_prod'] = oof_predictions['rsf_pred'] * oof_predictions['xgb_pred']
X_meta['rsf_cox_prod'] = oof_predictions['rsf_pred'] * oof_predictions['cox_pred']
X_meta['xgb_cox_prod'] = oof_predictions['xgb_pred'] * oof_predictions['cox_pred']

# Agreement features
X_meta['pred_variance'] = oof_predictions[['rsf_pred', 'xgb_pred', 'cox_pred']].var(axis=1)
X_meta['pred_agreement'] = 1 / (X_meta['pred_variance'] + 0.01)

# Risk stratification
percentile_75 = X_meta['pred_mean'].quantile(0.75)
percentile_25 = X_meta['pred_mean'].quantile(0.25)
X_meta['high_risk'] = (X_meta['pred_max'] > percentile_75).astype(int)
X_meta['low_risk'] = (X_meta['pred_min'] < percentile_25).astype(int)
X_meta['disagreement'] = (X_meta['pred_std'] > X_meta['pred_std'].median()).astype(int)

# Add top original features
top_features = [
    'BM_BLAST', 'HB', 'PLT', 'WBC', 
    'gene_TP53_present', 'gene_RUNX1_present', 'gene_ASXL1_present',
    'cyto_complex', 'cyto_normal', 'cyto_chr7_affected',
    'vaf_sum', 'mutation_count_total',
    'log_mutation_count', 'log_blast', 'vaf_tp53'
]

for feat in top_features:
    if feat in X_all.columns:
        X_meta[feat] = X_all[feat]

print(f"✓ Meta-features created: {X_meta.shape[1]} features")
print(f"  - Base predictions: 3")
print(f"  - Statistics: 4")
print(f"  - Differences: 3")
print(f"  - Products: 3")
print(f"  - Agreement: 2")
print(f"  - Risk flags: 3")
print(f"  - Original features: {len([f for f in top_features if f in X_all.columns])}")
print()

# ============================================================================
# TRAIN META-MODEL
# ============================================================================

print("=" * 80)
print("TRAINING META-MODEL")
print("=" * 80)
print()

# Prepare meta-features
imputer_meta = SimpleImputer(strategy='median')
X_meta_imputed = pd.DataFrame(
    imputer_meta.fit_transform(X_meta),
    index=X_meta.index,
    columns=X_meta.columns
)

# Standardize
scaler_meta = StandardScaler()
X_meta_scaled = pd.DataFrame(
    scaler_meta.fit_transform(X_meta_imputed),
    index=X_meta_imputed.index,
    columns=X_meta_imputed.columns
)

# Split for meta-model validation
from sklearn.model_selection import train_test_split
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
    X_meta_scaled, y_surv, test_size=0.3, random_state=42,
    stratify=target_clean['OS_STATUS'].astype(int)
)

print(f"Meta-model split: {len(X_meta_train)} train, {len(X_meta_val)} val")
print()

# Optimize meta-model
print("Optimizing meta-model hyperparameters (100 trials)...")

def meta_objective(trial):
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
    meta_model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, fit_baseline_model=True)
    meta_model.fit(X_meta_train, y_meta_train)
    
    y_pred_meta = meta_model.predict(X_meta_val)
    c_index = concordance_index_censored(
        y_meta_val['OS_STATUS'], y_meta_val['OS_YEARS'], y_pred_meta
    )[0]
    
    return c_index

study = optuna.create_study(direction='maximize')
study.optimize(meta_objective, n_trials=100, show_progress_bar=False)

best_l1_ratio = study.best_params['l1_ratio']
best_meta_c_index = study.best_value

print(f"✓ Best l1_ratio: {best_l1_ratio:.4f}")
print(f"✓ Meta-model C-index: {best_meta_c_index:.4f}")
print()

# Train final meta-model on all data
print("Training final meta-model on full training data...")
final_meta_model = CoxnetSurvivalAnalysis(l1_ratio=best_l1_ratio, fit_baseline_model=True)
final_meta_model.fit(X_meta_scaled, y_surv)

# Validate on full dataset
y_pred_final = final_meta_model.predict(X_meta_scaled)
c_index_full = concordance_index_censored(
    target_clean['OS_STATUS'], target_clean['OS_YEARS'], y_pred_final
)[0]

print(f"✓ Full training C-index: {c_index_full:.4f}")
print()

# ============================================================================
# COMPARISON WITH PREVIOUS VERSIONS
# ============================================================================

print("=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print()

comparison = pd.DataFrame([
    {'Version': 'V4 RSF', 'C-index': 0.7404, 'Method': 'Random Survival Forest'},
    {'Version': 'V6.1 XGBoost', 'C-index': 0.7413, 'Method': 'XGBoost Survival'},
    {'Version': 'V9 CoxPH', 'C-index': 0.7371, 'Method': 'Cox Proportional Hazards'},
    {'Version': 'V10 Ensemble', 'C-index': 0.7430, 'Method': 'Weighted average (3 models)'},
    {'Version': 'V11 Meta-Learning (CV)', 'C-index': scores_df['avg'].mean(), 'Method': 'Cross-validation average'},
    {'Version': 'V11 Meta-Learning (Final)', 'C-index': best_meta_c_index, 'Method': 'Stacked meta-model'},
])

print(comparison.to_string(index=False))
print()

improvement_v10 = best_meta_c_index - 0.7430
improvement_best = best_meta_c_index - max(0.7404, 0.7413, 0.7430)

print(f"📊 Improvements:")
print(f"  vs V10 Ensemble:   {improvement_v10:+.4f}")
print(f"  vs Best previous:  {improvement_best:+.4f}")
print()

print("=" * 80)
print("✓ META-LEARNING TRAINING COMPLETE")
print("=" * 80)
print()
print("Next steps:")
print("1. Generate test set predictions")
print("2. Create submission file")
print("3. Evaluate on leaderboard")
