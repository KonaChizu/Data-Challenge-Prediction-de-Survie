# =============================================================================
# SOUMISSION FINALE - Gradient Boosting (meilleur modèle)
# =============================================================================


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv

# --- 1. CHARGEMENT DES DONNÉES ---
clinical_train = pd.read_csv('raw_data/X_train/clinical_train.csv')
clinical_test = pd.read_csv('raw_data/X_test/clinical_test.csv')
molecular_train = pd.read_csv('raw_data/X_train/molecular_train.csv')
molecular_test = pd.read_csv('raw_data/X_test/molecular_test.csv')
target_train = pd.read_csv('raw_data/target_train.csv')

# --- 2. FEATURE ENGINEERING ---
def build_features(clinical_df, molecular_df, is_train=True, median_variants=None):
    """Construit les 37 features pour train ou test"""
    
    # Transformations log cliniques
    clinical_eng = clinical_df.copy()
    for col in ['BM_BLAST', 'ANC', 'HB', 'PLT']:
        clinical_eng[f'{col}_log'] = np.log1p(clinical_df[col])
    
    # Features cliniques avancées
    clinical_eng['severity_score'] = (
        clinical_eng['BM_BLAST_log'] * 2 +
        (1 / (clinical_eng['HB_log'] + 0.1)) +
        (1 / (clinical_eng['PLT_log'] + 0.1))
    )
    clinical_eng['HB_PLT_ratio'] = np.log1p(clinical_df['HB'] / (clinical_df['PLT'] + 1))
    clinical_eng['BLAST_ANC_ratio'] = np.log1p(clinical_df['BM_BLAST'] / (clinical_df['ANC'] + 0.1))
    clinical_eng['HB_low'] = (clinical_df['HB'] < 10).astype(int)
    clinical_eng['BLAST_high'] = (clinical_df['BM_BLAST'] > 20).astype(int)
    
    # Aggregation moléculaire
    molecular_agg = molecular_df.groupby('ID').agg({
        'CHR': 'count', 'VAF': ['mean', 'max']
    }).reset_index()
    molecular_agg.columns = ['ID', 'total_variants', 'mean_VAF', 'max_VAF']
    
    pertinent_effects = ['non_synonymous_codon', 'frameshift_variant', 'splice_site_variant', 'stop_gained']
    effect_counts = molecular_df[molecular_df['EFFECT'].isin(pertinent_effects)].pivot_table(
        index='ID', columns='EFFECT', aggfunc='size', fill_value=0
    )
    effect_counts.columns = [f'count_{col}' for col in effect_counts.columns]
    molecular_agg = molecular_agg.merge(effect_counts, left_on='ID', right_index=True, how='left').fillna(0)
    for effect in pertinent_effects:
        if f'count_{effect}' not in molecular_agg.columns:
            molecular_agg[f'count_{effect}'] = 0
    
    # Transformations log moléculaires
    molecular_eng = molecular_agg.copy()
    for col in ['total_variants', 'max_VAF', 'count_non_synonymous_codon']:
        molecular_eng[f'{col}_log'] = np.log1p(molecular_agg[col])
    
    # Features moléculaires avancées
    molecular_eng['weighted_mutation_score'] = (
        molecular_eng['total_variants_log'] +
        molecular_eng['count_non_synonymous_codon_log'] * 1.5 +
        np.log1p(molecular_agg['count_frameshift_variant']) * 2 +
        np.log1p(molecular_agg['count_stop_gained']) * 2.5
    )
    molecular_eng['VAF_ratio'] = np.log1p(molecular_agg['max_VAF'] / (molecular_agg['mean_VAF'] + 0.01))
    
    if is_train:
        median_variants = molecular_agg['total_variants'].median()
    molecular_eng['high_mutation_burden'] = (molecular_agg['total_variants'] > median_variants).astype(int)
    molecular_eng['mutation_diversity'] = (
        (molecular_agg['count_non_synonymous_codon'] > 0).astype(int) +
        (molecular_agg['count_frameshift_variant'] > 0).astype(int) +
        (molecular_agg['count_splice_site_variant'] > 0).astype(int) +
        (molecular_agg['count_stop_gained'] > 0).astype(int)
    )
    
    # Features cytogénétiques
    cyto = clinical_df['CYTOGENETICS'].fillna('MISSING')
    cyto_features = pd.DataFrame({'ID': clinical_df['ID']})
    cyto_features['is_normal_karyotype'] = cyto.str.match(r'^46,(xx|xy)(\[\d+\])?$', case=False).astype(int)
    cyto_features['has_monosomy_7'] = cyto.str.contains('-7', case=False, na=False).astype(int)
    cyto_features['has_del5q'] = cyto.str.contains(r'del.5', case=False, na=False).astype(int)
    cyto_features['has_del7q'] = cyto.str.contains(r'del.7', case=False, na=False).astype(int)
    cyto_features['has_trisomy_8'] = cyto.str.contains(r'\+8', case=False, na=False).astype(int)
    cyto_features['has_del20q'] = cyto.str.contains(r'del.20', case=False, na=False).astype(int)
    cyto_features['is_complex_karyotype'] = cyto.str.count('/').ge(2).astype(int)
    cyto_features['cyto_risk_score'] = 1
    cyto_features.loc[cyto_features['is_normal_karyotype'] == 1, 'cyto_risk_score'] = 0
    cyto_features.loc[cyto_features['has_del7q'] == 1, 'cyto_risk_score'] = 2
    cyto_features.loc[cyto_features['has_monosomy_7'] == 1, 'cyto_risk_score'] = 3
    cyto_features.loc[cyto_features['is_complex_karyotype'] == 1, 'cyto_risk_score'] = 4
    
    # Features génétiques
    gene_groups = {
        'high_risk': ['TP53', 'ASXL1', 'RUNX1', 'EZH2'],
        'splicing': ['SF3B1', 'SRSF2', 'U2AF1', 'ZRSR2'],
        'epigenetic': ['TET2', 'DNMT3A', 'IDH1', 'IDH2', 'EZH2', 'ASXL1'],
        'signaling': ['NRAS', 'KRAS', 'FLT3', 'CBL', 'NF1']
    }
    key_genes = ['TP53', 'ASXL1', 'RUNX1', 'EZH2', 'SF3B1', 'SRSF2', 'TET2', 'DNMT3A']
    genes_by_patient = molecular_df.groupby('ID')['GENE'].apply(set).reset_index()
    genes_by_patient.columns = ['ID', 'gene_set']
    gene_features = pd.DataFrame({'ID': genes_by_patient['ID']})
    for gene in key_genes:
        gene_features[f'has_{gene}'] = genes_by_patient['gene_set'].apply(lambda x: int(gene in x))
    for group_name, genes in gene_groups.items():
        gene_features[f'n_{group_name}_genes'] = genes_by_patient['gene_set'].apply(lambda x: len(x.intersection(genes)))
    gene_features['n_unique_genes'] = genes_by_patient['gene_set'].apply(len)
    
    # Merge final
    clinical_cols = ['BM_BLAST_log', 'ANC_log', 'HB_log', 'PLT_log', 'severity_score', 
                     'HB_PLT_ratio', 'BLAST_ANC_ratio', 'HB_low', 'BLAST_high']
    molecular_cols = ['total_variants_log', 'max_VAF_log', 'count_non_synonymous_codon_log',
                      'weighted_mutation_score', 'VAF_ratio', 'high_mutation_burden', 'mutation_diversity']
    cyto_cols = ['is_normal_karyotype', 'has_monosomy_7', 'has_del5q', 'has_del7q',
                 'has_trisomy_8', 'has_del20q', 'is_complex_karyotype', 'cyto_risk_score']
    gene_cols = ['has_TP53', 'has_ASXL1', 'has_RUNX1', 'has_EZH2', 'has_SF3B1',
                 'has_SRSF2', 'has_TET2', 'has_DNMT3A', 'n_high_risk_genes', 
                 'n_splicing_genes', 'n_epigenetic_genes', 'n_signaling_genes', 'n_unique_genes']
    
    # Merge: garder TOUS les patients cliniques (how='left' pour moléculaire)
    final_df = clinical_eng[['ID'] + clinical_cols].merge(
        molecular_eng[['ID'] + molecular_cols], on='ID', how='left')
    final_df = final_df.merge(cyto_features[['ID'] + cyto_cols], on='ID', how='left')
    final_df = final_df.merge(gene_features[['ID'] + gene_cols], on='ID', how='left')
    
    # Remplir les NaN moléculaires avec 0 pour les patients sans données moléculaires
    for col in molecular_cols:
        final_df[col] = final_df[col].fillna(0)
    
    all_features = clinical_cols + molecular_cols + cyto_cols + gene_cols
    
    return final_df, all_features, median_variants

# --- 3. PRÉPARATION DES DONNÉES ---
train_df, all_features, median_v = build_features(clinical_train, molecular_train, is_train=True)
train_df = train_df.merge(target_train[['ID', 'OS_YEARS', 'OS_STATUS']], on='ID', how='inner')

# Nettoyage des valeurs manquantes dans la target
train_df = train_df.dropna(subset=['OS_YEARS', 'OS_STATUS'])

test_df, _, _ = build_features(clinical_test, molecular_test, is_train=False, median_variants=median_v)

# --- 4. VALIDATION CROISÉE POUR C-INDEX (sur données brutes, imputation par fold) ---
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_ipcw

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(train_df):
    # Split sur données BRUTES (avant imputation)
    tr_fold = train_df.iloc[train_idx].copy()
    val_fold = train_df.iloc[val_idx].copy()
    
    # Imputation sur fold
    knn_cv = KNNImputer(n_neighbors=5)
    tr_fold[all_features] = knn_cv.fit_transform(tr_fold[all_features])
    val_fold[all_features] = knn_cv.transform(val_fold[all_features])
    
    # Standardisation sur fold
    sc_cv = StandardScaler()
    X_tr = sc_cv.fit_transform(tr_fold[all_features])
    X_val = sc_cv.transform(val_fold[all_features])
    
    # Target
    y_tr = Surv.from_arrays(tr_fold['OS_STATUS'].astype(bool), tr_fold['OS_YEARS'])
    y_val = Surv.from_arrays(val_fold['OS_STATUS'].astype(bool), val_fold['OS_YEARS'])
    
    # Modèle
    gb_cv = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    gb_cv.fit(X_tr, y_tr)
    
    # C-index IPCW
    scores = gb_cv.predict(X_val)
    c_idx = concordance_index_ipcw(y_tr, y_val, scores, tau=7)[0]
    cv_scores.append(c_idx)

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"C-index IPCW (5-fold CV): {cv_mean:.4f} ± {cv_std:.4f}")

# --- 5. PRÉPARATION FINALE POUR SOUMISSION ---
# Imputation sur tout le train
knn = KNNImputer(n_neighbors=5)
train_df[all_features] = knn.fit_transform(train_df[all_features])
test_df[all_features] = knn.transform(test_df[all_features])

# Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[all_features])
X_test = scaler.transform(test_df[all_features])

# Target
y_train = Surv.from_arrays(train_df['OS_STATUS'].astype(bool), train_df['OS_YEARS'])

# --- 6. ENTRAÎNEMENT DU MODÈLE FINAL ---
gb_final = GradientBoostingSurvivalAnalysis(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=3, 
    random_state=42
)
gb_final.fit(X_train, y_train)

# --- 7. PRÉDICTION ET SOUMISSION ---
risk_scores = gb_final.predict(X_test)

# Transformer en scores positifs (shift + scale)
# Les risk scores GB peuvent être négatifs, on les transforme en rang percentile
from scipy.stats import rankdata
risk_scores_positive = rankdata(risk_scores) / len(risk_scores)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'risk_score': risk_scores_positive
})
submission.to_csv('output/submission_final.csv', index=False)

# --- 8. RÉSUMÉ ---
print("=" * 60)
print("🏆 SOUMISSION FINALE - GRADIENT BOOSTING")
print("=" * 60)
print(f"✓ Modèle: GradientBoostingSurvivalAnalysis")
print(f"✓ Hyperparamètres: n_estimators=200, lr=0.05, max_depth=3")
print(f"✓ Features: {len(all_features)}")
print(f"✓ C-index CV (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
print(f"\\n✓ Fichier: output/submission_final.csv")
print(f"✓ Prédictions: {len(submission)}")
print("=" * 60)
submission.head()