import pandas as pd
from config import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis


def have_p_factor(p_factor_para=p_factor_vars):

    data = pd.read_csv('data/data_miced.csv')

    scaler = StandardScaler()
    data[p_factor_para] = scaler.fit_transform(data[p_factor_para])

    fa = FactorAnalysis(n_components=1, random_state=42)
    data['p-factor'] = fa.fit_transform(data[p_factor_para])
    data.drop(p_factor_para, axis=1, inplace=True)
    data = data.round(10)

    data.sort_values(by='p-factor', ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)

    data.rename(columns={'fam_history_q6a_depression': 'biological_father',
                         'fam_history_q6e_depression': 'maternal_grandfather',
                         'fam_history_q6f_depression': 'maternal_grandmother',
                         'fam_history_q6d_depression': 'biological_mother',
                         'fam_history_q6b_depression': 'paternal_grandfather',
                         'fam_history_q6c_depression': 'paternal_grandmother'}, inplace=True)

    data.to_csv('data/data_p_factor.csv', index=False)


