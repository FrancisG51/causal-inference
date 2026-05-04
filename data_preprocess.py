import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def process_row_data():

    data_demo = pd.read_csv('row_data/abcd_p_demo.csv')
    data_cbcl = pd.read_csv('row_data/mh_p_cbcl.csv')
    data_inter = pd.read_csv('row_data/mh_p_fhx.csv')

    # 人口统计变量
    data_demo = data_demo[['src_subject_id', 'eventname',
                           'demo_prnt_ed_v2', 'demo_brthdat_v2', 'demo_comb_income_v2',
                           'demo_race_a_p___10', 'demo_race_a_p___11', 'demo_sex_v2']]
    data_demo['demo_sex_v2'] = np.where(data_demo['demo_sex_v2'] == 1, 1, 0)

    # 青少年行为检查表
    data_cbcl = data_cbcl[['src_subject_id', 'eventname',
                           'cbcl_scr_syn_aggressive_t', 'cbcl_scr_syn_anxdep_t', 'cbcl_scr_syn_attention_t',
                           'cbcl_scr_syn_rulebreak_t', 'cbcl_scr_syn_social_t', 'cbcl_scr_syn_somatic_t',
                           'cbcl_scr_syn_thought_t', 'cbcl_scr_syn_withdep_t',
                           'cbcl_scr_syn_totprob_t', 'cbcl_scr_dsm5_depress_t']]

    # 各个干预变量
    data_inter = data_inter[['src_subject_id', 'eventname',
                             'fam_history_6_yes_no',
                             'fam_history_q6a_depression',
                             'fam_history_q6e_depression', 'fam_history_q6f_depression',
                             'fam_history_q6d_depression',
                             'fam_history_q6b_depression', 'fam_history_q6c_depression']]
    not_imputation = ['src_subject_id', 'eventname', 'fam_history_6_yes_no']
    need_imputation = [col for col in data_inter.columns.to_list() if col not in not_imputation]
    for i in range(data_inter.shape[0]):
        if data_inter.loc[i, 'fam_history_6_yes_no'] == 0:
            data_inter.loc[i, need_imputation] = 0
    data_inter.drop('fam_history_6_yes_no', axis=1, inplace=True)

    data_all_row = pd.merge(data_demo, data_cbcl, on=['src_subject_id', 'eventname'], how='outer')
    data_all_row = pd.merge(data_all_row, data_inter, on=['src_subject_id', 'eventname'], how='outer')
    data_all_row = data_all_row.loc[data_all_row['eventname'] == 'baseline_year_1_arm_1']
    data_all_row.drop('eventname', axis=1, inplace=True)

    data_all_row.replace([999, 888, 777, 666, 555, 444, 333, 222], np.nan, inplace=True)
    data_all_row = data_all_row.dropna(thresh=int(0.99999 * data_all_row.shape[1]), axis=0)

    lie_num_1 = data_all_row.shape[1]

    # 检查是否存在一列全是单一值的情况
    unique_counts = data_all_row.nunique()
    columns_keep = unique_counts[unique_counts >= 2].index
    data_all_row = data_all_row.loc[:, columns_keep]
    data_all_row.reset_index(drop=True, inplace=True)

    lie_num_2 = data_all_row.shape[1]
    if lie_num_1 != lie_num_2:
        print('存在列被移除的情况')

    columns = data_all_row.columns.to_list()
    if len(set(columns)) != len(columns):
        print('存在重复的列')

    data_all_row.to_csv('data/data_to_mice.csv', index=False)

    data1 = pd.read_csv('data/data_miced.csv')
    data2 = pd.read_csv('data/data_to_mice.csv')
    if data1.shape != data2.shape:
        print('出了差错')
        print(data1.shape)
        print(data2.shape)

