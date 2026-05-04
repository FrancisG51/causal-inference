import pandas as pd
import numpy as np


# 匹配后的各个协变量SMD值
def matched_standardized_bias(matched_id, matched_score, inter_char):
    '''
    matched_id     是匹配后的id表格
    inter_char     是干预变量的变量名称
    matched_score  是psm.predicted_data
    '''

    inter_group = matched_id.iloc[:, 0].to_list()
    matched_score_inter = matched_score.loc[matched_score['src_subject_id'].isin(inter_group), :].copy()
    matched_score_inter.reset_index(inplace=True, drop=False)

    control_group = matched_id.iloc[:, 1:].stack().dropna().to_list()
    matched_score_control = matched_score.loc[matched_score['src_subject_id'].isin(control_group), :].copy()
    matched_score_control.reset_index(inplace=True, drop=False)

    # 样本量决定了贡献标准差的权重
    num_inter = matched_score_inter.shape[0]
    num_control = matched_score_control.shape[0]

    # 选出来协变量都有哪些
    covariates = matched_score.columns.to_list()
    not_covariates = ['src_subject_id', 'propensity_score', 'propensity_logit', inter_char]
    covariates = [col for col in covariates if col not in not_covariates]

    result = []
    for var in covariates:
        mean_inter = matched_score_inter[var].mean()
        mean_control = matched_score_control[var].mean()
        std_inter = matched_score_inter[var].std(ddof=1)
        std_control = matched_score_control[var].std(ddof=1)

        # 合并标准差，标准化偏差
        fenmu = (num_control + num_inter - 2)
        fenzi = (num_inter - 1) * std_inter ** 2 + (num_control - 1) * std_control ** 2
        pooled_std = np.sqrt(fenzi / fenmu)
        sb = round(100 * abs(mean_inter - mean_control) / pooled_std, 1)

        result.append({'covariate': var, 'SMD': sb})

    sb_result = pd.DataFrame(result).reset_index(drop=True)
    sb_result['是否通过SMD检验'] = np.where(sb_result['SMD'] <= 10, '通过', 'no')

    sb_result.sort_values(by='SMD', ascending=False, inplace=True)

    return sb_result


# 匹配前的SMD值
def unmatched_sb_value(matched_score, inter_char):
    '''
    matched_score 倾向得分的值
    inter_char    干预变量名称
    '''
    inter_score = matched_score.loc[matched_score[inter_char] == 1, :].reset_index(drop=True)
    control_score = matched_score.loc[matched_score[inter_char] == 0, :].reset_index(drop=True)

    covariates = matched_score.columns.to_list()
    not_covariates = ['src_subject_id', 'propensity_score', 'propensity_logit', inter_char]
    covariates = [col for col in covariates if col not in not_covariates]

    num_inter = inter_score.shape[0]
    num_control = control_score.shape[0]

    result = []

    for var in covariates:
        mean_inter = inter_score[var].mean()
        mean_control = control_score[var].mean()
        std_inter = inter_score[var].std(ddof=1)
        std_control = control_score[var].std(ddof=1)

        # 合并标准差，标准化偏差
        fenmu = (num_control + num_inter - 2)
        fenzi = (num_inter - 1) * std_inter ** 2 + (num_control - 1) * std_control ** 2
        pooled_std = np.sqrt(fenzi / fenmu)
        sb = round(100 * abs(mean_inter - mean_control) / pooled_std, 1)

        result.append({'covariate': var, 'SMD_unmatched': sb})

    sb_result = pd.DataFrame(result).reset_index(drop=True)

    return sb_result

