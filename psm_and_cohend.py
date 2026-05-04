from config import *
from match_test import *
from sklearn.preprocessing import StandardScaler
from psmpy import PsmPy
from psm_matching import *
from calculate_cohend import *


def psm_and_cohend_function():

    data = pd.read_csv('data/data_p_factor.csv')

    # 卡钳半径，标准差的倍数
    caliper_rate = 0.01

    # 写一个函数来确定几比几匹配
    def get_match_ratio(dataframe, inter):
        one_num = dataframe[inter].sum()
        zero_num = dataframe.shape[0] - one_num
        ratio_num = int(zero_num/one_num)
        ratio_num = ratio_num - 1
        if ratio_num >= 4:
            ratio_num = 4
        return ratio_num

    for inter_var in inter_vars:

        print('****************************************************************')

        # 第一个临时表格，用于组成协变量，用于计算倾向得分以及用于匹配
        data_linshi_1 = copy.deepcopy(data)
        data_linshi_1_columns = copy.deepcopy(psm_covariates)
        data_linshi_1_columns.append('src_subject_id')
        data_linshi_1_columns.extend(inter_vars)
        data_linshi_1 = data_linshi_1[data_linshi_1_columns]

        # 第二个临时表格，用于获取因变量
        data_linshi_2 = copy.deepcopy(data)
        data_linshi_2_columns = copy.deepcopy(yinbianliang)
        data_linshi_2_columns.append('src_subject_id')
        data_linshi_2 = data_linshi_2[data_linshi_2_columns]

        # 因变量中再把那八个cbcl评分加进来
        data_miced = pd.read_csv('data/data_miced.csv')
        p_factor_vars_linshi = copy.deepcopy(p_factor_vars)
        p_factor_vars_linshi.append('src_subject_id')
        data_miced = data_miced[p_factor_vars_linshi]
        data_linshi_2 = pd.merge(data_linshi_2, data_miced, on='src_subject_id', how='inner')

        print('当前的干预变量为：', inter_var, '干预个体的数量为：', data_linshi_1[inter_var].sum())

        # 开始对于协变量进行标准化，用于计算倾向得分
        scaler = StandardScaler()
        data_linshi_1_columns.remove('src_subject_id')
        data_linshi_1_columns.remove(inter_var)
        data_linshi_1[data_linshi_1_columns] = scaler.fit_transform(data_linshi_1[data_linshi_1_columns]).round(10)

        # 获取倾向得分
        psm = PsmPy(data_linshi_1, treatment=inter_var, indx='src_subject_id', exclude=[])
        psm.logistic_ps(balance=True)
        matched_score = psm.predicted_data
        matched_score.reset_index(drop=True, inplace=True)
        caliper_value = round(caliper_rate * matched_score['propensity_score'].std(ddof=0), 8)

        # 计算匹配前的SMD值
        unmatched_smd = unmatched_sb_value(matched_score, inter_var)

        # 开始匹配
        print('匹配的比例：', get_match_ratio(dataframe=matched_score, inter=inter_var))
        matched_id = match_operation(matched_score=matched_score,
                                     match_ratio=get_match_ratio(dataframe=matched_score, inter=inter_var),
                                     caliper_value=caliper_value, inter_char=inter_var, random=42)
        matched_id.to_csv('data/matched_id_' + inter_var + '.csv', index=False)
        print('匹配后的干预组数量：', matched_id.shape[0])

        # 匹配后各个协变量SMD值
        matched_smd = matched_standardized_bias(matched_id, matched_score, inter_var)
        smd_dataframe = pd.merge(unmatched_smd, matched_smd, on='covariate', how='outer')
        smd_dataframe.sort_values(by='SMD', ascending=False, inplace=True)
        smd_dataframe.reset_index(drop=True, inplace=True)
        print(smd_dataframe)

        # 下面开始计算该干预变量的因变量的cohend效应量
        print('----------------------------------------------------------------')
        if set(smd_dataframe['是否通过SMD检验']) == {'通过'}:
            print(get_result_df(data_linshi_2, matched_id))
            get_result_df(data_linshi_2, matched_id).to_csv('data/cohenD_result_' + inter_var + '.csv', index=False)







