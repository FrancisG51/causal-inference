import copy
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


# 通过这个函数来获取当前自变量的三个因变量的效应量
def get_result_df(yinbianliang_df, matched_id):

    causal_inference_result = []

    inter_group = matched_id.iloc[:, 0].to_list()
    control_group = matched_id.iloc[:, 1:].stack().to_list()

    for yinbianliang in yinbianliang_df.columns.to_list():
        if yinbianliang != 'src_subject_id':

            yinbianliang_linshi_data = copy.deepcopy(yinbianliang_df[[yinbianliang, 'src_subject_id']])
            yinbianliang_linshi_data.reset_index(drop=True, inplace=True)

            yinbianliang_inter_data = yinbianliang_linshi_data.loc[yinbianliang_linshi_data['src_subject_id'].isin(inter_group), :]
            yinbianliang_control_data = yinbianliang_linshi_data.loc[yinbianliang_linshi_data['src_subject_id'].isin(control_group), :]

            yinbianliang_inter_data = yinbianliang_inter_data[yinbianliang]
            yinbianliang_control_data = yinbianliang_control_data[yinbianliang]

            # 先进行t检验
            _, p_value = ttest_ind(yinbianliang_inter_data, yinbianliang_control_data, equal_var=False)

            # 开始计算cohend效应量
            # 先取参数
            inter_num = yinbianliang_inter_data.shape[0]
            control_num = yinbianliang_control_data.shape[0]

            measures_inter_mean = yinbianliang_inter_data.mean()
            measures_control_mean = yinbianliang_control_data.mean()

            measures_inter_std = yinbianliang_inter_data.std(ddof=1)
            measures_control_std = yinbianliang_control_data.std(ddof=1)

            # 根据参数计算效应量
            fenmu = (control_num + inter_num - 2)
            fenzi = (inter_num - 1) * measures_inter_std ** 2 + (control_num - 1) * measures_control_std ** 2
            pooled_std = np.sqrt(fenzi / fenmu)
            cohend = round(((measures_inter_mean - measures_control_mean) / pooled_std) * 100, 2)

            causal_inference_result.append({'yinbianliang': yinbianliang,
                                            'cohend_value': cohend,
                                            'p_value': p_value,
                                            'ratio_num': int(control_num/inter_num)})

    causal_inference_result = pd.DataFrame(causal_inference_result)
    causal_inference_result.sort_values(by='cohend_value', ascending=False, inplace=True)
    causal_inference_result.reset_index(drop=True, inplace=True)

    return causal_inference_result
















