import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def match_operation(matched_score, match_ratio, caliper_value, inter_char, random):

    '''
    函数说明：
    matched_score是psmpy计算出的带有倾向得分的原始数据，其中倾向得分列名称为'propensity_score'
    match_ratio是匹配的比例，一个干预组匹配match_ratio数量的对照组
    caliper_value是卡钳值，在函数外面计算结合了倾向得分标准差的卡钳值，再传给函数
    inter_char是干预变量的变量名称
    random是随机种子，保证结果的可重复性

    本函数使用不放回匹配
    干预组应该是1，不干预是0
    '''

    matched_score_inter = matched_score.loc[matched_score[inter_char] == 1, :].copy()
    matched_score_inter.reset_index(inplace=True, drop=True)

    matched_score_control = matched_score.loc[matched_score[inter_char] == 0, :].copy()
    matched_score_control.reset_index(inplace=True, drop=True)

    # 建立一个新的dataframe最多一对四进行匹配，最终返回这个dataframe结果
    matched_id = pd.DataFrame({'src_subject_id': matched_score_inter['src_subject_id'],
                               'matched_id_1': np.nan, 'matched_id_2': np.nan,
                               'matched_id_3': np.nan, 'matched_id_4': np.nan})
    matched_id = matched_id.sample(frac=1, replace=False, axis=0, random_state=random).reset_index(drop=True)

    selected_control = []

    for i in range(matched_id.shape[0]):

        inter_id = matched_id.iloc[i, 0]
        inter_score = matched_score_inter.loc[matched_score_inter['src_subject_id'] == inter_id, 'propensity_score'].values[0]

        # 卡钳值筛选
        linshi = matched_score_control.copy()
        linshi['caliper_filter'] = np.abs(linshi['propensity_score'] - inter_score)
        linshi = linshi.loc[linshi['caliper_filter'] <= caliper_value, :].reset_index(drop=True)

        # 防止重复抽样筛选
        linshi = linshi.loc[~linshi['src_subject_id'].isin(selected_control)].reset_index(drop=True)

        if linshi.shape[0] >= match_ratio:

            # 如果满足匹配条件，先再次打乱顺序，再匹配
            linshi = linshi.sample(frac=1, replace=False, axis=0, random_state=random+1).reset_index(drop=True)

            # 开始进行k近邻匹配
            knn = NearestNeighbors(n_neighbors=match_ratio, metric='manhattan')
            fit_scores = linshi['propensity_score'].values.reshape(-1, 1)
            knn.fit(fit_scores)
            _, indices = knn.kneighbors([[inter_score]])
            control_id_list = linshi.iloc[indices[0], :]['src_subject_id'].tolist()

            selected_control.extend(control_id_list)

            # 获取的匹配的列表，开始为匹配id赋值
            for j in range(len(control_id_list)):
                matched_id.iloc[i, j + 1] = control_id_list[j]

    matched_id = matched_id.iloc[:, :match_ratio + 1]
    matched_id = matched_id.dropna(thresh=matched_id.shape[1], axis=0)
    matched_id.reset_index(drop=True, inplace=True)

    return matched_id
