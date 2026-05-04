import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_visualize_function():

    father = pd.read_csv('data/cohenD_result_biological_father.csv')
    mother = pd.read_csv('data/cohenD_result_biological_mother.csv')

    father = father[['yinbianliang', 'cohend_value']]
    mother = mother[['yinbianliang', 'cohend_value']]

    father.rename(columns={'cohend_value': 'cohend_value_fa'}, inplace=True)
    mother.rename(columns={'cohend_value': 'cohend_value_mo'}, inplace=True)

    cohend_data = pd.merge(father, mother, how='outer', on='yinbianliang')
    cohend_data['mean'] = (cohend_data['cohend_value_mo'] + cohend_data['cohend_value_fa'])/2
    cohend_data.sort_values(by='mean', ascending=False, inplace=True)
    cohend_data.reset_index(drop=True, inplace=True)
    cohend_data.drop(columns='mean', inplace=True)

    cohend_data['yinbianliang'] = cohend_data['yinbianliang'].replace({
        'cbcl_scr_syn_totprob_t': 'total score',
        'p-factor': 'p-factor',
        'cbcl_scr_dsm5_depress_t': 'dsm5-depress',
        'cbcl_scr_syn_thought_t': 'thought problems',
        'cbcl_scr_syn_aggressive_t': 'aggressive behavior',
        'cbcl_scr_syn_social_t': 'social problems',
        'cbcl_scr_syn_anxdep_t': 'anxious/depressed',
        'cbcl_scr_syn_attention_t': 'attention problems',
        'cbcl_scr_syn_rulebreak_t': 'rule-breaking behavior',
        'cbcl_scr_syn_withdep_t': 'withdrawn/depressed',
        'cbcl_scr_syn_somatic_t': 'somatic complaints',
    })

    print(cohend_data)
    cohend_data.to_csv('data/cohend_data.csv', index=False)

    # 开始画图
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.subplots_adjust(bottom=0.25)
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.gca().set_facecolor('aliceblue')

    font_size = 12
    x_zuobiao = (np.arange(len(cohend_data['yinbianliang'])) * 3) + 1

    plt.title('父母抑郁史对于青少年结局影响效应量',
              fontsize=font_size+6, weight='normal')

    plt.xlim(x_zuobiao.min() - 2, x_zuobiao.max() + 2)
    plt.xticks([])
    plt.ylim(0, 55)
    plt.yticks([10, 20, 30, 40, 50], ['10%', '20%', '30%', '40%', '50%'],
               fontsize=font_size, weight='normal')

    plt.bar(x_zuobiao - 0.5, cohend_data['cohend_value_fa'], color='#FEDC00',
            alpha=0.5, label='父亲抑郁史')
    plt.bar(x_zuobiao + 0.5, cohend_data['cohend_value_mo'], color='#018749',
            alpha=0.5, label='母亲抑郁史')

    for i in range(len(x_zuobiao)):
        plt.text(x=x_zuobiao[i]+0.5, y=-0.5, s=cohend_data['yinbianliang'][i],
                 fontsize=font_size, rotation=45, ha='right', va='top', weight='normal')

    plt.axhline(y=20, color='#DE2910', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='#DE2910', linestyle='--', alpha=0.5)
    plt.axhline(y=40, color='#DE2910', linestyle='--', alpha=0.5)

    plt.legend(frameon=False, shadow=False, loc='best', fontsize=font_size)
    plt.grid(False)
    plt.savefig('paper_and_picture/cohenD.png', bbox_inches='tight', dpi=900)



