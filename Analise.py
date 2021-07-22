import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('Dados ZPPR10.csv',';',encoding='iso-8859-1')
df = df.fillna(0)
'''
print(df.describe())

plt.figure(figsize = (15,15))
ax = sns.histplot(data = df, x = 'PP_', bins = 10, color='darkgreen')
ax.set_xlabel('Permeância', fontsize=20)
ax.set_ylabel('Contagem', fontsize=20)
ax.set_title('Distribuição dos Valores de Permeância', fontdict={'fontsize':25})
ax.tick_params(axis = 'both', labelsize =15)
plt.show()
'''


df.drop(columns=['Descrição', 'Material', 'Status'], inplace=True)
index = df[(df['PP_PERMEANCIA'] < 5)].index
df.drop(index, inplace=True)
index = df[(df['PP_TIPO_SACO_COLADO'] == 'COLBA') | (df['PP_TIPO_SACO_COLADO'] == 'COLPB')].index
df.drop(index, inplace=True)
index = df[(df['PP_TIP_PERF_FOL_EXT'] == '') | (df['PP_TIP_PERF_FOL_INT'] == '')].index
df.drop(index, inplace=True)


volume = []
for row in range(len(df)):
    altura = float(df['PP_ALTURA_UTIL'].iloc[row])
    fundo = (float(df['PP_FUNDO'].iloc[row]) + float(df['PP_FUNDO_INF'].iloc[row]))/2
    face = float(df['PP_FACE'].iloc[row])
    volume.append(np.round((((altura*0.2452)+(fundo*0.3275)-(face*0.1121))*(face**2))/10**6,3))


listPapeis = []
count = 0
for column in df.columns:
    if 'PP_TIP_PERF' in column:
        papeis = [str(row) for row in df[column].values]
        for papel in papeis:
            listPapeis.append(papel)
print(sorted(set(listPapeis)))

df['PP_VOLUME'] = volume


print(df.head())

categorical_feature_mask = df.dtypes==object
categorical_cols = df.columns[categorical_feature_mask].tolist()

for column in df[categorical_cols]:
    df[column] = [str(value) for value in df[column].values]

for column in df[categorical_cols]:
    print(column)
    print(sorted(set(df[column])))

for column in df[categorical_cols]:
    dum_df = pd.get_dummies(df[column], columns=[column], prefix=column)
    df = df.join(dum_df)

df.drop(columns = categorical_cols, inplace=True)
last_col = df.pop('PP_PERMEANCIA')
df.insert(len(df.columns), 'PP_PERMEANCIA', last_col)
print(df)
'''
sns.countplot(data=df, x="PP_PERMEANCIA", color='green')
plt.locator_params(axis='x', nbins=30)
plt.title('Distribuição Permeância', fontsize=15)
plt.xlabel('Permeância [m³/h]', fontsize=10)
plt.ylabel('Quantidade', fontsize=10)
plt.show()
'''
df_corr = df.corr()
#print(df_corr.loc['PP_VOLUME','PP_PERMEANCIA'])

'''
ax = sns.heatmap(
    df_corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20,220,n=200),
    square=True
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()
'''
''' 
ax = sns.lmplot(data = df, x = 'PP_VOLUME', y = 'PP_PERMEANCIA', palette = 'gray')
ax.fig.suptitle('Relação Entre o Volume e a Permeância da Sacaria', fontsize = 25)
ax.set_axis_labels("Volume", "Permeância", fontsize=20)
plt.show()
'''
'''
list = []
for row in df_corr['PP_PERMEANCIA']:
    list.append(row)

count = 0
for column in df.columns:
    if(list[count]>=0.3 or list[count]<=-0.3):
        plt.scatter(df['PP_PERMEANCIA'], df[column], c='black')
        plt.xlabel('PP_PERMEANCIA')
        plt.ylabel(column)
        plt.title('Correlação = {0}'.format(list[count]))
        plt.plot(np.unique(df['PP_PERMEANCIA']), np.poly1d(np.polyfit(df['PP_PERMEANCIA'], df[column], 1))(np.unique(df['PP_PERMEANCIA'])), linestyle='dashed', linewidth=3, color='blue')
        plt.show()
    count = count + 1
'''

df.to_csv('Dados4.csv', index=False)
