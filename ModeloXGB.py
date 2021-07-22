#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from xgboost import plot_importance
import pickle
from scipy import stats


df = pd.read_csv('Dados4.csv')
'''
df.drop(columns=['Status_C'],inplace=True)
df.drop(columns=['Status_L'],inplace=True)
df.drop(columns=['Status_S'],inplace=True)
'''
'''
z = np.abs(stats.zscore(df))
df = df[(z<3).all(axis=1)]
print(df)
'''
# Separando o Dataframe entre preditoras X e preditiva y
X, y = df.iloc[:,:-1], df.iloc[:,-1]
# Seprando X em X para treino e teste e y para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42)
# Este modelo pede os valores no formato DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Definindo métricas iniciais para comparação futura
# "Learn" the mean from the training data
mean_train = np.mean(y_train)
# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train
# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))
# Escolha dos parâmetros do modelo,
# estes valores foram definidos em outro programa
params = {   # Parameters that we are going to tune.
          'max_depth':12,
          'min_child_weight': 2,
          'eta':.2,
          'subsample': .55,
          'colsample_bytree': .85,
          'objective':'reg:squarederror',}
params['eval_metric'] = "mae" #Erro médio absoluto
num_boost_round = 200 #500
# Criando e treinando o modelo
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)
# Prevendo o RCT no dataframe de teste
y_pred = model.predict(dtest)
y_predtrain = model.predict(dtrain)
# Verificando o erro médio absoluto
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
# "Learn" the mean from the training data
mean_train = np.mean(y_pred)
# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train
# Compute MAE
print("Baseline MAE is {:.2f}".format(mae_baseline))
print(mean_absolute_error(y_pred, y_test))
print(r2_score(y_test,y_pred))
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(),\
y_test.max()], 'k--', lw=4)
ax.set_xlabel('Atual')
ax.set_ylabel('Previsto')
ax.set_title("Valores reais vs Previstos")
'''
# Gráfico de comparação entre previsto e real
plt.show()
plot_importance(model)
# Grafico das váriaveis mais importantes para o modelo
plt.show()


plt.title("Valores reais vs Previstos")
plt.scatter(y_test.index,y_pred, label='previsto')
plt.scatter(y_test.index,y_test, c='orange', label='real')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0), label="Dados de teste")
ax.scatter(y_train,y_predtrain, label="Dados do treino")
ax.plot([y_test.min(), y_test.max()], [y_test.min(),\
y_test.max()], 'k--', lw=4)
ax.set_xlabel('Atual')
ax.set_ylabel('Previsto')
ax.set_title("Valores reais vs Previstos")
# Gráfico de comparação entre previsto e real
plt.legend()
plt.show()


dict1 = model.get_score(importance_type='gain')
sorted_values = sorted(dict1.values()) # Sort the values
sorted_dict = {}

for i in sorted_values:
    for k in dict1.keys():
        if dict1[k] == i:
            sorted_dict[k] = dict1[k]
            break

print(sorted_dict)
'''

filename = "modeloXGBPermeancia3.pkl"
with open(filename, 'wb') as file:
    pickle.dump(model, file)

