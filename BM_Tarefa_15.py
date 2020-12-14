""""
Redes Neurais - Boltzmann Machines
Aprendizagem não supervisionada
Redução de dimensionalidade
Redução de dimensionalidade em imagens - Tarefa
Comparar os resultados com e sem a utilização de RBM aplicado em uma rede neural densa

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split
# Importação da classe para rede neural utilizando o scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


""" Carregar a base de dados do sklearn """
base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = base.target

""" Normalização dos dados """
normalizador = MinMaxScaler(feature_range=(0, 1))
previsores = normalizador.fit_transform(previsores)

""" Divisão da base de dados em treinamento e teste """
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.2, random_state=0)


""" Implementação """
rbm = BernoulliRBM(random_state=0)
rbm.n_iter = 25
rbm.n_components = 50

# O parâmetro hidden_layer_sizes cria as camadas escondidas, sendo que cada número
# 37 representa uma camada. Neste exemplo temos duas camadas escondidas com 37
# neurônios cada uma - usada a fórmula (entradas + saídas) / 2 = (64 + 10) / 2 = 37
# No scikit-learn não é necessário configurar a camada de saída, pois ele
# faz automaticamente. Definimos o max_iter com no máximo 1000 épocas, porém,
# quando a loos function não melhora depois de um certo número de rodadas ele
# pára a execução. O parâmetro verbosa mostra as mensagens na tela
mlp_rbm = MLPClassifier(hidden_layer_sizes=(37, 37), activation='relu', solver='adam', batch_size=50,
                        max_iter=1000, verbose=1)

""" Criação do pipeline para executarmos o rbm e logo após o mlp """
classificador_rbm = Pipeline(steps=[('rbm', rbm), ('mlp', mlp_rbm)])
classificador_rbm.fit(previsores_treinamento, classe_treinamento)

""" Previsoes utilizando rbm e a classe MLPClassifier """
previsoes_rbm = classificador_rbm.predict(previsores_teste)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)
print('\n')

""" Previsões sem a utilização do rbm """
mlp_simples = MLPClassifier(hidden_layer_sizes=(37, 37), activation='relu', solver='adam', batch_size=50,
                        max_iter=1000, verbose=1)
classificador_simples = mlp_simples.fit(previsores_treinamento, classe_treinamento)

previsoes_simples = classificador_simples.predict(previsores_teste)
precisao_simples = metrics.accuracy_score(previsoes_simples, classe_teste)

print('\n')
print('A precisão usando as técnicas de MLP e RBM é: ', precisao_rbm)
print('A precisão usando a técnica de MLP e sem RBM é: ', precisao_simples)


print('Fim')



