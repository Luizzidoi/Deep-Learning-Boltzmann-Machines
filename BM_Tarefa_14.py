""""
Redes Neurais - Boltzmann Machines
Aprendizagem não supervisionada
RBM - Restricted Boltzmann Machine
Sistemas de recomendação
Recomendação de filmes - Tarefa

"""

from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=3)


""" Criação de uma base de dados com 6 usuários """
base = np.array([[0,1,1,1,0,1], [1,1,0,1,1,1],
                 [0,1,0,1,0,1], [0,1,1,1,0,1],
                 [1,1,0,1,0,1], [1,1,0,1,1,1]])


"""" Cadastro dos filmes conforme a indicação proposta """
filmes = ["Freddy x Jason", "O Ultimato Bourne", "Star Trek",
          "Exterminador do Futuro", "Norbit", "Star Wars"]

""" Treinamento da rede neural """
rbm.train(base, max_epochs=5000)

""" Criação de um novo usuário """
leonardo = np.array([[0,1,0,1,0,0]])

""" Função que diz qual neuronio foi ativado """
camada_escondida = rbm.run_visible(leonardo)


""" Faz a recomendação e imprime o nome dos filmes """
recomendacao = rbm.run_hidden(camada_escondida)
print('Recomendações para Leonardo:')
for i in range(len(leonardo[0])):
    if leonardo[0, i] == 0 and recomendacao[0, i] == 1:
        print(filmes[i])
