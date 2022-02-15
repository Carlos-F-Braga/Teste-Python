import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm #Importando as Dependênciais necessárias

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11] #Declaração das minhas variáveis de dados em arrays

plt.scatter(x,y)
plt.show() #scatter de x,y e show pelo plt

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]]) 
y = [0,1,0,1,0,1] #novo array X e Y com novos dados

clf = svm.SVC(kernel='linear', C = 1.0) #definindo o classificador

clf.fit(X,y) #dando fit do clt com X e y

w = clf.coef_[0]
print(w)
a = -w[0] / w[1]
xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]
h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show() #últimas declarações necessárias e visualizando os dados