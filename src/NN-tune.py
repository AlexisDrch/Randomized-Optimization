 # In[29]:
from utils import *
from sklearn.neural_network import MLPClassifier


clf_nn_lbfgs = MLPClassifier(solver = 'lbfgs') # very heavy to compute (hessian every time)

array_activation = ['relu', 'logistic', 'tanh']
array_layer = [(100,),(100,100),(100,100,100),
(60,),(60,60),(60,60,60),
(70,),(70,70),(70,70,70),
(80,),(80,80),(80,80,80),
(90,),(90,90),(90,90,90)
]


parameter_grid = {
	'hidden_layer_sizes': array_layer,
	'activation' : array_activation,
}
    
grid = GridSearchCV(
    clf_nn_lbfgs,
    param_grid = parameter_grid,
    cv = cv
)  

#print()
#grid.fit(X1, y1)
#pima_svm_score = grid.best_score_
#print('Best score: {}'.format(pima_svm_score))
#print('Best parameters: {}'.format(grid.best_params_))

# ### 3. iteration curve: since nn is an iteration algorithm


# degree can be ignoer out of poly
max_iter = 5000 # set really large to see where we should stop
title_pima = "Iteration on pima dataset with optimal neural net"
clf_nn_lbfgs_opti = MLPClassifier(hidden_layer_sizes= (60,60), activation = 'relu', solver = 'lbfgs')
plot = plot_iterative_learning_curve(clf_nn_lbfgs_opti, title_pima, X1, y1, ylim=None, cv = cv, n_jobs=-1,#
	iterations=np.arange(1, max_iter, 200))
plot.savefig('./output/iterative-pima-nn.png')
plot.show()