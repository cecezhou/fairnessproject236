import sklearn.linear_model as SKL
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier as GPC
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import copy
import pandas as pd


model_strings = ["LR",
                 "RFC",
                 #"ABC",
                 "MLPC",
                 "KNC",
                 "SVC",
                 #"DTC",
                 #"GNB",
                 "QDA"]
model_types = [LR,
               RFC,
               #ABC,
               MLPC,
               KNC,
               SVC,
               #DTC,
               #GNB,
               QDA]
models = [LR(),
          RFC(n_estimators=30),
          #ABC(),
          MLPC(),
          KNC(),
          SVC(probability=True),
          #DTC(),
          #GNB(),
          QDA()]
models2 = copy.deepcopy(models)


### experiment bright students math finance
N = 10000 ## 1000 of each group (groups S and T)

minority_percent = 0.3
MIN = int(minority_percent * N)
MAJ = int((1 - minority_percent) * N)
# print(MIN, MAJ)
# p_S_brightmath = 0.9
# p_T_brightmath = 0.1
bright_percent = 0.2

d = 0.2 # this is percent shift of added bias (of MIN and MAJ groups)
r = 0.15

a = 7
b = 2

quota = 1700

## first attribute is 1 if in group S, otherwise 0 for group T

## Generate Group S, Group T data
### X columns:
### Group Label, Math Grades, Writing Grades, Bright (or not)

def generate_X(biased = False):
	### S
	S = np.zeros((MIN, 4))
	S[ :,0] = 1
	brightS = np.random.choice(MIN, int(bright_percent * MIN), replace = False)

	S[ :,1] = [np.random.beta(a,b) if i in brightS else np.random.beta(b,a) for i in range(MIN)]
	S[ :,2] = [np.random.beta(b,a) if i in brightS else np.random.beta(a,b) for i in range(MIN)]

	removed = set()
	if biased == True:
		removed = set(np.random.choice(brightS, int(d/2 * MIN), replace = False))
	brightS = set(brightS)
	brightS = brightS.difference(removed)
	# print("Bright S ",len(brightS))

	S[ :,3] = [1 if i in brightS else 0 for i in range(MIN)]

	### T
	T = np.zeros((MAJ, 4))
	T[ :,0] = 0
	brightT = np.random.choice(MAJ, int(bright_percent * MAJ), replace = False)
	notbrightT = list(set(range(MAJ)).difference(set(brightT)))

	T[ :,1] = [np.random.beta(b,a) if i in brightT else np.random.beta(a,b) for i in range(MAJ)]
	T[ :,2] = [np.random.beta(a,b) if i in brightT else np.random.beta(b,a) for i in range(MAJ)]

	added = set()
	if biased == True:
		added = set(np.random.choice(notbrightT, int(d/2 * MAJ), replace = False))
	brightT = set(brightT)
	brightT = brightT.union(added)
	# print("Bright T ",len(brightT))

	T[ :,3] = [1 if i in brightT else 0 for i in range(MAJ)]

	return S,T


X_S, X_T = generate_X(biased = True)
X_B = np.append(X_S, X_T, axis=0)



### test data of ideal world

X_S_test, X_T_test = generate_X(biased = False)
X_test = np.append(X_S_test, X_T_test, axis=0) ## use this for testing


def scatter(X, name):
	df1 = pd.DataFrame(X)
	dfSBright = df1[(df1[3] == 1) & (df1[0] == 1)]
	dfSNot = df1[(df1[3] == 0) & (df1[0] ==1) ]
	# print(dfSNot.shape)
	dfTBright = df1[(df1[3] == 1) & (df1[0] == 0)]
	dfTNot = df1[df1[3] == 0  & (df1[0] == 0)]

	# plt.scatter(dfSNot[1], dfSNot[2], c = "pink")
	# plt.scatter(dfSBright[1], dfSBright[2], c = "blue")

	# plt.scatter(dfTNot[1], dfTNot[2], c = "orange")
	# plt.scatter(dfTBright[1], dfTBright[2], c = "red")

	data = (dfSNot, dfSBright, dfTNot, dfTBright)
	colors = ("black", "blue", "yellow", "red")
	groups = ("SNot", "SBright", "TNot", "TBright")
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	for data, color, group in zip(data, colors, groups):
		ax.scatter(data[1],data[2], c= color, edgecolors = None, label = group, s = 1)
	plt.legend(loc = 3)
	# plt.show()
	plt.savefig(name)
	plt.close()


scatter(X_test, "IdealV2")
scatter(X_B, "BiasedV2")


### fit regular biased world
reg_LR = SKL.LogisticRegression()
reg_RFC = RFC(n_estimators = 30)

reg_LR.fit(X_B[:, :-1], X_B[ : ,-1])
reg_RFC.fit(X_B[:, :-1], X_B[ : ,-1])

pred_LR = reg_LR.predict(X_test[:, :-1])
pred_RFC = reg_RFC.predict(X_test[:, :-1])

print("Score for Biased LR", reg_LR.score(X_test[:, :-1], X_test[:, -1]))
print("Score for Biased RFC", reg_RFC.score(X_test[:, :-1], X_test[:, -1]))




### fit perturbed biased world
## predict on the training data to get some probabilities

# create perturbed data
def generate_X_P(model, perturb_ratio=0.5):
	X_B_copy = copy.deepcopy(X_B)
	ranks = model.predict_proba(X_B_copy[ : , :-1])

	newData = np.append(X_B_copy, np.array([[x] for x in ranks[ :,1]]), axis = 1)
	newDf = pd.DataFrame(newData)

	S = newDf[(newDf[0]  == 1) & (newDf[3]  == 0)]
	sorted_S = S.sort_values(4)

	# ### first do the perturbing
	not_bright_S  = []
	for idx,x in enumerate(X_S[:, -1]):
		if x < 1:
			not_bright_S += [idx]

	bright_T  = []
	for idx,x in enumerate(X_T[:, -1]):
		if x > 0:
			bright_T += [idx]

	# perturb_ratio = 0.5 #len(not_bright_S) / (len(not_bright_S) + len(bright_T))

	X_P = copy.deepcopy(X_B_copy)
	X_P = pd.DataFrame(X_P)
	num_perturb = int(r * MIN * (perturb_ratio))
	to_perturb = sorted_S.index.tolist()[-num_perturb:]
	X_P.at[to_perturb, 3] = 1

	# bright in T 
	T = newDf[(newDf[0]  == 0) & (newDf[3]  == 1)]
	sorted_T = T.sort_values(4)
	# 1-perturb ratio works for 0.5, fix for other values
	num_perturb_T = int(r * MAJ * (1- perturb_ratio))
	to_perturb_T = sorted_T.index.tolist()[:num_perturb_T]
	X_P.at[to_perturb_T,3] = 0

	# print(X_P.shape)
	X_P = np.array(X_P)
	return X_P

# scatter(X_P, "PerturbedV2")
X_P = generate_X_P(reg_RFC, perturb_ratio=0.5)

reg1_LR = SKL.LogisticRegression()
reg1_RFC = RFC(n_estimators = 30)

reg1_LR.fit(X_P[:, :-1], X_P[:,-1])
reg1_RFC.fit(X_P[:, :-1], X_P[:,-1])

pred1_LR = reg1_LR.predict(X_test[:, :-1])
pred1_RFC = reg1_RFC.predict(X_test[:, :-1])

print("Score for Biased LR perturbed" ,reg1_LR.score(X_test[:, :-1], X_test[:, -1]))
print("Score for Biased RFC perturbed" ,reg1_RFC.score(X_test[:, :-1], X_test[:, -1]))

def predict(X, C):
	ranks = C.predict_proba(X[ : , :-1])
	newData = np.append(X, np.array([[x] for x in ranks[ :,1]]), axis = 1)
	newDf = pd.DataFrame(newData)
	sorted_Df = newDf.sort_values(4)
	# quota = int(sum(pred1_RFC))
	top_indices = set(sorted_Df.index.tolist()[-quota:])
	# new_indices = 
	# print(newDf_P.iloc[top_indices[0]])
	# for indices: setting classifications to 1 and rest to 0
	pred = np.array([1 if i in top_indices else 0 for i in range(N)])
	return pred

pred2_LR = predict(X_test, reg1_LR)
pred2_RFC = predict(X_test, reg1_RFC)

results_LR = [int((a and not b) or (b and not a)) for a, b in zip(pred2_LR, X_test[:, -1])]
score_LR = (N - sum(results_LR)) / N

results_RFC = [int((a and not b) or (b and not a)) for a, b in zip(pred2_RFC, X_test[:, -1])]
score_RFC = (N - sum(results_RFC)) / N

print("Score for Biased LR perturbed quota", score_LR)
print("Score for Biased RFC perturbed quota", score_RFC)

# print("Score for Biased perturbed quota " ,reg2.score(X_test[:, :-1], X_test[:, -1]))


## prediction is different
# diff = pred1 + pred
# diff_people = []
# for idx,i in enumerate(diff):
# 	if i == 1:
# 		diff_people += [idx]


# F = []
# for i in diff_people:
# 	## if true value is also different
# 	F = F + [X_I[i][2]]
# len(diff_people)

# plt.hist(F)
		# print(X_B[i], X_P[i])
# plt.show()

# for i in range(2*N):
# 	## if true value is also different
# 	if (X_B[i][2] != X_P[i][2]):
# 		print(X_B[i], X_P[i])

# ---------------------------------------------------------------------------------
# The Four Fundamental Numbers (TN, FN, TP, FP)

# TN : Classified as 0 (pred) and really 0 (X_I_test[:, 2])

# FN : Classified as 0 but really 1

# FP : Classified as 1 but really 0

# TP : Classified as 1 and really 1

def get_fundamentals(pred, actual):
	num_tests = len(pred)
	fundamentals = [0] * 4  # TN, FN, FP, TP
	for i in range(num_tests):
		fundamentals[int(2*pred[i] + actual[i])] += 1
	return fundamentals

def get_stats(pred, actual):
	fundamentals = get_fundamentals(pred, actual)
	TN, FN, FP, TP = fundamentals
	PPV = TP / (0.001 + TP + FP)
	FPR = FP / (FP + TN)
	FNR = FN / (FN + TP)
	return PPV, FPR, FNR

PPV_S, FPR_S, FNR_S = get_stats(pred_RFC[:MIN], X_test[:MIN, -1]) # biased
PPV_T, FPR_T, FNR_T = get_stats(pred_RFC[MIN:], X_test[MIN:, -1]) # biased
PPV1_S, FPR1_S, FNR1_S = get_stats(pred1_RFC[:MIN], X_test[:MIN, -1]) # biased perturbed
PPV1_T, FPR1_T, FNR1_T = get_stats(pred1_RFC[MIN:], X_test[MIN:, -1]) # biased perturbed
PPV2_S, FPR2_S, FNR2_S = get_stats(pred2_RFC[:MIN], X_test[:MIN, -1]) # biased perturbed quota
PPV2_T, FPR2_T, FNR2_T = get_stats(pred2_RFC[MIN:], X_test[MIN:, -1]) # biased perturbed quota

print("b = biased, p = perturbed, q = quota")
print("S:")
print("       b       bp      bpq")
print("PPVs = {:.4f}, {:.4f}, {:.4f}".format(PPV_S, PPV1_S, PPV2_S))
print("FPRs = {:.4f}, {:.4f}, {:.4f}".format(FPR_S, FPR1_S, FPR2_S))
print("FNRs = {:.4f}, {:.4f}, {:.4f}".format(FNR_S, FNR1_S, FNR2_S))
print("T:")
print("       b       bp      bpq")
print("PPVs = {:.4f}, {:.4f}, {:.4f}".format(PPV_T, PPV1_T, PPV2_T))
print("FPRs = {:.4f}, {:.4f}, {:.4f}".format(FPR_T, FPR1_T, FPR2_T))
print("FNRs = {:.4f}, {:.4f}, {:.4f}".format(FNR_T, FNR1_T, FNR2_T))







preds, b_scores, bp_scores = [], [], []
for i, model in enumerate(models):
	## biased
	print(model_strings[i])
	model.fit(X_B[:, :-1], X_B[ : ,-1])
	pred = model.predict(X_test[:, :-1])
	b_score = model.score(X_test[:, :-1], X_test[:, -1])
	preds.append(pred)
	b_scores.append(b_score)
	
	## biased perturbed
	X_P = generate_X_P(model, perturb_ratio=0.5)
	model2 = models2[i]
	model2.fit(X_P[:, :-1], X_P[:,-1])
	bp_score = model2.score(X_test[:, :-1], X_test[:, -1])
	bp_scores.append(bp_score)


print(model_strings)
print(b_scores)
print(bp_scores)


## Testing individual fairness
### Similar individuals should be classified similarly

## create custom data
M = 100
X_IF = []
for i in range(2):
	for j in range(M):
		for k in range(M):
			X_IF.append([i, j/M, k/M])
X_IF = np.array(X_IF)


def produce_graphs(X, name_S, name_T):
	df = pd.DataFrame(X)
	dfSBright = df[(df[3] == 1) & (df[0] == 1)]
	dfSNot = df[(df[3] == 0) & (df[0] ==1) ]
	dfTBright = df[(df[3] == 1) & (df[0] == 0)]
	dfTNot = df[df[3] == 0  & (df[0] == 0)]

	data = (dfSNot, dfSBright)
	colors = ("blue", "red")
	groups = ("SNot", "SBright")

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	for data, color, group in zip(data, colors, groups):
		ax.scatter(data[1],data[2], c= color, edgecolors = None, label = group, s = 1)
	plt.legend(loc = 3)
	plt.savefig(name_S)
	plt.close()

	data = (dfTNot, dfTBright)
	colors = ("blue", "red")
	groups = ("TNot", "TBright")

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	for data, color, group in zip(data, colors, groups):
		ax.scatter(data[1],data[2], c= color, edgecolors = None, label = group, s = 1)
	plt.legend(loc = 3)
	plt.savefig(name_T)
	plt.close()


pred_IF, pred2_IF = [], []
for i, model in enumerate(models):
	X_IF_new = copy.deepcopy(X_IF)
	## predict using biased model + produce graph
	pred_IF.append(model.predict(X_IF[:,:3]))
	X_IF_new = np.c_[X_IF_new, np.array(pred_IF[i])]
	name_S = "individual_fairness/%s_S_M-%04.f_b" % (model_strings[i], M)
	name_T = "individual_fairness/%s_T_M-%04.f_b" % (model_strings[i], M)
	produce_graphs(X_IF_new, name_S, name_T)

	X_IF_new = copy.deepcopy(X_IF)
	## predict using biased, perturbed model + produce graph
	pred2_IF.append(models2[i].predict(X_IF[:,:3]))
	X_IF_new = np.c_[X_IF_new, np.array(pred2_IF[i])]
	name_S = "individual_fairness/%s_S_M-%04.f_bp" % (model_strings[i], M)
	name_T = "individual_fairness/%s_T_M-%04.f_bp" % (model_strings[i], M)
	produce_graphs(X_IF_new, name_S, name_T)










