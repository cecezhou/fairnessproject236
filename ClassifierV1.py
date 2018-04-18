import sklearn.linear_model as SKL
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import copy
import pandas as pd


### experiment bright students math finance
N = 10000 ## 1000 of each group (groups S and T)

minority_percent = 0.3
MIN = int(minority_percent * N)
MAJ = int((1 - minority_percent) * N)
print(MIN, MAJ)
# p_S_brightmath = 0.9
# p_T_brightmath = 0.1
bright_percent = 0.2

d = 0.3
r = 0.15

a = 7
b = 2
## first attribute is 1 if in group S, otherwise 0 for group T

### Generate Group S, Group T data
## Group Label, Math Grades, Writing Grades, Bright (or not)

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
	print("Bright S ",len(brightS))

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
	print("Bright T ",len(brightT))

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
	print(dfSNot.shape)
	dfTBright = df1[(df1[3] == 1) & (df1[0] == 0)]
	dfTNot = df1[df1[3] == 0  & (df1[0] == 0)]

	# plt.scatter(dfSNot[1], dfSNot[2], c = "pink")
	# plt.scatter(dfSBright[1], dfSBright[2], c = "blue")

	# plt.scatter(dfTNot[1], dfTNot[2], c = "orange")
	# plt.scatter(dfTBright[1], dfTBright[2], c = "red")

	data = (dfSNot, dfSBright, dfTNot, dfTBright)
	colors = ("green", "blue", "yellow", "red")
	groups = ("SNot", "SBright", "TNot", "TBright")
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	for data, color, group in zip(data, colors, groups):
		ax.scatter(data[1],data[2], c= color, edgecolors = None, label = group)
	plt.legend(loc = 3)
	# plt.show()
	plt.savefig(name)
	plt.close()



scatter(X_test, "IdealV2")
scatter(X_B, "BiasedV2")


### fit regular biased world
# reg = SKL.LogisticRegression()
reg = RFC(n_estimators = 30)

reg.fit(X_B[:, :-1], X_B[ : ,-1])
pred = reg.predict(X_test[:, :-1])
# reg.predict()
print("Score for Biased ", reg.score(X_test[:, :-1], X_test[:, -1]))





### fit perturbed biased world
## predict on the training data to get some probabilities

# C = SKL.LogisticRegression()
# C.fit(X_B[:, :-1], X_B[ : ,-1])
# ranksC = C.predict_proba(X_B[ : , : -1])
ranks = reg.predict_proba(X_B[ : , :-1])
print(ranks)

newData = np.append(X_B, np.array([[x] for x in ranks[ :,1]]), axis = 1)
newDf = pd.DataFrame(newData)

S = newDf[(newDf[0]  == 1) & (newDf[3]  == 0)]
sorted_S = S.sort_values(4)

# print(ranksC)

# ### first do the perturbing
not_bright_S  = []
for idx,x in enumerate(X_S[:, -1]):
	if x < 1:
		not_bright_S += [idx]

bright_T  = []
for idx,x in enumerate(X_T[:, -1]):
	if x > 0:
		bright_T += [idx]

perturb_ratio = 0.5 #len(not_bright_S) / (len(not_bright_S) + len(bright_T))
# print(perturb_ratio)

X_P = copy.deepcopy(X_B)
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

print(X_P.shape)
X_P = np.array(X_P)

# scatter(X_P, "PerturbedV2")


# reg1 = SKL.LogisticRegression()
reg1 = RFC(n_estimators = 20)
reg1.fit(X_P[:, :-1], X_P[:,-1])
pred1 = reg1.predict(X_test[:, :-1])

# reg.predict()
print("Score for Biased perturbed " ,reg1.score(X_test[:, :-1], X_test[:, -1]))

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

TN_S, FN_S, FP_S, TP_S = get_fundamentals(pred[:1000], X_test[:1000, -1])

PPV_S = TP_S / (0.001 + TP_S + FP_S)
FPR_S = FP_S / (FP_S + TN_S)
FNR_S = FN_S / (FN_S + TP_S)

TN_T, FN_T, FP_T, TP_T = get_fundamentals(pred[1000:], X_test[1000:, -1])

PPV_T = TP_T / (0.001 + TP_T + FP_T)
FPR_T = FP_T / (FP_T + TN_T)
FNR_T = FN_T / (FN_T + TP_T)

TN1_S, FN1_S, FP1_S, TP1_S = get_fundamentals(pred1[:1000], X_test[:1000, -1])

PPV1_S = TP1_S / (0.001 + TP1_S + FP1_S)
FPR1_S = FP1_S / (FP1_S + TN1_S)
FNR1_S = FN1_S / (FN1_S + TP1_S)

TN1_T, FN1_T, FP1_T, TP1_T = get_fundamentals(pred1[1000:], X_test[1000:, -1])

PPV1_T = TP1_T / (0.001 + TP1_T + FP1_T)
FPR1_T = FP1_T / (FP1_T + TN1_T)
FNR1_T = FN1_T / (FN1_T + TP1_T)

print("S:", "\n",
	  "PPVs =", PPV_S, PPV1_S, "\n",
	  "FPRs =", FPR_S, FPR1_S, "\n",
	  "FNRs =", FNR_S, FNR1_S)

print("T:", "\n",
	  "PPVs =", PPV_T, PPV1_T, "\n",
	  "FPRs =", FPR_T, FPR1_T, "\n",
	  "FNRs =", FNR_T, FNR1_T)












