import sklearn.linear_model as SKL
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import copy



### experiment bright students math finance
N = 2000 ## 1000 of each group (groups S and T)

minority_percent = 0.3
MIN = int(minority_percent * N)
MAJ = int((1 - minority_percent) * N)
print(MIN, MAJ)
p_S_brightmath = 0.9
p_T_brightmath = 0.1
bright_percent = 0.4

d = 0.2
r = 0.2


## first attribute is 1 if in group S, otherwise 0 for group T

### Generate Group S, Group T data

def generate_X(biased = False):
	### S
	S = np.zeros((MIN, 3))
	S[ :,0] = 1
	brightS = np.random.choice(MIN, int(bright_percent * MIN), replace = False)
	removed = set()
	if biased == True:
		removed = set(np.random.choice(brightS, int(d/2 * MIN), replace = False))
	brightS = set(brightS)
	brightS = brightS.difference(removed)

	S[ :,2] = [1 if i in brightS else 0 for i in range(MIN)]
	S[ :,1] = [1 if ((np.random.rand() > p_S_brightmath and i in brightS) or 
				 (np.random.rand() > 1 - p_S_brightmath and i not in brightS)) else 0 for i in range(MIN)]

	### T
	T = np.zeros((MAJ, 3))
	T[ :,0] = 0
	brightT = np.random.choice(MAJ, int(bright_percent * MAJ), replace = False)
	added = set()
	if biased == True:
		added = set(np.random.choice(brightT, int(d/2 * MAJ), replace = False))
	brightT = set(brightT)
	brightT = brightT.union(added)

	T[ :,2] = [1 if i in brightT else 0 for i in range(MAJ)]
	T[ :,1] = [1 if ((np.random.rand() > p_T_brightmath and i in brightT) or 
				 (np.random.rand() > 1 - p_T_brightmath and i not in brightT)) else 0 for i in range(MAJ)]

	return S,T


X_S, X_T = generate_X(biased = True)
X_B = np.append(X_S, X_T, axis=0)


### test data of ideal world
X_S_test, X_T_test = generate_X(biased = False)
X_test = np.append(X_S_test, X_T_test, axis=0) ## use this for testing




### fit regular biased world
reg = SKL.LogisticRegression()

reg.fit(X_B[:, :2], X_B[ : ,2])
pred = reg.predict(X_test[:, :2])
# reg.predict()
print("Score for Biased ", reg.score(X_test[:, :2], X_test[:, 2]))

### fit perturbed biased world
### first do the perturbing
not_bright_S  = []
for idx,x in enumerate(X_S[:, 2]):
	if x < 1:
		not_bright_S += [idx]

bright_T  = []
for idx,x in enumerate(X_T[:, 2]):
	if x > 0:
		bright_T += [idx]

perturb_ratio =  0.5 # len(not_bright_S) / (len(not_bright_S) + len(bright_T))
# print(perturb_ratio)


indices_S = np.random.choice(not_bright_S, int(r * N * (perturb_ratio)),replace = False)
X_P_S = copy.deepcopy(X_S)
for i in indices_S:
	X_P_S[i, 2] = 1



indices_T = np.random.choice(bright_T, int(r * N * (1- perturb_ratio)), replace = False)
X_P_T = copy.deepcopy(X_T)

for i in indices_T:
	X_P_T[i, 2] = 0

X_P = np.append(X_P_S, X_P_T, axis = 0)


reg1 = SKL.LogisticRegression()
reg1.fit(X_P[:, :2], X_P[:,2])
pred1 = reg1.predict(X_test[:, :2])

# reg.predict()
print("Score for Biased perturbed " ,reg1.score(X_test[:, :2], X_test[:, 2]))

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

TN_S, FN_S, FP_S, TP_S = get_fundamentals(pred[:1000], X_test[:1000, 2])

PPV_S = TP_S / (0.001 + TP_S + FP_S)
FPR_S = FP_S / (FP_S + TN_S)
FNR_S = FN_S / (FN_S + TP_S)

TN_T, FN_T, FP_T, TP_T = get_fundamentals(pred[1000:], X_test[1000:, 2])

PPV_T = TP_T / (0.001 + TP_T + FP_T)
FPR_T = FP_T / (FP_T + TN_T)
FNR_T = FN_T / (FN_T + TP_T)

TN1_S, FN1_S, FP1_S, TP1_S = get_fundamentals(pred1[:1000], X_test[:1000, 2])

PPV1_S = TP1_S / (0.001 + TP1_S + FP1_S)
FPR1_S = FP1_S / (FP1_S + TN1_S)
FNR1_S = FN1_S / (FN1_S + TP1_S)

TN1_T, FN1_T, FP1_T, TP1_T = get_fundamentals(pred1[1000:], X_test[1000:, 2])

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












