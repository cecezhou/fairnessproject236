### experiment bright students math finance

N = 1000 ## 1000 of each group (groups S and T)
import sklearn.linear_model as SKL
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import copy

p = 0.2
q = 0.3
d = 0.2
r = 0.1


## first attribute is 1 if in group S, otherwise 0 for group T

mu = 0.4






### Generate Group S, Group T data
def generate_X(p, q, mu, N):
	covMatrix_S = np.array([[1, 1-q],[1-q,1]])
	MVN_S = np.random.multivariate_normal(np.array([p, mu]), covMatrix_S, N)
	X_S = np.append(np.array([[1]]*N), MVN_S, axis = 1)
	X_S[:, 2] = [percentileofscore(X_S[:,2], x) for x in X_S[:,2]]

	covMatrix_T = np.array([[1, q],[q,1]])
	MVN_T = np.random.multivariate_normal(np.array([1-p, mu]), covMatrix_T, N)
	X_T = np.append(np.array([[0]]*N), MVN_T, axis = 1)
	X_T[:, 2] = [percentileofscore(X_T[:, 2], x) for x in X_T[:, 2]]

	return X_S, X_T

X_S, X_T = generate_X(p, q, mu, N)
X_S_test, X_T_test = generate_X(p, q, mu, N)


## Group S

# covMatrix_S = np.array([[1, 1-q],[1-q,1]])
# MVN_S = np.random.multivariate_normal(np.array([p, mu]), covMatrix_S, N)

# X_S = np.append(np.array([[1]]*N), MVN_S, axis = 1)

# X_S[:, 2] = [percentileofscore(X_S[:,2], x) for x in X_S[:,2]]





## Group T
# covMatrix_T = np.array([[1, q],[q,1]])
# MVN_T = np.random.multivariate_normal(np.array([1-p, mu]), covMatrix_T, N)
# X_T = np.append(np.array([[0]]*N), MVN_T, axis = 1)
# X_T[:, 2] = [percentileofscore(X_T[:, 2], x) for x in X_T[:, 2]]


X = np.append(X_S, X_T, axis=0)
X_test = np.append(X_S_test, X_T_test, axis=0)

## Ideal world
X_I = copy.deepcopy(X)
X_I[:, 2] = [1 if x > 80 else 0 for x in X[:,2]]

## Ideal world (test)
X_I_test = copy.deepcopy(X_test)
X_I_test[:, 2] = [1 if x > 80 else 0 for x in X_test[:,2]]



## biased world
X_B_S = copy.deepcopy(X_S)
X_B_T = copy.deepcopy(X_T)

X_B_S[:, 2] = [1 if x > 90 else 0 for x in X_S[:,2]] 
X_B_T[:, 2] = [1 if x > 70 else 0 for x in X_T[:,2]] 

X_B = np.append(X_B_S, X_B_T, axis = 0)



## Group S

# covMatrix_B_S = np.array([[1, 1-q],[1-q,1]])
# MVN_B_S = np.random.multivariate_normal(np.array([p, mu_I - d/2]), covMatrix_B_S, N)
# X_B_S = np.append(np.array([[1]]*N), MVN_B_S, axis = 1)


# ## Group T

# covMatrix_B_T = np.array([[1, q],[q,1]])
# MVN_B_T = np.random.multivariate_normal(np.array([1-p, mu_I + d/2]), covMatrix_B_T, N)
# X_B_T = np.append(np.array([[0]]*N), MVN_B_T, axis = 1)

# X_B = np.append(X_B_S, X_B_T, axis = 0)


### fit regular biased world
reg = SKL.LogisticRegression()

reg.fit(X_B[:, :2], X_B[ : ,2])
pred = reg.predict(X_I_test[:, :2])
# reg.predict()
print("Score for Biased ", reg.score(X_I_test[:, :2], X_I_test[:, 2]))

### fit perturbed biased world
not_bright_S  = []
for idx,x in enumerate(X_B_S[:, 2]):
	if x < 1:
		not_bright_S += [idx]


bright_T  = []
for idx,x in enumerate(X_B_T[:, 2]):
	if x > 0:
		bright_T += [idx]


num_perturb = len(not_bright_S) / (len(not_bright_S) + len(bright_T))
# print(num_perturb)

indices_S = np.random.choice(not_bright_S, int(r * N * (num_perturb)),replace = False)


X_P_S = copy.deepcopy(X_B_S)

for i in indices_S:
	X_P_S[i, 2] = 1






indices_T = np.random.choice(bright_T, int(r * N * (1-num_perturb)), replace = False)

X_P_T = copy.deepcopy(X_B_T)

for i in indices_T:
	X_P_T[i, 2] = 0


X_P = np.append(X_P_S, X_P_T, axis = 0)


reg1 = SKL.LogisticRegression()
reg1.fit(X_P[:, :2], X_P[:,2])
pred1 = reg1.predict(X_I_test[:, :2])

# reg.predict()
print("Score for Biased perturbed " ,reg1.score(X_I_test[:, :2], X_I_test[:, 2]))

## prediction is different
diff = pred1 + pred
diff_people = []
for idx,i in enumerate(diff):
	if i == 1:
		diff_people += [idx]


F = []
for i in diff_people:
	## if true value is also different
	F = F + [X_I[i][2]]
len(diff_people)

plt.hist(F)
		# print(X_B[i], X_P[i])
# plt.show()

# for i in range(2*N):
# 	## if true value is also different
# 	if (X_B[i][2] != X_P[i][2]):
# 		print(X_B[i], X_P[i])






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

TN_S, FN_S, FP_S, TP_S = get_fundamentals(pred[:1000], X_I_test[:1000, 2])

PPV_S = TP_S / (TP_S + FP_S)
FPR_S = FP_S / (FP_S + TN_S)
FNR_S = FN_S / (FN_S + TP_S)

TN_T, FN_T, FP_T, TP_T = get_fundamentals(pred[1000:], X_I_test[1000:, 2])

PPV_T = TP_T / (TP_T + FP_T)
FPR_T = FP_T / (FP_T + TN_T)
FNR_T = FN_T / (FN_T + TP_T)

TN1_S, FN1_S, FP1_S, TP1_S = get_fundamentals(pred1[:1000], X_I_test[:1000, 2])

PPV1_S = TP1_S / (TP1_S + FP1_S)
FPR1_S = FP1_S / (FP1_S + TN1_S)
FNR1_S = FN1_S / (FN1_S + TP1_S)

TN1_T, FN1_T, FP1_T, TP1_T = get_fundamentals(pred1[1000:], X_I_test[1000:, 2])

PPV1_T = TP1_T / (TP1_T + FP1_T)
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












