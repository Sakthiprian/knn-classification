import pandas as pd
import numpy as np
import math

data = pd.read_csv("D:\coding\PRML\lab3\iris.csv")

#split dataset into each class

versicolor = data.loc[data["Species"] == "Iris-versicolor"]
setosa = data.loc[data["Species"] == "Iris-setosa"]
virginica = data.loc[data["Species"] == "Iris-virginica"]

versicolordrop = versicolor.drop(["Species", "Id"], axis=1)
setosadrop = setosa.drop(["Species", "Id"], axis=1)
virginicadrop = virginica.drop(["Species", "Id"], axis=1)

#randomly select a sample from each class
test = pd.concat([setosa.sample(), versicolor.sample(), virginica.sample()])

#drop the testset from trainset
train = data.drop(test.index, errors='ignore')

#compute mean vector of each class
meanvectset = setosadrop.mean().values
meanvectver = versicolordrop.mean().values
meanvectvir = virginicadrop.mean().values

print("Setosa",meanvectset)
print("Versicolor",meanvectver)
print("Virginica",meanvectvir)

#compute covariance matrix
covMatSet = setosadrop.cov().values
covMatVer = versicolordrop.cov().values
covMatVir = virginicadrop.cov().values

print(np.linalg.inv(covMatSet))

print(covMatSet)

#function for mahalanobis distance
def Mlb(Meanvect, Testvect, covMat):
    inverse = np.linalg.inv(covMat)
    mlb = math.sqrt(np.dot(np.dot(np.subtract(Testvect, Meanvect), inverse), np.transpose(np.subtract(Testvect, Meanvect))))
    return mlb

predlist = []

#for each item in test df, compute mahalanobis distance from mean of setosa,versicolor and virginica
for index, row in test.iterrows():
    testvect = row[["PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"]].values
    print(testvect)

    #append the distances into a list
    lst = [Mlb(meanvectset, testvect, covMatSet), Mlb(meanvectver, testvect, covMatVer), Mlb(meanvectvir, testvect, covMatVir)]
    print(lst)
    minval = min(lst)
    min_index = lst.index(minval)

    #based on index, append class name to prediction list
    if min_index == 0:
        predlist.append("Setosa")
    elif min_index == 1:
        predlist.append("Versicolor")
    else:
        predlist.append("Virginica")

#print prediction list
print(predlist)
print(test["Species"])
