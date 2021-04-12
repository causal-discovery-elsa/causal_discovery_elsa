import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
import scipy.stats
import re
import time
import subprocess
import os
from features import *
import scipy.signal as ss
import numpy as np
import math
from numpy import diff
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
#import feather

#"scako" was removed because wave 1 had different scale
trtmntVar = set(["scfrda","scfrdg","scfrdm","heacta", "heactb","heactc", "scorg03","scorg06","scorg05","scorg07","heskb"]) #11
confoundersVar = set(["indager", "hehelf","dhsex","totwq10_bu_s"])   #6
binaryVariables = set(["scorg03","scorg06","scorg05","scorg07","dhsex"])
targetVar = set(["memIndex"])  #1
auxVar = set(["cfdscr","cflisen", "cflisd","cfdatd"])
drvVar = set(["memIndexChange", "baseMemIndex"])  #1
allVar = trtmntVar|confoundersVar|targetVar

weights = {"scfrda":1,"scfrdg":1,"scfrdm":1,"heacta":1, "heactb":1,"heactc":1, "scorg03":1,"scorg06":1,"scorg05":1,"scorg07":1,"heskb":1,"indager":2, "hehelf":1,"dhsex":1,"totwq10_bu_s":1,"baseMemIndex":1}



basePath = "/home/ali/Downloads/UKDA-5050-stata_3/stata/stata11_se"
REFUSAL=-9
DONT_KNOW=-8
NOT_APPLICABLE=-1
SCHD_NOT_APPLICABLE=-2

NOT_ASKED=-3

NOT_IMPUTED = -999.0
NON_SAMPLE = -998.0
INST_RESPONDENT=  -995.0


def returnSample(df):
	df = df.sample(20)
	varNames = ["heactb","scfrdm","scorg06"]
	col_list = []
	
		# for num in range(1,8):
			

	for num in range(1,8):
		
		for var in varNames:
			col_list.append("{}_b_{}".format(var,num))
		col_list.append( "memIndex_{}".format(num))		
	return df[col_list]

def removeMedianValues(df):
	for i in range(1,8):
		columnName = "scfrdm_b_{}".format(i)
		df= df.drop( df[ (df[columnName]==2)].index)
	return df

def report(df, var):
	for i in [3,4,5]:
		print "wave",i
		print "min", df["{}_{}".format(var, i)].min()
		print "max", df["{}_{}".format(var, i)].max()
		print "mean", df["{}_{}".format(var, i)].mean()
		print "std", df["{}_{}".format(var, i)].std()

def harmonizeData(df):
	# print allVar
	for var in (trtmntVar|confoundersVar):
		df[var] = df[var].apply((globals()[var].harmonize))
	
	# for var in ["heacta", "heactb", "heactc", "scako", "heskb"]:
	#     df[var] = df[var].apply((globals()[var].binarize))

	return df



def binarizeData(df):
	pattern = r"[a-zA-Z0-9_]*_n$"
	# cols = list(df.columns)
	# cols.remove('idauniq')
	for var in trtmntVar:
		if not re.match(pattern, var):  
		    col_bin = var + '_b'
		    df[col_bin] = df[var].apply((globals()[var].binarize))
	return df


def normalizeData(df):
	# cols = list(df.columns)
	# cols.remove('idauniq')
	# for col in cols:
	#     col_norm = col + '_n'
	#     df[col_norm] = (df[col] - df[col].min())/(df[col].max()- df[col].min())

	for var in (trtmntVar|confoundersVar|targetVar|drvVar):
		print "var:{}".format(var)
		dfs=[]
		for i in range(1,8):
			col = "{}_{}".format(var,i)
			dfs.append(pd.DataFrame( {var: df[col]}))
		mergedDf = pd.concat(dfs)
		mean= mergedDf[var].mean()
		len(np.where( mergedDf[var]))
		std = mergedDf[var].std(ddof=1)
		print "mean:{}, std:{}, size:{}".format(mean, std, len(mergedDf[var]))
		minValue = mergedDf[var].min()
		maxValue = mergedDf[var].max()
		for i in range(1,8):
			col = "{}_{}".format(var,i)
			col_norm = "{}_n_{}".format(var,i)
			df[col_norm] = (df[col] - mean)/(std)

	return df

def readWave1Data(basePath):
	waveNumber=1
	Core = pd.read_stata("{}/wave_1_core_data_v3.dta".format(basePath, waveNumber),convert_categoricals=False)
	Drv =  pd.read_stata("{}/wave_1_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
	FinDrv = pd.read_stata('{}/wave_1_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
	
	s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
	df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])
	
	df = df.rename(columns = {'scorg3':'scorg03'})
	df = df.rename(columns = {'scorg5':'scorg05'})
	df = df.rename(columns = {'scorg6':'scorg06'})
	df = df.rename(columns = {'scorg7':'scorg07'})
	
	col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
	df = df [col_list] 

	df = addMemIndex(df)
	df = removeAuxVars(df)

	df = harmonizeData(df)
	# df = normalizeData(df)
	df = binarizeData(df)
	# df= addSuffix(df,1)
	# df = df.ix[0:50,:]
	return df

def addSuffix(df, num):
	for var in (trtmntVar|confoundersVar|targetVar):
		newName = "{}_{}".format(var,num)
		df = df.rename(columns = {var:newName})
	return df


def readWave2Data(basePath):
	waveNumber=2
	Core = pd.read_stata("{}/wave_2_core_data_v4.dta".format(basePath, waveNumber),convert_categoricals=False)
	Drv =  pd.read_stata("{}/wave_2_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
	FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
	

	s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
	df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])

	df = df.rename(columns = {'HeActa':'heacta'})
	df = df.rename(columns = {'HeActb':'heactb'})
	df = df.rename(columns = {'HeActc':'heactc'})
	df = df.rename(columns = {'Hehelf':'hehelf'})
	df = df.rename(columns = {'HeSkb':'heskb'})
	df = df.rename(columns = {'DhSex':'dhsex'})

	df = df.rename(columns = {'CfDScr':'cfdscr'})
	df = df.rename(columns = {'CfLisEn':'cflisen'})
	df = df.rename(columns = {'CfLisD':'cflisd'})
	df = df.rename(columns = {'CfDatD':'cfdatd'})

	col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
	df = df [col_list] 

	df = addMemIndex(df)
	df = removeAuxVars(df)

	df = harmonizeData(df)
	# df = normalizeData(df)
	df = binarizeData(df)
	# df = df.ix[0:50,:]
	return df


def readWave3Data(basePath):
	waveNumber=3
	Core = pd.read_stata("{}/wave_{}_elsa_data_v4.dta".format(basePath, waveNumber),convert_categoricals=False)
	Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
	FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
	

	s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
	df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])

	df = df.rename(columns = {'hegenh':'hehelf'})
	col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
	df = df [col_list] 

	df = addMemIndex(df)
	df = removeAuxVars(df)

	df = harmonizeData(df)
	# df = normalizeData(df)
	df = binarizeData(df)
	# df = df.ix[0:50,:]
	return df



def readWave4Data(basePath):
	waveNumber=4
	Core = pd.read_stata("{}/wave_{}_elsa_data_v3.dta".format(basePath, waveNumber),convert_categoricals=False)
	Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
	FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
	
	s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
	df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])

	col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
	df = df [col_list] 

	df = addMemIndex(df)
	df = removeAuxVars(df)

	df = harmonizeData(df)
	# df = normalizeData(df)
	df = binarizeData(df)
	# df = df.ix[0:50,:]
	return df


def readWave5Data(basePath):
	waveNumber=5
	Core = pd.read_stata("{}/wave_{}_elsa_data_v4.dta".format(basePath, waveNumber),convert_categoricals=False)
	Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
	FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)

	s1 = pd.merge(Core, Drv, how='inner', on=['idauniq'])
	df = pd.merge(s1, FinDrv, how='inner', on=['idauniq'])

	col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
	df = df [col_list] 

	df = addMemIndex(df)
	df = removeAuxVars(df)

	df = harmonizeData(df)
	# df = normalizeData(df)
	df = binarizeData(df)
	# df = df.ix[0:50,:]
	return df


def readWave6Data(basePath):
	waveNumber=6
	w6Core = pd.read_stata("{}/wave_{}_elsa_data_v2.dta".format(basePath, waveNumber),convert_categoricals=False)
	w6Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
	w6FinDrv = pd.read_stata('{}/wave_{}_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
	

	s1 = pd.merge(w6Core, w6Drv, how='inner', on=['idauniq'])
	df = pd.merge(s1, w6FinDrv, how='inner', on=['idauniq'])

	# df = df.rename(columns = {'hegenh':'hehelf'})
	df = df.rename(columns = {'HeActa':'heacta'})
	df = df.rename(columns = {'HeActb':'heactb'})
	df = df.rename(columns = {'HeActc':'heactc'})
	df = df.rename(columns = {'Hehelf':'hehelf'})
	df = df.rename(columns = {'HeSkb':'heskb'})
	df = df.rename(columns = {'DhSex':'dhsex'})

	df = df.rename(columns = {'CfDScr':'cfdscr'})
	df = df.rename(columns = {'CfLisEn':'cflisen'})
	df = df.rename(columns = {'CfLisD':'cflisd'})
	df = df.rename(columns = {'CfDatD':'cfdatd'})



	# col_list = ["idauniq","heacta","heactb","heactc", "scorg03", "scorg06", "scorg05", "scorg07", "hehelf",
	# 			 "scfrda" , "scfrdg","scako", "heskb", "indager", "dhsex" , "scfrdm", "memtotb","totwq10_bu_s",  ]
	
	col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
	df = df [col_list] 

	df = addMemIndex(df)
	df = removeAuxVars(df)

	df = harmonizeData(df)
	# df = normalizeData(df)
	df = binarizeData(df)
	# df = df.ix[0:50,:]
	return df




def readWave7Data(basePath):
	waveNumber=7
	w6Core = pd.read_stata("{}/wave_{}_elsa_data.dta".format(basePath, waveNumber),convert_categoricals=False)
	w6Drv =  pd.read_stata("{}/wave_{}_ifs_derived_variables.dta".format(basePath, waveNumber),convert_categoricals=False)
	w6FinDrv = pd.read_stata('{}/wave_7_financial_derived_variables.dta'.format(basePath, waveNumber),convert_categoricals=False)
	

	s1 = pd.merge(w6Core, w6Drv, how='inner', on=['idauniq'])
	df = pd.merge(s1, w6FinDrv, how='inner', on=['idauniq'])

	df = df.rename(columns = {'HeActa':'heacta'})
	df = df.rename(columns = {'HeActb':'heactb'})
	df = df.rename(columns = {'HeActc':'heactc'})
	df = df.rename(columns = {'Hehelf':'hehelf'})
	df = df.rename(columns = {'HeSkb':'heskb'})
	df = df.rename(columns = {'DhSex':'dhsex'})
	df = df.rename(columns = {'scfrdl':'scfrdm'})



	df = df.rename(columns = {'CfDScr':'cfdscr'})
	df = df.rename(columns = {'CfLisEn':'cflisen'})
	df = df.rename(columns = {'CfLisD':'cflisd'})
	df = df.rename(columns = {'CfDatD':'cfdatd'})



	# col_list = ["idauniq","heacta","heactb","heactc", "scorg03", "scorg06", "scorg05", "scorg07", "hehelf",
	# 			 "scfrda" , "scfrdg","scako", "heskb", "indager", "dhsex" , "scfrdm", "memtotb","totwq10_bu_s",  ]
	
	col_list = ["idauniq"] + list(trtmntVar) + list(confoundersVar) + list(auxVar); 
	df = df [col_list] 

	df = addMemIndex(df)
	df = removeAuxVars(df)

	df = harmonizeData(df)
	# df = normalizeData(df)
	df = binarizeData(df)
	# df = df.ix[0:50,:]
	return df


def removeAuxVars(df):
	df= df.drop(list(auxVar),axis=1)
	return df


def readData(mergeMethod="inner"):
	df1 = readWave1Data(basePath)
	df2 = readWave2Data(basePath)
	df3 = readWave3Data(basePath)
	df4 = readWave4Data(basePath)
	df5 = readWave5Data(basePath)
	df6 = readWave6Data(basePath)
	df7 = readWave7Data(basePath)


	df12 = pd.merge(df1,  df2, how=mergeMethod, on=['idauniq'],suffixes=('_1', ''))
	df13 = pd.merge(df12, df3, how=mergeMethod, on=['idauniq'],suffixes=('_2', ''))
	df14 = pd.merge(df13, df4, how=mergeMethod, on=['idauniq'],suffixes=('_3', ''))
	df15 = pd.merge(df14, df5, how=mergeMethod, on=['idauniq'],suffixes=('_4', ''))
	df16 = pd.merge(df15, df6, how=mergeMethod, on=['idauniq'],suffixes=('_5', ''))
	df17 = pd.merge(df16, df7, how=mergeMethod, on=['idauniq'],suffixes=('_6', '_7'))

	# df34 = pd.merge(df3, df4, how='inner', on=['idauniq'],suffixes=('_3', ''))
	# df35 = pd.merge(df34, df5, how='inner', on=['idauniq'],suffixes=('_4', '_5'))

	return df17

	# return [df1, df2, df3, df4, df5, df6 , df7]

def addMemIndex(df):
	df["memIndex"] = df.apply(computeMemIndex, axis=1)
	df= df.dropna(subset=["memIndex"])
	return df


def computeMemIndexChange2(row, waveNumber):
	memtotVarCur = "memIndex_{}".format(waveNumber) 
	memtotVarPrev = "memIndex_{}".format(waveNumber-2)
	return row[memtotVarCur] - row[memtotVarPrev]


def computeMemIndexChange(row, waveNumber):
        memtotVarCur = "memIndex_{}".format(waveNumber)
        memtotVarPrev = "memIndex_{}".format(waveNumber-1)
        return row[memtotVarCur] - row[memtotVarPrev]




def computeMemIndex(row):
	if row["cfdatd"] == REFUSAL:
		return np.nan
	if row ["cflisd"] == DONT_KNOW:
		row ["cflisd"] = 0

	if row ["cflisen"] == DONT_KNOW:
		row ["cflisen"] = 0

	if (row ["cfdscr"]<0) or (row ["cflisd"]<0) or (row ["cflisen"]<0):
		return np.nan
	else:
		return row["cfdscr"] + row["cflisd"] + row["cflisen"]


def computeDistance(row1,row2, weights_local):
	
	# print "row1" 
	# print row1
	# print "row2"
	# print row2
	# print "weights"
	# print weights_local
	diff  = row1 - row2
	diff = diff*weights_local
	diff = diff[~np.isnan(diff)]
	return np.linalg.norm(diff)/len(diff)


def preProcessData2(df):
	# df= df.dropna(axis=0, how="any")
	# df= df.dropna(subset=["memIndex_1", "memIndex_", "memIndex", "memIndex","memIndex","memIndex","memIndex"])
    
	df["memIndexChange_1"]=  np.nan
	df["memIndexChange_2"] = np.nan
	df["memIndexChange_3"] = df.apply(computeMemIndexChange2,waveNumber=3,axis=1)
	df["memIndexChange_4"] = df.apply(computeMemIndexChange2,waveNumber=4,axis=1)
	df["memIndexChange_5"] = df.apply(computeMemIndexChange2,waveNumber=5,axis=1)
	df["memIndexChange_6"] = df.apply(computeMemIndexChange2,waveNumber=6,axis=1)
	df["memIndexChange_7"] = df.apply(computeMemIndexChange2,waveNumber=7,axis=1)


	df["baseMemIndex_7"] = df["memIndex_5"]
	df["baseMemIndex_6"] = df["memIndex_4"]
	df["baseMemIndex_5"] = df["memIndex_3"]
	df["baseMemIndex_4"] = df["memIndex_2"]
	df["baseMemIndex_3"] = df["memIndex_1"]
	df["baseMemIndex_2"] = np.nan
	df["baseMemIndex_1"] = np.nan


	df = normalizeData(df)
	
	
	return df


def preProcessData(df):
        # df= df.dropna(axis=0, how="any")
        # df= df.dropna(subset=["memIndex_1", "memIndex_", "memIndex", "memIndex","memIndex","memIndex","memIndex"])

        df["memIndexChange_1"]=  np.nan
        df["memIndexChange_2"] = df.apply(computeMemIndexChange,waveNumber=2,axis=1)
        df["memIndexChange_3"] = df.apply(computeMemIndexChange,waveNumber=3,axis=1)
        df["memIndexChange_4"] = df.apply(computeMemIndexChange,waveNumber=4,axis=1)
        df["memIndexChange_5"] = df.apply(computeMemIndexChange,waveNumber=5,axis=1)
        df["memIndexChange_6"] = df.apply(computeMemIndexChange,waveNumber=6,axis=1)
        df["memIndexChange_7"] = df.apply(computeMemIndexChange,waveNumber=7,axis=1)


        df["baseMemIndex_7"] = df["memIndex_6"]
        df["baseMemIndex_6"] = df["memIndex_5"]
        df["baseMemIndex_5"] = df["memIndex_4"]
        df["baseMemIndex_4"] = df["memIndex_3"]
        df["baseMemIndex_3"] = df["memIndex_2"]
        df["baseMemIndex_2"] = df["memIndex_1"]
        df["baseMemIndex_1"] = np.nan


        df = normalizeData(df)


        return df



def getTreatmentGroups(df, indVariable, waveNumber):
	varCurrWave = "{}_b_{}".format(indVariable, waveNumber)
	varPrevWave = "{}_b_{}".format(indVariable, waveNumber-1)
	memIndexChangeVar = "memIndexChange_{}".format(waveNumber)
	
	currentWave  = np.array(df[varCurrWave])
	prevWave = np.array(df[varPrevWave])
	memChange = np.isnan(df.loc[:,memIndexChangeVar]).astype(int)

	C = np.multiply(1-prevWave, 1-currentWave)
	C=  np.multiply(C, 1-memChange)

	T = np.multiply(1-prevWave, currentWave)
	T=  np.multiply(T, 1-memChange)	

	controlIndexes = np.where(C==1)[0]
	treatmentIndexes = np.where(T==1)[0]

	return [controlIndexes, treatmentIndexes]

def getTreatmentGroups2(df, indVariable, waveNumber):
	varCurrWave = "{}_b_{}".format(indVariable, waveNumber)
	varPrevWave = "{}_b_{}".format(indVariable, waveNumber-1)
	varPrev2Wave = "{}_b_{}".format(indVariable, waveNumber-2)
	memIndexChangeVar = "memIndexChange_{}".format(waveNumber)
	
	currentWave  = np.array(df[varCurrWave])
	prevWave = np.array(df[varPrevWave])
	prev2Wave = np.array(df[varPrev2Wave])
	memChange = np.isnan(df.loc[:,memIndexChangeVar]).astype(int)

	C = np.multiply(1- prev2Wave, 1-prevWave)
	C = np.multiply(C,1- currentWave)
	C=  np.multiply(C, 1-memChange)

	T = np.multiply(1- prev2Wave, prevWave)
	T = np.multiply(T, currentWave)
	T=  np.multiply(T, 1-memChange)	

	controlIndexes = np.where(C==1)[0]
	treatmentIndexes = np.where(T==1)[0]

	return [controlIndexes, treatmentIndexes]



def ComputeCostMatrix(df, treatmentGroups, indVariable, waveNumber):
	controlIndexes = treatmentGroups[0]
	treatmentIndexes = treatmentGroups[1]

	# cols = df.columns.tolist()
	# cols.remove('idauniq')
	# pattern = r"[a-zA-Z0-9]*_n_{}$".format(waveNumber)
	confounders = []
	# for colName in cols:
	# 	if (re.match(pattern, colName) and not (indVariable in colName)):
	# 		confounders.append(colName)

	weights_local = []

	for var in ((trtmntVar| confoundersVar | set(["baseMemIndex"]))- set([indVariable])):
		if var in binaryVariables:
			colName = "{}_{}".format(var,waveNumber)
		else:
			colName= "{}_n_{}".format(var,waveNumber)
		confounders.append(colName)	
		if var == "indager":
			weights_local.append(1)
		elif var == "baseMemIndex":
			weights_local.append(100)
		else:
			weights_local.append(1)

	#print "conf:"
	#print confounders
	confDF = df[confounders]
	# print confounders

	numTreat = len(treatmentIndexes)
	numControl = len(controlIndexes)
	C = np.zeros(shape = (numTreat, numControl))
	for i in tqdm(range(numTreat)):
		for j in range(numControl):
	# for i in tqdm([0]):
	# 	for j in [1]:			
			# print confDF.loc[treatmentIndexes[i]]
			# print confDF.loc[treatmentIndexes[i]].values
			# print weights_local
			C[i,j] = computeDistance(confDF.loc[treatmentIndexes[i]].values, confDF.loc[controlIndexes[j]].values,weights_local)

	return C



def run_cmd(cmd, working_directory=None):
	if working_directory!= None:
		try:
			output = subprocess.check_output(cmd,shell=True,cwd=working_directory)
			print "output:"+output
		except:
			print "failed:"+cmd
			# pass
	else:
		try:
			output = subprocess.check_output(cmd,shell=True)
			print(output)
		except:
			print "failed:"+cmd
			# pass


def performMatching(C):

	r,c = C.shape
	with open('matrix.txt', 'w') as f:
		f.write("{} {}\n".format(r,c))
		for i in range(0,r):
			for j in range(0,c):
				f.write( "{} ".format(C[i][j]))
			f.write("\n")

	command = "hungarian/test"
	run_cmd(command)
	
	costs = []
	with open('matching.txt', 'r') as f:
		indexes = []
		for line in f:
			words = line.rstrip('\n').split(',')
			L = int(words[0])
			R = int(words[1])
			if R!= -1:
				pair = (L,R)
				indexes.append(pair)
				costs.append(C[L,R])

	costs = np.array(costs)
	passedPairs = [pair for idx, pair in enumerate(indexes) if costs[idx]< 0.3 ]			
	# m = Munkres()
	# indexes = m.compute(C)
	return passedPairs

def getTargetValues2(df, treatmentGroups, indexes, waveNumber):
	memTotChangeVar = "memIndex_{}".format(waveNumber)
	prevWave = "memIndex_{}".format(waveNumber-2)
	controlIndexes = treatmentGroups[0]
	treatmentIndexes = treatmentGroups[1]
	memtotT = [  df.loc[treatmentIndexes[i[0]]][memTotChangeVar]  for i in indexes]
	memtotC = [  df.loc[controlIndexes[i[1]]][memTotChangeVar]  for i in indexes]

	memtotT_P = [  df.loc[treatmentIndexes[i[0]]][prevWave]  for i in indexes]
	memtotC_P = [  df.loc[controlIndexes[i[1]]][prevWave]  for i in indexes]

	f = open("sample_ids.txt", "a")
	for i in indexes:
		f.write("{} {} {}\n".format( treatmentIndexes[ i[0]], controlIndexes[i[1]], waveNumber))
	f.close()
	return [memtotC, memtotT,  memtotC_P, memtotT_P]



def getTargetValues(df, treatmentGroups, indexes, waveNumber):
        memTotChangeVar = "memIndex_{}".format(waveNumber)
        prevWave = "memIndex_{}".format(waveNumber-1)
        controlIndexes = treatmentGroups[0]
        treatmentIndexes = treatmentGroups[1]
        memtotT = [  df.loc[treatmentIndexes[i[0]]][memTotChangeVar]  for i in indexes]
        memtotC = [  df.loc[controlIndexes[i[1]]][memTotChangeVar]  for i in indexes]

        memtotT_P = [  df.loc[treatmentIndexes[i[0]]][prevWave]  for i in indexes]
        memtotC_P = [  df.loc[controlIndexes[i[1]]][prevWave]  for i in indexes]

        return [memtotC, memtotT,  memtotC_P, memtotT_P]



def computeDropouts(df, var):
	v1 = np.isnan(df["{}_1".format(var)])
	v2 = np.logical_and(v1, (np.isnan(df["{}_1".format(var)])))
	# v3 = np.logical_and(v2, np.isnan(df["memIndex_3"]))
	v3= np.isnan(df["{}_2".format(var)])
	v4 = np.logical_and(v3, np.isnan(df["{}_3".format(var)]))
	v5 = np.logical_and(v4, np.isnan(df["{}_4".format(var)]))
	v6 = np.logical_and(v5, np.isnan(df["{}_5".format(var)]))
	v7 = np.logical_and(v6, np.isnan(df["{}_6".format(var)]))

	print (len(np.where(v1==True)[0]))
	print (len(np.where(v2==True)[0]))
	print (len(np.where(v3==True)[0]))
	print (len(np.where(v4==True)[0]))
	print (len(np.where(v5==True)[0]))
	print (len(np.where(v6==True)[0]))
	print (len(np.where(v7==True)[0]))


def computePValue(X,Y):
	res= scipy.stats.wilcoxon(X,Y,"wilcox")
	pVal = res[1]
	return pVal


def getVariableData(df, var):
	Xvars=[]
	for i in range(1,8):
		Xvar = "{}_{}".format(var, i)
		Xvars.append(Xvar)
	# print Xvars

	newDF = df[Xvars]

	# minVal= float("-inf")
	# maxVal = float("inf")
	# for var in Xvars:
	# 	# print var
	# 	if newDF[var].min()<minVal:
	# 		minVal = newDF[var].min()
	# 	if newDF[var].max()>maxVal:
	# 		maxVal = newDF[var].max()
	
	# for var in Xvars:
	# 	newDF[var] =  (newDF[var] - newDF[var].min())/(newDF[var].max()- newDF[var].min())	
			
	return newDF



def exportVariables(df):

	for var in (trtmntVar|targetVar):
		D = getVariableData(df, var)
		D.to_csv("{}_data.csv".format(var))

	return



	X = getVariableData(df, X)

def getVariableDataBinary(df, var):
	Xvars=[]
	for i in range(1,8):
		Xvar = "{}_b_{}".format(var, i)
		Xvars.append(Xvar)
	# print Xvars

	newDF = df[Xvars]

	minVal= float("-inf")
	maxVal = float("inf")
	for var in Xvars:
		# print var
		if newDF[var].min()<minVal:
			minVal = newDF[var].min()
		if newDF[var].max()>maxVal:
			maxVal = newDF[var].max()
	
	for var in Xvars:
		newDF[var] =  (newDF[var] - newDF[var].min())/(newDF[var].max()- newDF[var].min())	
			
	return newDF

def detectLag(a,b):
	# print a
	# print b
	# # result = ss.correlate(Xvec, Yvec, method="direct")
	# print "a", len(a)
	# print "b", len(b)

	# result= ss.correlate(a - np.mean(a), b - np.mean(b), method='direct')/(np.std(a)*np.std(b)*len(a))
	result= ss.correlate(a, b, method='direct')
	return result

def findWaveVars(inputList):
	pattern = r"[a-zA-Z0-9_]*_1$"
	for var in inputList:
		if re.match(pattern, var):
			print var 


def computeLag(df, X,Y):
	X = getVariableData(df, X)
	Y = getVariableData(df, Y)

	X = X.interpolate()

	res = []

	# lags = {}
	# counter= {}
	# for i in range(-7,8):
		# lags[i]=0
		# counter[i] = 0


	for i in range(0,len(Y)):
		lag = detectLag(diff(X.loc[i]), diff(Y.loc[i]))
		res.append(lag)
		# print X.loc[i]
		# print Y.loc[i]
		# print lag
		# if  not math.isnan(lag[1]):
		# 	# print type(lag[1])
		# 	# print lag[1]
		# 	lags[lag[0]]+= lag[1]
		# 	counter[lag[0]]+= 1

	return (X,Y,res)

	# for i in range(-7,8):
	# 	print "lag: {} , sum: {:.2f}".format(i, lags[i]) 
	# 	if counter[i]:
	# 		print "\t avg: {0:.2f}".format(lags[i]/counter[i])	

	# return lags, counter


def computeLagForAllVars(df):
	for var in trtmntVar:
		print "examining ", var
		res= computeLag(df,var, list(targetVar)[0])

dfPath="ELSA1.pkl"




def f():
	# start_time = time.time()

	if (os.path.isfile(dfPath)):
		df = pd.read_pickle(dfPath)
	else:
		df = readData()
	df = preProcessData2(df)  #alternative



	pVals = {}
	for indVariable in trtmntVar:
		pVals[indVariable] = []
	
	for indVariable in trtmntVar:
	#for indVariable in ["heactb"]:
		s =time.time()
		print indVariable
		controlValues= []
		treatmentValues= []
		controlValuesPrev = []
		treatmentValuesPrev = []
		for waveNumber in [3,4,5,6,7]:
		# for waveNumber in [5]:
			print waveNumber
			treatmentGroups = getTreatmentGroups2(df,indVariable, waveNumber) #alternative
			C= ComputeCostMatrix(df, treatmentGroups, indVariable, waveNumber)
			matchedPairs = performMatching(C)
			targetValues = getTargetValues2(df,treatmentGroups, matchedPairs, waveNumber) #alternative
			controlValues = controlValues+ targetValues[0]
			treatmentValues = treatmentValues + targetValues[1]
			controlValuesPrev = controlValuesPrev+ targetValues[2]
			treatmentValuesPrev = treatmentValuesPrev + targetValues[3]
			pval = computePValue(targetValues[0], targetValues[1])
			print "len:{}".format(len(targetValues[0]))
			print "pval", pval
			print "C:{}, T:{}, PrevC:{}, prevT:{}".format(np.mean(targetValues[0]), np.mean(targetValues[1]), np.mean(targetValues[2]), np.mean(targetValues[3]))	

		print "len:{}".format(len(controlValues))
		pval = computePValue(controlValues, treatmentValues)
		print "pval", pval
		pVals[indVariable].append(pval)	
		elapsedTime = time.time()-s
		print "C:{}, T:{}, PrevC:{}, prevT:{}".format(np.mean(controlValues), np.mean(treatmentValues), np.mean(controlValuesPrev), np.mean(treatmentValuesPrev))
		print "processing time:", elapsedTime/60		

	return (pVals, controlValues, treatmentValues, controlValuesPrev, treatmentValuesPrev) 


def readTreatedGroup():
	ids = []
	with open("sample_ids.txt") as f:
		for line in f:
			words= line.split(" ")
			tid = int(words[0])
			w = int(words[2])
			ids.append((tid,w))	
	return ids


def detectTreatedGroup(df, indVariable, waveNum):
	res = []
	for waveNumber in range(waveNum,waveNum+1):	
		varCurrWave = "{}_b_{}".format(indVariable, waveNumber)
		varPrevWave = "{}_b_{}".format(indVariable, waveNumber-1)
		varPrev2Wave = "{}_b_{}".format(indVariable, waveNumber-2)
		memIndexChangeVar = "memIndexChange_{}".format(waveNumber)

		currentWave  = np.array(df[varCurrWave])
		prevWave = np.array(df[varPrevWave])
		prev2Wave = np.array(df[varPrev2Wave])
		memChange = np.isnan(df.loc[:,memIndexChangeVar]).astype(int)

		T = np.multiply(1- prev2Wave, prevWave)
		T = np.multiply(T, currentWave)
		T=  np.multiply(T, 1-memChange)

		treatmentIndexes = np.where(T==1)[0]
		pairs = zip( treatmentIndexes,  np.repeat(waveNumber, len(treatmentIndexes)) )
	#		print pairs.shape
		res = res+ pairs
		
	return res




def getMIsFromID(ids, df):
	MIs= []
	for pair in ids:
		w= pair[1]
		n_id = int(pair[0])
		n_id = df.index[n_id]
		MI = [ df.loc[n_id, "memIndex_{}".format(w-2)],  df.loc[n_id, "memIndex_{}".format(w-1)],  df.loc[n_id, "memIndex_{}".format(w)]]
		MIs.append(MI)
	return MIs 	

def getConfsFromID(ids, df):
        MIs= []
        for pair in ids:
                w= pair[1]
                n_id = int(pair[0])
                n_id = df.index[n_id]
                MI = [ df.loc[n_id, "indager_{}".format(w)],  df.loc[n_id, "dhsex_{}".format(w)]]
                MIs.append(MI)
        return MIs





def getMIs(df):
	cols = []
	for i in range(1,8):
		name = "memIndex_{}".format(i)
		cols.append(name)
	return df[cols]
	

def getDerivative(df):
	return np.diff( df.as_matrix() ,axis=1)


def heidegger():
	df = readData()
	


def harmonizeBias(D):
	for i in range(0,len(D)):
		D[i] = D[i]-D[i][0]
	return D



def computeSimilarityMatrix(D):
	N=len(D)
	S = np.zeros((N,N))
	for i in  tqdm(range(0,N)):
		for j in range(0,N):
			diff = D[i]-D[j]
			S[i][j] = np.linalg.norm(diff)
	return S

def runKMean(D, k):

	Sum_of_squared_distances = []
	silhouette_scores = []
	K = range(2,20)
	for k in K:
	   km = KMeans(n_clusters=k)
	   km = km.fit(D)
	   labels= km.labels_
	   print len(labels)
	   print len(D)
	
	   Sum_of_squared_distances.append(km.inertia_)
	   silhouette_scores.append(silhouette_score(D, labels))
	
	plt.plot(K, Sum_of_squared_distances, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Sum_of_squared_distances')
	plt.title('Elbow Method For Optimal k')
	plt.show()

	plt.plot(K, silhouette_scores, 'bx-')
	plt.xlabel('k')
	plt.ylabel('silhouette scores')
	# plt.title('Elbow Method For Optimal k')
	plt.show()


	return Sum_of_squared_distances

	kmeans = KMeans(n_clusters=k, n_init=100 )
	kmeans.fit(D)
	centroids = kmeans.cluster_centers_

	for i in range(0,k):
		print "size of cluster :{}".format(i)
		print len(np.where(kmeans.labels_==i)[0])
		#np.where(kmeans.labels_==i)


	#T = pd.DataFrame()
	#for i in range(0,k):
	#	T[len(np.where(kmeans.labels_==i)[0])]=centroids[i]
	# T.plot(subplots=True, legend=False)
	#T.plot()	
	#plt.show()
	return kmeans

def runKMeanWithSimilarity(S, D, k):

	# Sum_of_squared_distances = []
	# K = range(1,30)
	# for k in K:
	#     km = KMeans(n_clusters=k)
	#     km = km.fit(D)
	#     Sum_of_squared_distances.append(km.inertia_)
	# plt.plot(K, Sum_of_squared_distances, 'bx-')
	# plt.xlabel('k')
	# plt.ylabel('Sum_of_squared_distances')
	# plt.title('Elbow Method For Optimal k')
	# plt.show()

	# return Sum_of_squared_distances
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(D)
	centroids = kmeans.cluster_centers_
	centroids =zeros((k,D.shape[1]))
	L = kmeans.labels_

	for i in range(0,k):
		np.average(np.where(L==i)[0],0)

	T = pd.DataFrame()
	for i in range(0,k):
		T[i]=centroids[i]
	T.plot(subplots=True, legend=False)	
	plt.show()
	return kmeans




def fitLine(D, degree, f_num):
	S = np.zeros(shape=(len(D), f_num))
	for i in range(len(D)):
		coeff = np.polyfit([1,2,3],D[i,:], degree)
		S[i]= coeff[0:f_num]

	return S


def getFeaturesFromIDs(IDs, variables, df):
	F = np.zeros(shape=(len(IDs),len(variables)*3))
	for i, pair in enumerate(IDs):
		tid = pair[0]
		tid = df.index[tid]
		w = pair[1]
		columns=[]
	        for var in variables:
                	for offset in [2,1,0]:
                        	columns.append("{}_{}".format(var, w-offset))
		F[i,:]=np.array(df.loc[tid,columns])
	return F


def getTrtFeaturesFromIDs(IDs, indVariable, df):
	F = np.zeros(shape=(len(IDs),3))
	for i, pair in enumerate(IDs):
		tid = pair[0]
		tid = df.index[tid]
		w = pair[1]
		columns=[]
		for i in [2,1,0]:
			columns.append("{}_{}".format(indVariable, w-i))
		F[i,:]=np.array(df.loc[tid,columns])
	return F





def fillNans(D):
	df = pd.DataFrame(D)
		

from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier


def normalizeArray(D):
	D2 = np.zeros(shape=D.shape)
	for i in range(D.shape[1]):
		mean = np.mean(D[:,i])
		std = np.std(D[:,i])
		D2[:,i] = (D[:,i]-mean)/std
	return D2
		

def getCriticalVars(indVariable, waveNum, features):
	# cols= [ "indager",  "hehelf",  "totwq10_bu_s", "scfrdm",  "dhsex", "scorg05", "heactb"]
	res = []
	for var in features:
		for i in [waveNum-2,waveNum-1,waveNum]:
			res.append("{}_{}".format(var, i))
	for i in range(waveNum-2,waveNum+1):
		res.append("{}_{}".format(indVariable, i))
	for i in [waveNum-2,waveNum]:
		res.append("{}_{}".format(list(targetVar)[0], i))
	print "vars:"
	print res
	return res


def getFeatures(indVariable,df):
	dfo = df.copy()
	
	cols = ["indager",  "hehelf",  "totwq10_bu_s", "scfrdm",  "dhsex", "scorg05", "heactb", "baseMemIndex"]
	criticalVarsBase = ["indager",  "hehelf",  "totwq10_bu_s", "scfrdm",  "dhsex", "scorg05", "heactb"]
	
	totalMIs =  np.array([], dtype=np.int64).reshape(0,1)
	features =  np.array([], dtype=np.int64).reshape(0,len(cols)*3)	
	
	for waveNum in  range(3,8):
		df= dfo.copy()
		indVars = [ "indager",  "hehelf",  "totwq10_bu_s", "scfrdm",  "dhsex", "scorg05", "heactb"]
		criticalVars = getCriticalVars(indVariable, waveNum, criticalVarsBase)
		df= df.dropna(subset = criticalVars)
		ids = detectTreatedGroup(df,indVariable, waveNum)
		
		MIs  = getMIsFromID(ids, df )
		D= np.array(MIs)	
		diff = D[:,2]-D[:,0]
		diff = diff.reshape(len(diff),1)
		totalMIs = np.concatenate( (totalMIs, diff), axis=0)
		
		S= getFeaturesFromIDs(ids, cols, df)
		features= np.concatenate( (features, S), axis=0)	

	indVarCols=[]
	for var in cols:
			for offset in [2,1,0]:
				indVarCols.append("{}_{}".format(var, 3-offset))

	
	refinedDF = pd.DataFrame(data= features, columns=(indVarCols))
	refinedDF["target"] = totalMIs.reshape(len(totalMIs))

	for var in criticalVarsBase:
		print "{}_{}".format(var, 3)
		print "{}_{}".format(var,2)
		print "{}_{}".format(var, 1)
		newCol = (refinedDF["{}_{}".format(var, 3)] + refinedDF["{}_{}".format(var,2)]+refinedDF["{}_{}".format(var, 1)] )/3
		refinedDF[var] = newCol 

	refinedDF.to_csv("refinedDF_{}.csv".format(indVariable), index=False)

def getFeatureImportance(df, indVariable):
	K=2

	dfo = df.copy()
	cols = (trtmntVar | confoundersVar | targetVar)-set([indVariable])
	print cols
	cols = list(cols)
	targetValues =  np.array([], dtype=np.int64).reshape(0,1)
	totalMIs =  np.array([], dtype=np.int64).reshape(0,1)
	totalTrts = np.array([], dtype=np.int64).reshape(0,3)
	features =  np.array([], dtype=np.int64).reshape(0,len(cols))
	for waveNum in  range(3,8):
		df= dfo.copy()
		criticalVars = getCriticalVars(indVariable, waveNum)
		df= df.dropna(subset = criticalVars)
		ids = detectTreatedGroup(df,indVariable, waveNum)
		print "size of ids:"
		print len(ids)	
		MIs  = getMIsFromID(ids, df )
		D= np.array(MIs)	
		diff = D[:,2]-D[:,0]
		diff = diff.reshape(len(diff),1)
		totalMIs = np.concatenate( (totalMIs, diff), axis=0)
		D2 = fitLine(D, 1, 1)
		confs = getConfsFromID(ids, df)
		confs = np.array(confs)
		confs = normalizeArray(confs)
		#finalD = np.concatenate( (D2, confs), axis=1)
		finalD= D2
		targetValues= np.concatenate( (targetValues, finalD), axis=0)
		S= getFeaturesFromIDs(ids, cols, df  )
		trts =  getTrtFeaturesFromIDs(ids, indVariable, df  )
		totalTrts= np.concatenate( (totalTrts, trts), axis=0)
		features= np.concatenate( (features, S), axis=0)

	

	indVarCols=[]
	for i in range(1,4):
		indVarCols.append("{}_{}".format(indVariable, i))
	data= np.concatenate( (features, totalTrts), axis=1)
	refinedDF = pd.DataFrame(data= data, columns=(cols+indVarCols))
	refinedDF["target"] = totalMIs.reshape(len(totalMIs))
	refinedDF.to_csv("refinedDF_{}.csv".format(indVariable), index=False)
    #feather.write_dataframe(refinedDF, "/home/ali/Documents/SFU/Research/dementia/R/refined_df_{}.feather".format(indVariable))

	kmean = runKMean(targetValues,K)
	print kmean.cluster_centers_
	print "silhoutte score: {}".format( silhouette_score (targetValues, kmean.labels_, metric='euclidean'))
	L=kmean.labels_

	#S2=np.nan_to_num(S)
	imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
	imputer = imputer.fit(features)
        S2 = imputer.transform(features)
	S2_original = S2.copy()
	S2= normalizeArray(S2)
	rf = RandomForestClassifier()	
	rf.fit(S2,L)
	#print rf.feature_importances_
	p=zip(cols,  rf.feature_importances_)
	print p	
	for i in range(0, len(cols)):
		print  cols[i]
		print rf.feature_importances_[i]
		for label in range(0,K):
			mean = np.mean(S2_original[np.where(L==label), i])
			print "\tL:{} - Mean: {}".format(label, mean)
				

if __name__ == "__main__":
	print "a"
