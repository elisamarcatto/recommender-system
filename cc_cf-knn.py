# loading libraries
import pandas as pd
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

pd.options.mode.chained_assignment = None  # default='warn'

def normalize(df):

	# Colunas que serao normalizadas
	featureList = ['pretensao'];

	result = df.copy();
	for feature_name in featureList:
	    max_value = df[feature_name].max();
	    min_value = df[feature_name].min();
	    result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value);
	return result;

def load_data():

	# Obtendo dados dos usuarios
	names = ['dormir', 'estado', 'frequencia', 'id_cidade', 'id_contratante', 'id_segmento',
	'latitude', 'longitude', 'nota_vaga', 'pretensao', 'id_profissional', 'ranking_1', 'ranking_2', 'ranking_3'];
	hirerDf = pd.read_csv('/home/elisa/TCC/test/cc_hirer.csv', header=None, names=names, delimiter=",");

	names = ['id_profissional', 'id_cidade', 'estado', 'nota_curriculo', 'avaliacao_geral', 'id_segmento',
	'frequencia', 'pretensao', 'dormir'];
	professionalDf = pd.read_csv('/home/elisa/TCC/test/cc_professional1.csv', header=None, names=names, delimiter=",");

	# Normalizando colunas 
	normalizedHirerDf = normalize(hirerDf);
	normalizedProfessionalDf = normalize(professionalDf);	

	return normalizedHirerDf, normalizedProfessionalDf;

def check_equality(value1, value2):
	if(value1 == value2):
		return 0;
	else:
		return 1;	

def knn(trainingSet, testSet, k):

	columns = ['id_profissional', 'id_contratante', 'index', 'distancia', 'ranking', 'score'];
	distanceDf = pd.DataFrame(columns=columns);
	neighborsDf = pd.DataFrame(columns=columns);

	for row1 in testSet.itertuples():
		print("id", row1[5])

		for row2 in trainingSet.itertuples():
			if (row2[5] != row1[5]):

				# Campos em que semelhanca tem resultado discreto
				sleepDistance = check_equality(row1[1], row2[1]);		
				stateDistance = check_equality(row1[2], row2[2]);		
				frequencyDistance = check_equality(row1[3], row2[3]);	
				cityDistance = check_equality(row1[4], row2[4]);			
				
				# Campos em que semelhanca tem resultado continuo
				salaryDistance = euclidean_distances(row1[10], row2[10]);			

				totalDistance = sleepDistance + stateDistance + frequencyDistance + cityDistance + salaryDistance;			

				distanceDf.loc[len(distanceDf.index)] = [row2[12], row1[5], row1.Index, totalDistance, 1, 0];
				distanceDf.loc[len(distanceDf.index)] = [row2[13], row1[5], row1.Index, totalDistance, 2, 0];
				distanceDf.loc[len(distanceDf.index)] = [row2[14], row1[5], row1.Index, totalDistance, 3, 0];				

		print("distance-df", distanceDf)			

		res = distanceDf.sort_values(['distancia'],ascending=True).head(k*3);			
		neighborsDf = neighborsDf.append(res, ignore_index=True);
		distanceDf.drop(distanceDf.index, inplace=True);

	return neighborsDf;	

def recommend(neighbors, testSet, k):

	columns = ['id_profissional', 'id_contratante', 'index', 'score'];
	scoreDf = pd.DataFrame(columns=columns);
	index = 0;

	for row1 in neighbors.itertuples():

		if(row1[4] == 0):
			score = (1/0.000000000000000001)/(row1[5]);
		else:
			score = (1/row1[4])/(row1[5]);	

		neighbors.set_value(row1.Index,'score', score);
		id_contratante = row1[2];
		
	for row1 in neighbors.itertuples():
		totalScore = neighbors.loc[neighbors['id_profissional'] == row1[1], 'score'].sum();	
		scoreDf.loc[len(scoreDf.index)] = [row1[1], row1[2], row1[3], totalScore];

	res = scoreDf.sort_values(['score'],ascending=False).head(3);

	testSet['ranking_1P'] = 0;
	testSet['ranking_2P'] = 0;
	testSet['ranking_3P'] = 0;
	testSet['score'] = 0.000000000000000000000;

	meanScore = res['score'].mean();	

	for row1 in testSet.itertuples():		
		if(row1[5] == id_contratante):	
			testSet.set_value(row1.Index,'score', meanScore);				
			testSet.set_value(row1.Index,'ranking_1P', res.iloc[0, 0]);	
			testSet.set_value(row1.Index,'ranking_2P', res.iloc[1, 0]);		
			testSet.set_value(row1.Index,'ranking_3P', res.iloc[2, 0]);							
			break;		

	return testSet.loc[(testSet['ranking_1P'] != 0) & testSet['id_contratante'].isin([row1[5]])];	

def random_prediction(recommendedDf, professionalDf):

	# Criando estrutura	que contem profissionais recomendados aleatoriamente
	columns = recommendedDf.dtypes.index;
	randomPredictionDf = pd.DataFrame(columns=columns);
	randomPredictionDf = randomPredictionDf.append(recommendedDf, ignore_index=True);

	idList = [];

	for row1 in randomPredictionDf.itertuples():
		for i in range(3):
			randomRowNumber = random.randrange(professionalDf.shape[0]);
			randomProfessionalId = professionalDf.iloc[randomRowNumber,0];	
			idList.append(randomProfessionalId);
		randomPredictionDf.set_value(row1.Index,'ranking_1P', idList[0]);	
		randomPredictionDf.set_value(row1.Index,'ranking_2P', idList[1]);
		randomPredictionDf.set_value(row1.Index,'ranking_3P', idList[2]);				
			
	return randomPredictionDf;

def professional_similarity(recommended, chosen):
	
	# Campos em que semelhanca tem resultado discreto
	sleepDistance = check_equality(recommended[8], chosen[8]);		
	stateDistance = check_equality(recommended[2], chosen[2]);		
	frequencyDistance = check_equality(recommended[6], chosen[6]);	
	cityDistance = check_equality(recommended[1], chosen[1]);			
				
	# Campos em que semelhanca tem resultado continuo
	salaryDistance = euclidean_distances(recommended[7], chosen[7]);			

	totalDistance = sleepDistance + stateDistance + frequencyDistance + cityDistance + salaryDistance;			

	return totalDistance;			

def calculate_accuracy_score(professionalDf, analisedDf):

	columns = ['id_profissional', 'distance'];
	similarityDf = pd.DataFrame(columns=columns);	
	distanceDf = pd.DataFrame(columns=columns);

	for row1 in analisedDf.itertuples():
		# Profissionais escolhidos pelo contratante	
		chosen1 = professionalDf[professionalDf.id_profissional.isin([row1[12]])];
		chosen2 = professionalDf[professionalDf.id_profissional.isin([row1[13]])];	
		chosen3 = professionalDf[professionalDf.id_profissional.isin([row1[14]])];

		for x in xrange(15,18):
			distance1 = professional_similarity(row1[x], chosen1);
			distance2 = professional_similarity(row1[x], chosen2);
			distance3 = professional_similarity(row1[x], chosen3);
			similarityDf.loc[len(similarityDf.index)] = [row1[x], distance1];
			similarityDf.loc[len(similarityDf.index)] = [row1[x], distance2];
			similarityDf.loc[len(similarityDf.index)] = [row1[x], distance3];

		res = similarityDf.sort_values(['distance'],ascending=True).head(k*3);	
		totalDistance = res.['distance']sum();	
		distanceDf.loc[len(similarityDf.index)] = [row1[5], totalDistance];

	return distanceDf;	
	
def check_recommendation(recommendedDf, professionalDf):
	# correct = 0;
	# for row1 in testSet.itertuples():
	# 	for x in xrange(15,18):
	# 		idChosen1 = row1[12];
	# 		idChosen2 = row1[13];
	# 		idChosen3 = row1[14];
			
	# 		chosen1 = professionalDf[professionalDf.id_profissional.isin([row1[12]])];
	# 		chosen2 = professionalDf[professionalDf.id_profissional.isin([row1[13]])];	
	# 		chosen3 = professionalDf[professionalDf.id_profissional.isin([row1[14]])];
	# 		predicted = professionalDf[professionalDf.id_profissional.isin([row1[x]])];

	# 		eucl1 = euclidean_distances(chosen1, predicted);
	# 		eucl2 = euclidean_distances(chosen2, predicted);
	# 		eucl3 = euclidean_distances(chosen3, predicted);
			
	# 		print(row1)
	# 		print(x)
			
	# 		print("1",eucl1)
	# 		print("2",eucl2)
	# 		print("3",eucl3)
	# 		if(eucl1 <= 2204160.00032757 or eucl2 <= 2204160.00032757 or eucl3 <= 2204160.00032757):
	# 			correct += 1;

	# return (correct/float(len(testSet)*3)) * 100.0		

	# Criando recomendacoes aleatorias
	randomDf = random_prediction(recommendedDf, professionalDf);

	# Calculando pontuacao para verificar acuracia da recomendacao
	randomDistanceDf = calculate_accuracy_score(professionalDf, randomDf);
	recommendedDistanceDf = calculate_accuracy_score(professionalDf, recommendedDf);		

# Obtendo dados dos usuarios
hirerDf, professionalDf = load_data();

# Separando em conjuntos de treinamento e de teste
trainingSet, testSet = train_test_split(hirerDf, test_size=0.5);

print("teste", testSet)
print("train", trainingSet)


# Obtendo os k contratantes mais parecidos
k = 3;

neighborsDf = knn(trainingSet, testSet, k);
print("neighbors-df", neighborsDf)

# Recomendando profissionais
columns = testSet.dtypes.index;
recommendedDf = pd.DataFrame(columns=columns);
recommendedDf['ranking_1P'] = 0;
recommendedDf['ranking_2P'] = 0;
recommendedDf['ranking_3P'] = 0;
recommendedDf['score'] = 0;

for row1 in neighborsDf.itertuples():	
	print("row1", row1[2])	
	candidates = neighborsDf.loc[neighborsDf['id_contratante'].isin([row1[2]])];
	if(candidates.empty == False):		
		result = recommend(candidates, testSet, k);		
		recommendedDf = recommendedDf.append(result, ignore_index=True);

# Removendo linhas duplicadas apos recomendacao
recommendedDf = recommendedDf.drop_duplicates(subset=['id_contratante'], keep='last');

print(recommendedDf)

# # Calculando acuracia do modelo
# accuracy_score = calculate_accuracy(recommendedDf, professionalDf);
# # print(accuracy_score)






