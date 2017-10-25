# loading libraries
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None  # default='warn'

def loading_data():
	# Obtendo dados dos usuarios
	names = ['cep', 'cidade', 'dormir', 'endereco', 'estado', 'experiencia', 'frequencia', 'id_cidade', 'id_contratante', 'id_segmento',
	 'latitude', 'longitude', 'nota_vaga', 'pretensao', 'id_profissional'];
	hirerDf = pd.read_csv('/home/elisa/TCC/user_hirer2.csv', header=None, names=names, delimiter=";");

	# Obtendo dados dos profissionais	
	names = ['id_profissional', 'cep', 'bairro', 'id_cidade', 'cidade', 'id_estado', 'estado', 'latitude', 
	'longitude', 'id_profissional_1',
	 'endereco', 'bairro_1', 'cep_1', 'latitude_1', 'longitude_1', 'id_cidade_1', 'nota_curriculo', 'avaliacao_geral'];
	professionalDf = pd.read_csv('/home/elisa/TCC/professional_profile.csv', header=None, names=names, delimiter=";");

	return hirerDf, professionalDf;

def data_preprocessing(hirerDf, professionalDf):
	# Removendo colunas que adicionam muitos atributos nulos e/ou que nao sao utilizadas
	hirerDf.drop('cep', axis=1, inplace=True);
	hirerDf.drop('endereco', axis=1, inplace=True);
	hirerDf.drop('experiencia', axis=1, inplace=True);
	hirerDf.drop('cidade', axis=1, inplace=True);	
	professionalDf.drop('id_profissional_1', axis=1, inplace=True);	
	professionalDf.drop('endereco', axis=1, inplace=True);	
	professionalDf.drop('bairro_1', axis=1, inplace=True);
	professionalDf.drop('cep_1', axis=1, inplace=True);
	professionalDf.drop('latitude_1', axis=1, inplace=True);
	professionalDf.drop('longitude_1', axis=1, inplace=True);
	professionalDf.drop('id_cidade_1', axis=1, inplace=True);

	# Removendo linhas que contem pelo menos um dos atributos nulo
	hirerDf = hirerDf.dropna();

	# Removendo linhas que nao sao do segmento de 'Empregada Domestica'
	hirerDf = hirerDf[hirerDf.id_segmento == 1];

	# Removendo linhas que relacionam um contratante mais de uma vez a um mesmo profissional
	hirerDf = hirerDf.drop_duplicates(subset=['id_contratante', 'id_profissional', 'id_segmento'], keep='first');

	# Adicionando colunas para serem armazenados os profissionais rankeados
	hirerDf['ranking_1'] = 'default';
	hirerDf['ranking_2'] = 'default';
	hirerDf['ranking_3'] = 'default';

	# Restringindo numero de amostras analisadas
	hirerDf.drop(hirerDf.index[902:], inplace=True)

	# Rankeando os profissionais com quem os contratantes tiveram contato
	for index, row1 in hirerDf.iterrows():
		value_list = [row1['id_contratante']]	
		data = hirerDf[hirerDf.id_contratante.isin(value_list)]
		if (len(data) >= 3):
			for row2 in hirerDf.itertuples():
				if (row2[5] == data.iloc[0, 4]):
					hirerDf.set_value(row2.Index,'ranking_1',(data.iloc[0,10]))
					hirerDf.set_value(row2.Index,'ranking_2',(data.iloc[1,10]))
					hirerDf.set_value(row2.Index,'ranking_3',(data.iloc[2,10]))

	# Removendo linhas que nao possuem profissionais rankeados				
	hirerDf = hirerDf[hirerDf.ranking_1 != 'default'];		

	# Tranformando campos nominais em numericos
	le = LabelEncoder()
	hirerDf['dormir'] = le.fit_transform(hirerDf['dormir'])
	hirerDf['frequencia'] = le.fit_transform(hirerDf['frequencia'])
	hirerDf['estado'] = le.fit_transform(hirerDf['estado'])
	professionalDf['estado'] = le.fit_transform(professionalDf['estado'])	
	professionalDf['bairro'] = le.fit_transform(professionalDf['estado'])
	professionalDf['cidade'] = le.fit_transform(professionalDf['cidade'])	

	# Removendo linhas duplicadas apos rankeamento
	hirerDf = hirerDf.drop_duplicates(subset=['id_contratante', 'ranking_1', 'ranking_2', 'ranking_3',
	 'id_segmento', 'id_cidade'], keep='last');	

	return hirerDf, professionalDf;

def knn(trainingSet, testSet, k):
	# Calculando similaridade entre contratantes 
	# distance = 10000000000000000000000000000;
	
	# testSet['ranking_1P'] = 0;
	# testSet['ranking_2P'] = 0;
	# testSet['ranking_3P'] = 0;
	# trainingSet['ranking_1P'] = 0;
	# trainingSet['ranking_2P'] = 0;
	# trainingSet['ranking_3P'] = 0;
	# for row1 in testSet.itertuples():
	# 	for row2 in trainingSet.itertuples():
	# 		if (row2[5] != row1[5]):		
	# 			eucl = euclidean_distances(row1, row2)
	# 			if (eucl[0, 0] < distance):
	# 				distance = eucl;
	# 				knn1 = row2;			
			
	# 	testSet.set_value(row1.Index,'ranking_1P',knn1[12]);			
	# 	testSet.set_value(row1.Index,'ranking_2P',knn1[13]);			
	# 	testSet.set_value(row1.Index,'ranking_3P',knn1[14]);

	columns = ['id_profissional', 'id_contratante', 'index', 'distancia', 'ranking', 'score'];
	distanceDf = pd.DataFrame(columns=columns);
	neighborsDf = pd.DataFrame(columns=columns);

	for row1 in testSet.itertuples():
		for row2 in trainingSet.itertuples():
			if (row2[5] != row1[5]):		
				eucl = euclidean_distances(row1, row2);
				distanceDf.loc[len(distanceDf.index)] = [row2[12], row1[5], row1.Index, eucl, 1, 0];
				distanceDf.loc[len(distanceDf.index)] = [row2[13], row1[5], row1.Index, eucl, 2, 0];
				distanceDf.loc[len(distanceDf.index)] = [row2[14], row1[5], row1.Index, eucl, 3, 0];				

		res = distanceDf.sort_values(['distancia'],ascending=True).head(k*3);			
		neighborsDf = neighborsDf.append(res, ignore_index=True);
		distanceDf.drop(distanceDf.index, inplace=True);

	return neighborsDf;

def recommend(neighbors, testSet, k):

	columns = ['id_profissional', 'id_contratante', 'index', 'score'];
	scoreDf = pd.DataFrame(columns=columns);
	index = 0;

	for row1 in neighbors.itertuples():
		score = (1/row1[4])/(row1[5]);
		neighbors.set_value(row1.Index,'score', score);
		id_contratante = row1[2];
		
	for row1 in neighbors.itertuples():
		totalScore = neighbors.loc[neighbors['id_profissional'] == row1[1], 'score'].sum();	
		scoreDf.loc[len(scoreDf.index)] = [row1[1], row1[2], row1[3], totalScore];

	# for row1 in scoreDf.itertuples():
	# 	maxScore = scoreDf.groupby(['id_profissional'])['score'].max();
	# 	maxRow = scoreDf.loc[scoreDf['id_profissional'].isin([row1[1]]) & scoreDf['score'].isin([row1[5]])];

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

	print("in", testSet.loc[(testSet['ranking_1P'] != 0) & testSet['id_contratante'].isin([row1[5]])])		

	return testSet.loc[(testSet['ranking_1P'] != 0) & testSet['id_contratante'].isin([row1[5]])];	

def calculate_accuracy(recommendedDf):
	correct = 0;
	for row1 in testSet.itertuples():
		for x in xrange(15,18):
			idChosen1 = row1[12];
			idChosen2 = row1[13];
			idChosen3 = row1[14];
			
			chosen1 = professionalDf[professionalDf.id_profissional.isin([row1[12]])];
			chosen2 = professionalDf[professionalDf.id_profissional.isin([row1[13]])];	
			chosen3 = professionalDf[professionalDf.id_profissional.isin([row1[14]])];
			predicted = professionalDf[professionalDf.id_profissional.isin([row1[x]])];

			eucl1 = euclidean_distances(chosen1, predicted);
			eucl2 = euclidean_distances(chosen2, predicted);
			eucl3 = euclidean_distances(chosen3, predicted);
			
			print(row1)
			print(x)
			
			print("1",eucl1)
			print("2",eucl2)
			print("3",eucl3)
			if(eucl1 <= 2204160.00032757 or eucl2 <= 2204160.00032757 or eucl3 <= 2204160.00032757):
				correct += 1;

	return (correct/float(len(testSet)*3)) * 100.0		

# Inicializando dados do contratante
hirerDf, professionalDf = loading_data();
hirerDf, professionalDf = data_preprocessing(hirerDf, professionalDf);

# Separando em conjuntos de treinamento e de teste
trainingSet, testSet = train_test_split(hirerDf, test_size=0.5);
print("dasdasdasdasdadasdada", testSet)

# Obtendo os k contratantes mais parecidos
k = 3;
neighborsDf = knn(trainingSet, testSet, k);
print("neighbors", neighborsDf)

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
		print("cand", candidates)
		result = recommend(candidates, testSet, k);
		print("result", result)
		recommendedDf = recommendedDf.append(result, ignore_index=True);

# Removendo linhas duplicadas apos recomendacao
recommendedDf = recommendedDf.drop_duplicates(subset=['id_contratante'], keep='last');

print(recommendedDf)


# Calculando acuracia do modelo
# accuracy_score = calculate_accuracy(testSet);
# print(accuracy_score)

# 19 e 19: 33305
# 19 e 77: 497580.48079708
# 77 e 77: 212315.
# 19 e 179: 447768.083458
# 77 e 179: 400082.51791658

# 0 e 0: 0
# 0 e 1: 2325959.07127039
# 0 e 2: 4132930.00631123
# 1 e 2: 1807014.54912182
# 0 e 13: 2204160.00032757
# 0 e 6: 685970.00029159
# 0 e 5: 10256000.52114086






