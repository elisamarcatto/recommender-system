# loading libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None  # default='warn'

def loading_data():
	# Obtendo dados dos usuarios
	names = ['cep', 'cidade', 'dormir', 'endereco', 'estado', 'experiencia', 'frequencia', 'id_cidade', 'id_contratante', 'id_segmento',
	 'latitude', 'longitude', 'nota_vaga', 'pretensao', 'id_profissional'];
	hirerDf = pd.read_csv('/home/elisa/TCC/user_hirer2.csv', header=None, names=names, delimiter=";");

	# Obtendo dados dos profissionais	
	names = ['id_profissional', 'cep', 'bairro', 'id_cidade', 'cidade', 'id_estado', 'estado', 'latitude', 
	'longitude', 'endereco', 'nota_curriculo', 'avaliacao_geral', 'id_segmento', 'frequencia', 'pretensao', 'pretensao_dormir',
	'dormir'];
	professionalDf = pd.read_csv('/home/elisa/TCC/professional_profile2.csv', header=None, names=names, delimiter=";");

	return hirerDf, professionalDf;

def data_preprocessing(hirerDf, professionalDf):

	print("entrou", len(hirerDf));

	# Removendo colunas que adicionam muitos atributos nulos e/ou que nao sao utilizadas
	hirerDf.drop('cep', axis=1, inplace=True);
	hirerDf.drop('cidade', axis=1, inplace=True);
	hirerDf.drop('endereco', axis=1, inplace=True);
	hirerDf.drop('experiencia', axis=1, inplace=True);

	print("entrou2", len(hirerDf));

	professionalDf.drop('cep', axis=1, inplace=True);
	professionalDf.drop('bairro', axis=1, inplace=True);
	professionalDf.drop('cidade', axis=1, inplace=True);
	professionalDf.drop('id_estado', axis=1, inplace=True);
	professionalDf.drop('latitude', axis=1, inplace=True);
	professionalDf.drop('longitude', axis=1, inplace=True);
	professionalDf.drop('endereco', axis=1, inplace=True);	
	professionalDf.drop('pretensao_dormir', axis=1, inplace=True);

	# Removendo linhas que contem pelo menos um dos atributos nulo
	hirerDf = hirerDf.dropna();

	print("entrou3", len(hirerDf));

	professionalDf = professionalDf.dropna();

	# Removendo linhas que nao sao do segmento de 'Empregada Domestica'
	hirerDf = hirerDf[hirerDf.id_segmento == 1];
	professionalDf = professionalDf[professionalDf.id_segmento == 1];

	print("entrou4", len(hirerDf));

	# Removendo linhas que relacionam um contratante mais de uma vez a um mesmo profissional
	hirerDf = hirerDf.drop_duplicates(subset=['id_contratante', 'id_profissional', 'id_segmento'], keep='first');

	print("entrou5", len(hirerDf));	

	# Restringindo numero de amostras analisadas
	hirerDf.drop(hirerDf.index[4002:], inplace=True)
	

	for row1 in hirerDf.itertuples():			
		if(professionalDf[professionalDf.id_profissional.isin([row1[11]])].empty == True):
			hirerDf = hirerDf[hirerDf.id_profissional != row1[11]];	

	# Adicionando colunas para serem armazenados os profissionais rankeados
	hirerDf['ranking_1'] = 'default';
	hirerDf['ranking_2'] = 'default';
	hirerDf['ranking_3'] = 'default';		

	print("entrou6", len(hirerDf));
	
	for row1 in hirerDf.itertuples():
		print("id", row1[5])
		value_list = [row1[5]]	
		data = hirerDf[hirerDf.id_contratante.isin(value_list)];
		if (len(data) >= 3):			
			if (row1[5] == data.iloc[0, 4]):
				print("data", data)
				print "\n\n\n";
				hirerDf.set_value(row1.Index,'ranking_1',(data.iloc[0,10]))
				hirerDf.set_value(row1.Index,'ranking_2',(data.iloc[1,10]))
				hirerDf.set_value(row1.Index,'ranking_3',(data.iloc[2,10]))

	# Removendo linhas que nao possuem profissionais rankeados

	print("entrou7", len(hirerDf));

	hirerDf = hirerDf[hirerDf.ranking_1 != 'default'];		

	print("entrou8", len(hirerDf));

	# Tranformando campos nominais em numericos
	le = LabelEncoder()
	hirerDf['dormir'] = le.fit_transform(hirerDf['dormir'])	
	hirerDf['estado'] = le.fit_transform(hirerDf['estado'])
	hirerDf['frequencia'] = le.fit_transform(hirerDf['frequencia'])
	professionalDf['estado'] = le.fit_transform(professionalDf['estado'])	
	professionalDf['frequencia'] = le.fit_transform(professionalDf['frequencia'])	

	print("entrou9", len(hirerDf));
	
	# Removendo linhas duplicadas apos rankeamento
	hirerDf = hirerDf.drop_duplicates(subset=['id_contratante', 'ranking_1', 'ranking_2', 'ranking_3',
	 'id_segmento', 'id_cidade'], keep='last');	

	print("entrou10", len(hirerDf));

	return hirerDf, professionalDf;

# Inicializando dados do contratante
hirerDf, professionalDf = loading_data();
# print(len(professionalDf)) 396944
# print(len(hirerDf)) 349845

hirerDf, professionalDf = data_preprocessing(hirerDf, professionalDf);

# # # print(hirerDf.head())
# # # print(professionalDf.head())	

# print("hirer", hirerDf.head())
# print("n_hirer", len(hirerDf))
# print("n_col", hirerDf.shape[1])

# print("prof", professionalDf)
# print("n_prof", len(professionalDf))
# print("n_col", professionalDf.shape[1])

hirerDf.to_csv('cc_hirer-extra.csv');
professionalDf.to_csv('cc_professional1-extra.csv');