import numpy as np
import re
from keras.models import Sequential
from keras.models import load_model, save_model
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.initializers import ones
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
%matplotlib

def userClassifier(X, y, epochs=30, verbose=0, prop=60, amostras=90, fold=10, n_money=10000.0, n_papel=0.0):
	amostras = len(X) * amostras / 100
	prop = amostras * prop / 100
	X = np.array(X[:amostras]).astype('float32')
	y = np.array(y[:amostras]).astype('float32')
	X_train, X_test, y_train, y_test = X[:prop], X[prop:], y[:prop], y[prop:]

	model = Sequential()
	model.add(Dense(12, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
	model.add(Dense(8,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	fited_model = model.fit(X_train, y_train, epochs=epochs, batch_size=2, verbose=verbose)
	model_output = model.predict(X_test)

	print model.evaluate(X_test, y_test)

	return model_output, X_test, y_test, model

def useLSTM(X, y, epochs=30, verbose=0, prop=60, amostras=90, fold=10, n_money=10000.0, n_papel=0.0, plotInfos=True):
	amostras = len(X) * amostras / 100
	prop = amostras * prop / 100

	X = np.array(X[:amostras]).astype('float32')
	y = np.array(y[:amostras]).astype('float32')

	X = X.reshape(X.shape[0], X.shape[-1])

	scaler = MinMaxScaler(feature_range=(0, 1))
	X = scaler.fit_transform(X)
	y = scaler.fit_transform(y.reshape(-1, 1))

	X_train, X_test, y_train, y_test = X[:prop], X[prop:], y[:prop], y[prop:]
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

	model = Sequential()
	model.add(LSTM(10, input_shape=(X_train.shape[1], 1)))
	model.add(Dropout(0.3))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'], kernel_initializer='ones')
	fited_model = model.fit(X_train, y_train, epochs=epochs, batch_size=2, verbose=verbose)
	model_train = scaler.inverse_transform(model.predict(X_train).reshape(-1, 1))
	model_output = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1))
	model.save(path + 'last_run.h5')

	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

	aux = []
	bin_mean = (list((diffDados(y_test)[1] == diffDados(model_output)[1].T).astype('int')[0]).count(1) * 100) / float(X_test.shape[0]-1)
	delta_mean = delta(diffDados(scaler.inverse_transform(y_test.reshape(-1, 1)))[0], diffDados(model_output)[0])
	a, grid = kfoldDiff(fold, diffDados(X_test.T[0])[1])
	b, _ = kfoldDiff(fold, diffDados(model_output)[1])
	for i in range(fold):
		aux.append((list((np.array(a[i]) == np.array(b[i])).astype('int')[0]).count(1) * 100) / float(len(a[i])))

	fn_money, fn_papel = teste_gerencia(y_test, model_output, scaler, n_money, n_papel)
	max_money, max_papel = teste_gerencia(y_test, scaler.inverse_transform(y_test), scaler, n_money, n_papel)	

	print 'Folds: ', aux
	print 'Dias: ', X_test.shape[0]
	print u'Assertividade:\n\tdelta: ', delta_mean, u'\n\tbin: ', bin_mean
	print u'Dinheiro:\n\tinicio: {}\n\ttermino: {}\nPapel:\n\ttermino: {}\n'.format(n_money, fn_money[-1], fn_papel[-1])
	print u'Max: dinheiro: {}'.format(max_money[-1])

	if plotInfos == True:
		plot_infos(X_train, X_test, y_train, y_test, model_output, model_train, fn_money, fn_papel, scaler)

	return {"model": fited_model, "money": fn_money}

def useDelay(X, y, epochs=10, verbose=0, prop=60, amostras=90, fold=10, n_money=10000.0, n_papel=0.0, plotInfos=True, showDetails=True):
	amostras = len(X) * amostras / 100
	prop = amostras * prop / 100

	X = np.array(X[:amostras])
	y = np.array(y[:amostras])
	X = X.reshape(X.shape[0], X.shape[-1])

	scaler = MinMaxScaler(feature_range=(0, 1))
	X = scaler.fit_transform(X)
	y = scaler.fit_transform(y.reshape(-1, 1))

	X_train, X_test, y_train, y_test = X[:prop], X[prop-1:], y[:prop], y[prop-1:]

	model = Sequential()
	model.add(Dense(10, activation='tanh', input_dim=X.shape[1]))
	model.add(Dropout(0.3))
	model.add(Dense(1))

	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'], kernel_initializer='ones', bias_initializer='ones', seed=7)
	fited_model = model.fit(X_train, y_train, batch_size=8, epochs=epochs, verbose=verbose)
	model_train = scaler.inverse_transform(model.predict(X_train).reshape(-1, 1))
	model_output = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1))
	model.save(path + 'last_run.h5')

	fn_money, fn_papel = teste_gerencia(y_test, model_output, scaler, n_money, n_papel, diffDados(model_output)[1])
	max_money, max_papel = teste_gerencia(y_test, scaler.inverse_transform(y_test), scaler, n_money, n_papel, diffDados(y_test)[1])	

	if showDetails == True:
		aux = []
		bin_mean = (list((diffDados(y_test)[1] == diffDados(model_output)[1].T).astype('int')[0]).count(1) * 100) / float(X_test.shape[0]-1)
		delta_mean = delta(diffDados(scaler.inverse_transform(y_test.reshape(-1, 1)))[0], diffDados(model_output)[0])
		a, grid = kfoldDiff(fold, diffDados(X_test.T[0])[1])
		b, _ = kfoldDiff(fold, diffDados(model_output)[1])
		for i in range(fold):
			tdiff = (np.array(a[i]) == np.array(b[i])).astype('int')[0]
			aux.append((list(tdiff).count(1) * 100) / float(len(a[i])))
			#print 'fold[{}]: {}'.format(i, np.array(a[i]) == np.array(b[i]))
			print 'fold[{}]: {}'.format(i, aux[-1])

		print 'Folds: ', aux
		print 'Dias: ', X_test.shape[0]
		print u'Assertividade:\n\tdelta: ', delta_mean, u'\n\tbin: ', bin_mean
		print u'Dinheiro:\n\tinicio: {}\n\ttermino: {}\nPapel:\n\ttermino: {}\n'.format(n_money, fn_money[-1], fn_papel[-1])
		print u'Max: dinheiro: {}'.format(max_money[-1])

	if plotInfos == True:
		plot_infos(X_train, X_test, y_train, y_test, model_output, model_train, fn_money, fn_papel, scaler)

	return {"model": fited_model, "money": fn_money}

def plot_infos(X_train, X_test, y_train, y_test, model_output, model_train, n_money, n_papel, scaler):
	fig = pyplot.figure()
	treino = fig.add_subplot(311)
	teste = fig.add_subplot(312)
	money = fig.add_subplot(325)
	papel = fig.add_subplot(326)

	Treinamento1 = treino.plot(scaler.inverse_transform(model_train), 'ro-', label='Treinamento')
	Treinamento2 = treino.plot(scaler.inverse_transform(model_train.reshape(-1,1)), label='Treinamento-AI')
	Teste = teste.plot(scaler.inverse_transform(y_test).reshape(-1, 1), label='Teste')
	Predicao = teste.plot(model_output, label='Predicao')
	teste.legend()
	treino.legend()
	money.plot(np.array(n_money), label='money')
	money.legend()
	papel.plot(np.array(n_papel), label='papel')
	papel.legend()

def venda(n_papel, valor_p):
	n_money = n_papel * valor_p
	#n_money = n_money / 1.0325 # taxa
	#print 'Venda: {} * {} = '.format(n_papel, valor_p), n_papel * valor_p
	return n_money, 0.0

def compra(n_money, valor_p):
	n_papel = n_money / float(valor_p)
	#print 'Compra: {} / {} = '.format(n_money, valor_p), n_money / float(valor_p)
	return n_papel, 0.0

def teste_gerencia(y_test, model_output, scaler, n_money, n_papel, aux):
	output = [[],[]]
	for i in range(len(y_test)-2):
		t0 = scaler.inverse_transform(y_test[i:i+1])[0][0] / 100.0
		t1 = model_output[i+1:i+2][0][0] / 100.0
		dt10 = (scaler.inverse_transform(y_test[i:i+1])[0][0] - model_output[i+1:i+2][0][0]) / 100.0
		dt11 = aux[i+1]
		if dt11 < 1 and n_papel > 0.0:
			n_money, n_papel = venda(n_papel, t0)
			n_money = n_money/1.025
			output[0].append(n_money)
		elif dt11 > 0 and n_money > 0.0:
			n_papel, n_money = compra(n_money, t0)
			output[1].append(n_papel)
	return output

def parseDados(papel, correlated=None):
    X, y = [], []
    result = [[],[],[],[],[],[],[],[],[]]
    pattern = re.compile('{}[^a-zA-Z0-9]\s*'.format(papel))

    with open(path + "dados-2008.csv") as csvfile:
        for i in csvfile:
            if pattern.match(i.split(";")[2]) != None:
                data,_,_,tipoM,inicio,vmax,vmin,_,_,_,_,NumNegPreg,NumNegMerc,VolNegMerc,_,_,_ = i.split(";")
                result[0].append(int(data))
                result[1].append(int(inicio))
                result[2].append(int(vmax))
                result[3].append(int(vmin))
                result[4].append(int(NumNegPreg))
                result[5].append(int(NumNegMerc))
                result[6].append(int(VolNegMerc.replace('\n','')))
                result[7].append(valor_juros(result[0][-1]))
                result[8].append(dolar(result[0][-1]))
                try:
                	c_inicio, c_max, c_min = qcorrelated(result[0][-1], correlated)
                	if len(result[0]) == 1:
                		result.append([])
                		result.append([])
                		result.append([])
                	result[9].append(c_inicio)
                	result[10].append(c_max)
                	result[11].append(c_min)
                except Exception as e:
                	pass
    return result

def dolar(qd):
    try:
        vDolar = histDolar[[w[0] for w in [x.split(";") for x in histDolar]].index(str(qd))].split(";")[1]
    except:
        return False
    return float(vDolar.replace('\n','').replace(',','.'))

def qcorrelated(qd, correlated):
	ar = min(filter(lambda x: x <= int(qd), correlated[0]), key=lambda x:abs(x-int(qd)))
	return np.array(correlated)[[1,2,3],correlated[0].index(ar)]

def data_juros(qd):
	return min(filter(lambda x: x <= int(qd), [int(x.split(";")[0]) for x in histJuros]),key=lambda x:abs(x-int(qd)))

def valor_juros(qd):
	return float(histJuros[[int(x.split(";")[0]) for x in histJuros].index(data_juros(qd))].split(";")[-1].replace('\n',''))

def create_dataset(dataset, fields, look_back=1, field_resp=0):
	dataset = np.array(dataset).T[0:,fields]
	dataX, dataY, diffInicio = [], [], [0]
	for i in range(dataset.shape[0]-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back, [field_resp]])
	X, y = np.concatenate(np.array(dataX).T).T, np.concatenate(dataY)
	dataset = dataset[:,field_resp]
	for i in range(dataset.shape[0]-2-look_back):
		i += 1
		diffInicio.append(dataset[i] - dataset[i-1])
	return X, y, [np.array(diffInicio), (np.array(diffInicio) >= 0).astype('float32')]

def diffDados(dataset):
	diffInicio = []
	for i in range(dataset.shape[0]-1):
		i += 1
		diffInicio.append(dataset[i] - dataset[i-1])
	return [np.array(diffInicio), (np.array(diffInicio) >= 0).astype('float32')]

def kfoldDiff(num, qbin):
	arr, aux = [], []
	for i in range(num):
		arr.append(int((len(qbin)/float(num)) * ((i+1))))
	aux.append(qbin[:arr[0]])
	grid = int((len(qbin)/float(num)) * ((i+1)))
	for i in range(len(arr)-1):
		aux.append(qbin[arr[i]:arr[i+1]])
	aux.append(qbin[i+1:])
	return aux, grid

def delta(v1, v2):
	aux = []
	for i,j in zip(v1, v2):
		j,i = sorted([abs(i+0.000000001),abs(j+0.000000001)])
		aux.append((j*100)/float(i))
	return np.mean(np.array(aux)[~np.isnan(aux)])

if __name__ == '__main__':
	path = '/home/maramos/Documentos/estudar/tcc/dados/'
	histDolar = open(path + 'histDolar.csv').readlines()
	histJuros = open(path + 'histJuros.csv').readlines()
	np.random.seed(7)

	#bovespa = parseDados('BVMF3')
	bradesco = parseDados('BBDC3')
	result = parseDados('ITUB4', correlated=bradesco)

	look_back=1
	features = [1,2,3,6,7,8,9,10,11]
	X, y, [ds_diff, ds_qbin] = create_dataset(result, features, look_back=look_back)
	Xv = np.concatenate((X.T, ds_qbin.reshape(ds_qbin.shape[0], 1).T, ds_diff.reshape(ds_diff.shape[0], 1).T)).T

	X_bin, y_bin, [_, _] = create_dataset(ds_qbin.reshape(1,-1), [0], look_back=look_back)

	params = {"verbose": False, "epochs": 50, "prop": 96, "fold": 10, "plotInfos": False, "showDetails": False}

	saida_delay = useDelay(Xv, y, **params)
	saida_LSTM = useLSTM(X, y, **params)
	saida_classifier = userClassifier(X, y_bin.reshape(-1,1), **params)

def test():
	aux = [[],[]]
	for i in range(10):
	    try:
	        aux[0].append(useDelay(X, y, **params)['money'][-1])
	        aux[1].append(useDelay(Xv, y, **params)['money'][-1])
	    except Exception as e:
	        print str(e)
	return aux
