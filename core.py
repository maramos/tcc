import numpy as np
import re
from keras.models import Sequential
from keras.models import load_model, save_model
from keras.layers import Dense, LSTM
from keras.initializers import ones
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
%matplotlib gtk

def useLSTM(X, y, epochs=30, verbose=0, prop=60, amostras=90, show_values=False):
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
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'], kernel_initializer='ones')
	
	print X_train.shape
	saida = model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=verbose)

	Treinamento = pyplot.plot([x for x in range(len(X_train))], [x[0] for x in X_train], label='Treinamento')
	Treinamento2 = pyplot.plot([x for x in range(len(X_train))], model.predict(X_train), label='Treinamento-AI')
	Teste = pyplot.plot(np.array([x for x in range(len(y_test))])+prop, y_test, label='Teste')
	Predicao = pyplot.plot(np.array([x for x in range(len(X_test))])+prop, model.predict(X_test), label='Predicao')
	pyplot.legend()

	model.save(path + 'last_run.h5')

	vX_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
	print list((diffDados(vX_test.T[0])[1] == diffDados(model.predict(X_test))[1]).astype('int')).count(1) * 100 / X_test.shape[0]

	if show_values == True:
		for j, i in enumerate(X_test):
			print i, y_test[j], model.predict(np.array([i]))[0]

	return saida

def useDelay(X, y, epochs=30, verbose=0, prop=60, amostras=90, show_values=False):
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
	model.add(Dense(10, activation='relu', input_dim=X.shape[1]))
	model.add(Dense(1))

	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'], kernel_initializer='ones')
	saida = model.fit(X_train, y_train, batch_size=16, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test))

	Treinamento1 = pyplot.plot([x for x in range(len(X_train))], [x[0] for x in X_train], label='Treinamento')
	Treinamento2 = pyplot.plot([x for x in range(len(X_train))], model.predict(X_train), label='Treinamento-AI')
	Teste = pyplot.plot(np.array([x for x in range(len(y_test))])+prop, y_test, label='Teste')
	Predicao = pyplot.plot(np.array([x for x in range(len(X_test))])+prop, model.predict(X_test), label='Predicao')
	pyplot.legend()

	model.save(path + 'last_run.h5')

	print list((diffDados(X_test.T[0])[1] == diffDados(model.predict(X_test))[1]).astype('int')).count(1) * 100 / X_test.shape[0]

	if show_values == True:
		for j, i in enumerate(X_test):
			print i, y_test[j], model.predict(np.array([i]))[0]

	return saida

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
	dataX, dataY = [], []
	for i in range(dataset.shape[0]-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back, [field_resp]])
	return np.concatenate(np.array(dataX).T).T, np.concatenate(dataY), diffDados(dataset[1])

def diffDados(dataset):
	diffInicio = [0]
	tanH = [0]
	for i in range(len(dataset)-1):
		i += 1
		diffInicio.append(dataset[i] - dataset[i-1])

	return [np.array(diffInicio), (np.array(diffInicio) >= 0).astype('float32')]

if __name__ == '__main__':
	path = '/home/maramos/Documentos/estudar/tcc/dados/'
	histDolar = open(path + 'histDolar.csv').readlines()
	histJuros = open(path + 'histJuros.csv').readlines()
	np.random.seed(7)

	bovespa = parseDados('BVMF3')
	result = parseDados('BBDC3', correlated=bovespa)

	look_back=2
	features = [1,2,3,6,7,8,9,10,11]
	X, y, [ds_diff, ds_qbin] = create_dataset(result, features, look_back=look_back)

	params = {"verbose": 2, "show_values": False, "epochs": 10, "prop": 60}

	saida_delay = useDelay(X, y, **params)
	saida_LSTM = useLSTM(X, y, **params)

TODO: 
	pre-processing (nao supervizionado)
	cluster no qbin
	qbin como param
	predizer abs(diferença) + qbin