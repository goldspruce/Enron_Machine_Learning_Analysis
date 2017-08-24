from sklearn.preprocessing import MinMaxScaler

sklearn.preprocessing.MinMaxScaler
import numpy
weights = numpy.array([[115.],140.],[175.])
scale = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)

scaler = MinMaxScaler()

features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test) 