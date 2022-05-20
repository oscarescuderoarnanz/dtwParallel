import configparser


def create_file_ini():
	config = configparser.ConfigParser()
	config['DEFAULT'] = {'errors_control': 'True',
						 'type_dtw': 'd',
						 'MTS': 'False',
						 'verbose': 0,
						 'distance': "euclidean",
						 'n_threads': 1,
						 'visualization': 'False',
						 'output_file': 'False'}
						 

	with open('configuration.ini', "w") as config_file:
		config.write(config_file)


