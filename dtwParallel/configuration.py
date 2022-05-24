import configparser


def create_file_ini():
	config = configparser.ConfigParser()
	config['DEFAULT'] = {'errors_control': 'True',
	                     'distance': "euclidean",
						 'type_dtw': 'd',
						 'MTS': 'False',
						 'n_threads': -1,
						 'visualization': 'False',
						 'output_file': 'False',
						 'DTW_to_kernel': 'True',
						 'sigma': 1}
						 

	with open('configuration.ini', "w") as config_file:
		config.write(config_file)


