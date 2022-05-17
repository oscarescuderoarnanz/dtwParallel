import configparser

config = configparser.ConfigParser()
config['DEFAULT'] = {'errors_control': 'True',
					 'type_dtw': 'd',
					 'MTS': 'False',
					 'verbose': 0,
					 'n_threads': 1,
					 'visualization': 'False',
					 'output_fie': 'True'}

with open('configuration.ini', 'w') as configfile:
	config.write(configfile)
