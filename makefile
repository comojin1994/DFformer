start_docker:
	cd docker && docker-compose up -d

setup:
	cd setups && python setup.py

preprocess_bci2a:
	python setups/preprocess_bci2a.py

preprocess_bci2b:
	python setups/preprocess_bci2b.py

preprocess_sleepedf:
	python setups/preprocess_sleepedf.py

preprocess_shhs:
	python setups/preprocess_shhs.py

bci2a:
	nohup python training_bci2a.py > logs/output_bci2a.log 2> logs/error_bci2a.log &

bci2b:
	nohup python training_bci2b.py > logs/output_bci2b.log 2> logs/error_bci2b.log &

shhs:
	nohup python training_shhs.py > logs/output_shhs.log 2> logs/error_shhs.log &

sleepedf:
	nohup python training_sleepedf.py > logs/output_sleepedf.log 2> logs/error_sleepedf.log &

autoencoder:
	nohup python training_autoencoder.py > logs/output_autoencoder.log 2> logs/error_autoencoder.log &