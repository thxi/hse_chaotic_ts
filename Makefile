command:
	cat Makefile

jupyter:
	jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password='' --no-browser

data: FORCE
	mkdir -p data
	cd data && kaggle competitions download -c ieee-fraud-detection
	cd data && unzip -o ieee-fraud-detection.zip

FORCE: ;

