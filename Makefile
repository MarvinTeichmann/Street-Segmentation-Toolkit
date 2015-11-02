ldocs:
	python setup.py upload_docs --upload-dir docs/_build/html

localinstall:
	python setup.py install --user
	rm /home/mteichmann/.local/lib/python2.7/site-packages/site.py
	rm /home/mteichmann/.local/lib/python2.7/site-packages/site.pyc

test:
	nosetests --with-coverage --cover-erase --cover-package sst --logging-level=INFO --cover-html

testall:
	make test
	cheesecake_index -n sst -v

count:
	cloc . --exclude-dir=docs,cover,dist,sst.egg-info

countc:
	cloc . --exclude-dir=docs,cover,dist,sst.egg-info,tests

countt:
	cloc tests

clean:
	rm -f *.hdf5 *.yml *.csv
	find . -name "*.pyc" -exec rm -rf {} \;
	find . -type d -name "__pycache__" -delete
	rm -rf build
	rm -rf cover
	rm -rf dist
	rm -rf sst.egg-info
	rm -rf docs/build
