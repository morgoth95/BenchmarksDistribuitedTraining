all: install_prerequisites

install_prerequisites:
	if [ ! -d "gpt-2-Pytorch" ]; then git clone https://github.com/graykode/gpt-2-Pytorch.git gpt-2-Pytorch; fi;
	cd gpt-2-Pytorch && pip3 install -r requirements.txt
	pip3 install transformers
	pip3 install datasets
