Here we have equivalend pytorch code

## Requirements:

	virtualenv -p /usr/bin/python3 ../venv
	source ../venv/bin/activate
	pip install torch							# 1.5 and above

## Results:

N=200 K=64 T=514

	| Hardware  | time of 30 iterations | 
	| :---:     | :---:   | 
	| cpu       | 553.91s | 
	| Tesla M40 |  68.87s | 
