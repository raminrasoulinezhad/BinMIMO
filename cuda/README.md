Main files are:

	GAMP_cpuFuncs.cu
	GAMP_gpuFuncs.cu
	GAMP_main.cu

## Ramin:
	cd cuda
	make all
	
## Results

ProfPhillet:

	Device Name: Tesla M40
	shared mem size: 65536
	N=200 K=64 T=514	 CPU=288.897 ms GPU=63.923 ms GPU-kernel=3.08845 ms mse=29884977152.000000

DrFish:

	Device Name: GeForce RTX 2080 Ti
	shared mem size: 65536
	N=200 K=64 T=514   	CPU=224.522 ms GPU=235.301 ms GPU-kernel=1.77885 ms mse=31134828544.000000
	N=200 K=64 T=514	CPU=224.939 ms GPU=226.591 ms GPU-kernel=1.78445 ms mse=29884977152.000000

guppy:

	Device Name: GeForce GTX 1080 Ti
	shared mem size: 65536
	N=200 K=64 T=514	 CPU=191.369 ms GPU=43.056 ms GPU-kernel=1.8847 ms mse=29884977152.000000
	N=200 K=64 T=514	 CPU=191.497 ms GPU=46.695 ms GPU-kernel=1.88506 ms mse=29884977152.000000

Ramin Thinkpad:
	
	Device Name: GeForce 940MX
	shared mem size: 65536
	N=200 K=64 T=514	 CPU=270.094 ms GPU=36.585 ms GPU-kernel=5.70739 ms mse=29884977152.000000



# To Use www.vast.ai
First, creat your machine instance. Then, using the dashbord, open a termianl. Then write these in your terminal:

	apt install git-all

	git clone https://github.com/raminrasoulinezhad/BinMIMO.git
	cd BinMIMO/cuda 
	make all


## results:

Device Name: GeForce RTX 3090
shared mem size: 65536
	
	N=200 K=64 T=514         CPU=365.083 ms GPU=489.842 ms GPU-kernel=1.49485 ms mse=inf
	N=200 K=64 T=514         CPU=380.511 ms GPU=284.424 ms GPU-kernel=1.49901 ms mse=29882308608.000000
	N=200 K=64 T=514         CPU=367.71 ms GPU=295.088 ms GPU-kernel=1.49693 ms mse=29885755392.000000
	N=200 K=64 T=514         CPU=368.674 ms GPU=291.052 ms GPU-kernel=1.49878 ms mse=501121885929472.000000

	N=200 K=64 T=514         CPU=367.411 ms GPU=454.682 ms GPU-kernel=1.4985 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=366.165 ms GPU=294.635 ms GPU-kernel=1.4999 ms mse=inf
	N=200 K=64 T=514         CPU=371.516 ms GPU=297.891 ms GPU-kernel=1.49782 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=365.732 ms GPU=282.147 ms GPU-kernel=1.50298 ms mse=470025329534758226268394451107840.000000
	N=200 K=64 T=514         CPU=365.335 ms GPU=291.913 ms GPU-kernel=1.49942 ms mse=20391455443584503173522800689148854272.000000
	N=200 K=64 T=514         CPU=364.738 ms GPU=306.611 ms GPU-kernel=1.49875 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=360.272 ms GPU=286.261 ms GPU-kernel=1.4976 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=360.848 ms GPU=291.365 ms GPU-kernel=1.49952 ms mse=inf
	N=200 K=64 T=514         CPU=358.223 ms GPU=284.533 ms GPU-kernel=1.49645 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=413.722 ms GPU=290.419 ms GPU-kernel=1.49709 ms mse=inf
	N=200 K=64 T=514         CPU=382.691 ms GPU=284.424 ms GPU-kernel=1.4999 ms mse=-nan
	N=200 K=64 T=514         CPU=368.956 ms GPU=287.452 ms GPU-kernel=1.49981 ms mse=29884950528.000000
	N=200 K=64 T=514         CPU=363.993 ms GPU=285.644 ms GPU-kernel=1.50112 ms mse=29884977152.000000

Device Name: GeForce RTX 2080 Ti
shared mem size: 65536

	N=200 K=64 T=514         CPU=226.251 ms GPU=206.616 ms GPU-kernel=1.8463 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=226.872 ms GPU=85.024 ms GPU-kernel=1.8495 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=216.982 ms GPU=89.986 ms GPU-kernel=1.84941 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=209.694 ms GPU=77.416 ms GPU-kernel=1.84944 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=220.073 ms GPU=79.72 ms GPU-kernel=1.84752 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=226.325 ms GPU=87.622 ms GPU-kernel=1.8497 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=220.989 ms GPU=76.724 ms GPU-kernel=1.8424 ms mse=29884977152.000000

Device Name: GeForce RTX 3080
shared mem size: 65536

	N=200 K=64 T=514         CPU=274.125 ms GPU=184.778 ms GPU-kernel=1.41997 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=273.692 ms GPU=62.443 ms GPU-kernel=1.42691 ms mse=inf
	N=200 K=64 T=514         CPU=274.203 ms GPU=64.259 ms GPU-kernel=1.42896 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=274.221 ms GPU=63.069 ms GPU-kernel=1.4185 ms mse=inf
	N=200 K=64 T=514         CPU=273.713 ms GPU=64.817 ms GPU-kernel=1.41994 ms mse=29884743680.000000
	N=200 K=64 T=514         CPU=273.308 ms GPU=57.849 ms GPU-kernel=1.42061 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=274.86 ms GPU=59.95 ms GPU-kernel=1.42189 ms mse=29884977152.000000

Device Name: TITAN RTX
shared mem size: 65536

	N=200 K=64 T=514         CPU=293.748 ms GPU=226.69 ms GPU-kernel=1.82426 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=286.468 ms GPU=95.228 ms GPU-kernel=1.84973 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=296.623 ms GPU=89.378 ms GPU-kernel=1.84192 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=293.714 ms GPU=90.288 ms GPU-kernel=1.84074 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=280.09 ms GPU=78.821 ms GPU-kernel=1.84797 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=282.613 ms GPU=78.135 ms GPU-kernel=1.84448 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=287.452 ms GPU=85.605 ms GPU-kernel=1.84115 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=291.543 ms GPU=86.857 ms GPU-kernel=1.84221 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=283.616 ms GPU=86.829 ms GPU-kernel=1.84717 ms mse=29884977152.000000

Device Name: GeForce GTX 1070 Ti
shared mem size: 65536

	N=200 K=64 T=514         CPU=574.396 ms GPU=554.609 ms GPU-kernel=2.03446 ms mse=29885042688.000000
	N=200 K=64 T=514         CPU=574.333 ms GPU=267.653 ms GPU-kernel=2.0441 ms mse=inf
	N=200 K=64 T=514         CPU=575.825 ms GPU=261.066 ms GPU-kernel=2.0576 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=575.803 ms GPU=274.544 ms GPU-kernel=2.05088 ms mse=inf
	N=200 K=64 T=514         CPU=574.916 ms GPU=257.873 ms GPU-kernel=2.04118 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=575.576 ms GPU=260.762 ms GPU-kernel=2.04915 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=584.074 ms GPU=258.826 ms GPU-kernel=2.05363 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=574.655 ms GPU=257.94 ms GPU-kernel=2.03139 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=574.483 ms GPU=257.8 ms GPU-kernel=2.02806 ms mse=inf
	N=200 K=64 T=514         CPU=574.47 ms GPU=267.096 ms GPU-kernel=2.04304 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=574.567 ms GPU=247.155 ms GPU-kernel=2.02653 ms mse=29479469056.000000
	N=200 K=64 T=514         CPU=620.659 ms GPU=280.295 ms GPU-kernel=2.0353 ms mse=29884977152.000000
	N=200 K=64 T=514         CPU=579.839 ms GPU=243.215 ms GPU-kernel=2.0241 ms mse=388258878051752614606563625324773376.000000



