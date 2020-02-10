Main files are:

	GAMP_cpuFuncs.cu
	GAMP_gpuFuncs.cu
	GAMP_main.cu

## Ramin:
	cd model
	make all
	
## Results

ProfPhillet:

	Device Name: Tesla M40
	shared mem size: 65536
	N=200 K=64 T=514	 CPU=288.897 ms GPU=63.923 ms GPU-kernel=3.08845 ms mse=29884977152.000000

DrFish:

	Device Name: GeForce RTX 2080 Ti
	shared mem size: 65536
	N=200 K=64 T=514   CPU=224.522 ms GPU=235.301 ms GPU-kernel=1.77885 ms mse=31134828544.000000

guppy:

	Device Name: GeForce GTX 1080 Ti
	shared mem size: 65536
	N=200 K=64 T=514	 CPU=209.731 ms GPU=69.254 ms GPU-kernel=1.75251 ms mse=29884977152.000000
	N=200 K=64 T=514	 CPU=191.497 ms GPU=46.695 ms GPU-kernel=1.88506 ms mse=29884977152.000000

Ramin Thinkpad:
	
	Device Name: GeForce 940MX
	shared mem size: 65536
	N=200 K=64 T=514	 CPU=270.094 ms GPU=36.585 ms GPU-kernel=5.70739 ms mse=29884977152.000000



