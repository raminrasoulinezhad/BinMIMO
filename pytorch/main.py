# Source: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import torch
import argparse
import time
import math  

def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--device', type=str, default="cpu", help='cpu or cuda:0')

	parser.add_argument('--exp_try_max', type=int, default=20)
	parser.add_argument('--iter_try_max', type=int, default=30)
	
	parser.add_argument('--N', type=int, default=200)
	parser.add_argument('--Td', type=int, default=450)
	parser.add_argument('--Tt', type=int, default=50)
	parser.add_argument('--K', type=int, default=64)

	parser.add_argument('--eps', type=float, default=0.001)
	parser.add_argument('--SNR', type=float, default=10)

	parser.add_argument('--H_init', type=str, default="zero", help='zero or M_LMMSE')
	return parser.parse_args()


def init_H_hat(Y, X_t, K, P_n, device='cpu', dtype=torch.float):
	I_K = torch.eye(K, device=device, dtype=dtype)
	X_t_conj_t = torch.transpose(torch.conj(X_t), 0, 1)

	#temp = torch.inverse(X_t.mm(X_t_conj_t) + K * P_n * I_K)
	temp = matmul_complex(X_t, X_t_conj_t)
	temp += K * P_n * I_K	
	temp = torch.inverse(temp)
	
	return matmul_complex( Y, matmul_complex(X_t_conj_t, temp) )


def matmul_complex(t1,t2):
	# sourse:	https://stackoverflow.com/questions/63855692/matrix-multiplication-for-complex-numbers-in-pytorch
	return torch.view_as_complex(torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real),dim=2))

def erfcx(A): 
	return torch.exp(A*A)*torch.erfc(A)

def estim(Z, Phat, Pvar):

	PMonesY_R = torch.sign(Z.real - 0.1);
	PMonesY_I = torch.sign(Z.imag - 0.1);

	Phat_R = Phat.real ;
	Phat_I = Phat.imag ;
	
	Z_mean = torch.mean(Z.real)
	Z_var = torch.var(Z.real, unbiased=True)

	VarN = 0.5 * Z_var ; 
	Pvar = 0.5 * Pvar;   
	
	temp = torch.sqrt( Pvar + VarN )
	C_R = PMonesY_R * ((Phat_R - Z_mean) / temp);
	C_I = PMonesY_I * ((Phat_I - Z_mean) / temp);
	
	ratio_R = (2/math.sqrt(2*3.1415)) * (erfcx(-C_R / math.sqrt(2)).pow(-1));
	ratio_I = (2/math.sqrt(2*3.1415)) * (erfcx(-C_I / math.sqrt(2)).pow(-1));

	Zhat_R = Phat_R + PMonesY_R * ((Pvar / torch.sqrt(Pvar + VarN)) * ratio_R);
	Zhat_I = Phat_I + PMonesY_I * ((Pvar / torch.sqrt(Pvar + VarN)) * ratio_I);
	Zhat = Zhat_R + 1j*Zhat_I ; 
	
	
	temp = (Pvar.pow(2) / (Pvar + VarN))

	Zvar_R = Pvar - (temp * ratio_R) * (C_R + ratio_R);
	Zvar_I = Pvar - (temp * ratio_I) * (C_I + ratio_I);

	Zvar = (Zvar_R + Zvar_I);

	return Zhat, Zvar 


if __name__ == "__main__":

	args = get_args()

	
	dtype_c = torch.cfloat
	dtype_f = torch.float
	#print ("Selected data type: %s" % (dtype))
	
	device = torch.device(args.device) # "cpu", "cuda:0"
	print ("Selected device: %s" % (args.device))

	# ################
	# parameters
	# ################
	N = args.N
	Td = args.Td
	Tt = args.Tt
	T = Td + Tt
	K = args.K
	
	eps = args.eps
	SNR = args.SNR

	Y = torch.randn(N, T, device=device, dtype=dtype_c)

	Z = torch.zeros(N, T, device=device, dtype=dtype_c)

	X_t = torch.zeros(K, Tt, device=device, dtype=dtype_c)
	X_d = torch.randn(K, Td, device=device, dtype=dtype_c)

	V_t = torch.zeros(K, Tt, device=device, dtype=dtype_f)
	V_d = torch.ones(K, Td, device=device, dtype=dtype_f)

	S_hat = torch.zeros(N, T, device=device, dtype=dtype_f)

	V_h = torch.ones(N, K, device=device, dtype=dtype_f)

	P_n = 10 if (SNR < 20) else 14 

	print ("Selected init method (H_hat): %s" % (args.H_init))
	if args.H_init == 'zero':
		H_hat = torch.zeros(N, K, device=device, dtype=dtype_c)
	elif args.H_init == 'M_LMMSE':
		H_hat = init_H_hat(Y, X_t, K, P_n, device=device, dtype=dtype_c)
	else:
		raise Exception('args.H_init: %s is not supported!' % str(args.H_init))

	start_time = time.time()
	
	for exp_try in range(args.exp_try_max):
	
		for iter_try in range(args.iter_try_max):
			
			# line 1
			temp = H_hat.abs().pow(2).mm( torch.cat((V_t, V_d), 1) ) 
			temp += V_h.mm( torch.cat((X_t, X_d), 1).abs().pow(2) )
			
			V_p = temp + V_h.mm( torch.cat((V_t, V_d), 1) ) 
			
			# line 2

			P_hat = S_hat * temp 
			P_hat = matmul_complex( H_hat, torch.cat((X_t, X_d), 1) ) - P_hat

			# line 3 and 4
			Zhat, Zvar = estim(Z, P_hat, V_p)

			# line 5 and 6
			V_s = (1-Zvar/V_p)/V_p
			S_hat = (Z - P_hat)/V_p

			# line 7
			V_r = 1 / (torch.transpose(H_hat, 0, 1).abs().pow(2).mm(V_s))

			# line 8
			R_hat = torch.cat((X_t, X_d), 1) * (1 - V_r * torch.transpose(V_h, 0, 1).mm(V_s) ) + V_r * matmul_complex(H_hat.conj().transpose(0,1), S_hat)


#			# line 9
#			V_q = 1 / (V_s.mm(torch.cat((X_t, X_d), 1).transpose(0,1).pow(2)))
#
#			# line 10
#			Q_hat = H_hat * (1 - V_q * V_s.mm(torch.transpose(V_x, 0, 1))) + V_q * (S_hat.mm(   torch.cat((X_t, X_d), 1).conj().transpose(0, 1)  ))
#
#			# line 11-12
#			Zhat, Zvar = estim(torch.cat((X_t, X_d), 1), R_hat[:,Tt:], V_r[Tt:])
#			# line 13-14
#			Zhat, Zvar = estim(H_hat, Q_hat, V_q)

			#exit()
			#print(V_p.shape)
			# Forward pass: compute predicted y
			#h = x.mm(w1)
			#h_relu = h.clamp(min=0)
			#loss = (y_pred - y).pow(2).sum().item()
			#grad_w2 = h_relu.t().mm(grad_y_pred)
			

	end_time = time.time()

	print ("Code execution time in ms is %f", 1000*(end_time - start_time) / args.exp_try_max)
