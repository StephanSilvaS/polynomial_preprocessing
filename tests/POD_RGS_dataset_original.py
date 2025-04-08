from astropy.io import fits
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation
from math import e, pi
import math
from array import *
import time
from line_profiler import profile

#namefile_psnr = "psnr_matrix_natural_S_30_21_Sigma_500_Rep_1.npz"
#psnr_matrix = np.load(namefile_psnr)

#psnr_matrix = psnr_matrix["arr_0"][:,:]

fits_image = fits.open("dirty_images_natural_251.fits")
header = fits_image[0].header

print(header)

# weight,x,y: dim n
# weight: real
# x,y: complex
# return: number
# =sum_k{w_k*x_k*y_k}
def dot(weights,x,y):
  return(np.sum((x*weights*np.conjugate(y))))

# weight,x: dim n
# weight: real
# x: complex
# return number
# =sum_k{w_k*|x_k|^2}}
def norm(weights,x):
  return(np.sqrt(np.sum(weights*np.absolute(x)**2)))

# weights,pol : dim n
# weight: real
# matrix : dim (N,N,n)
# matrix : complex
# pol : complex
# return[N,N](k,j) = {dot(weight, matrix[k,j,:],pol)}  : given a pol return dot over previous pol(matrix)
#def dot2x2(weights,matrix,pol):
#    weights = cp.array(weights)
#    matrix = cp.array(matrix)
#    pol = cp.array(pol)

#    N1,N2,n = matrix.shape
#    w = cp.ones(shape=(N1,N2,n),dtype=float)*weights # w: dim(N1,N2,n)

#    final_dot = matrix*w*cp.conjugate(pol)
#    final_dot = cp.sum(final_dot,axis=2)  # dot(weight, matrix[k,j,:],pol)
#    final_dot = cp.reshape(final_dot,(N1,N2,1))# *Important for further broadcasting in GS*
 
#    return cp.asnumpy(final_dot) # back to cpu
	
@profile
def dot2x2(weights,matrix,pol,chunk_data):
	weights = cp.array(weights)
	pol = cp.array(pol)
	N1,N2,n = matrix.shape
	sub_size = int(N1/chunk_data) + 1
	final_dot = np.zeros(shape=(N1,N2,1),dtype=np.complex128)
	for chunk1 in range(0,sub_size):
		for chunk2 in range(0,sub_size):
			if chunk1 + chunk2 < sub_size:
				sub_m = cp.array(matrix[chunk1*chunk_data:(chunk1+1)*chunk_data, chunk2*chunk_data:(chunk2+1)*chunk_data,:])
				N3,N4,n2 = sub_m.shape
				w = cp.ones(shape=(N3,N4,n2),dtype=float)*weights
				subsum = sub_m*w*cp.conjugate(pol)
				subsum = cp.sum(subsum,axis=2)
				subsum = cp.reshape(subsum,(N3,N4,1))
				final_dot[chunk1*chunk_data:(chunk1+1)*chunk_data, chunk2*chunk_data:(chunk2+1)*chunk_data,:] = cp.asnumpy(subsum)

	return final_dot


# weights : dim n
# weight: real
# matrix : dim (N,N,n)
# matrix : complex
# return[N,N](k,j) : matrix of norm for each polynomials k,j
#def norm2x2(weights,matrix):
#    weights = cp.array(weights)
#    matrix = cp.array(matrix)
#    N1,N2,n = matrix.shape   
#    w = cp.ones(shape=(N1,N2,n),dtype=float)*weights
#    final_norm = w*cp.absolute(matrix)**2
#    final_norm = cp.sum(final_norm,axis=2)
#    final_norm = cp.sqrt(final_norm)
#    final_norm = cp.reshape(final_norm,(N1,N2,1)) # important for further broadcasting in GS
	
#    return cp.asnumpy(final_norm) # back to cpu
 
def norm2x2(weights,matrix,chunk_data):
	weights = cp.array(weights)
	N1,N2,n = matrix.shape
	sub_size = int(N1/chunk_data) + 1
	final_norm = np.zeros(shape=(N1,N2,1),dtype=np.complex128)
	for chunk1 in range(0,sub_size):
		for chunk2 in range(0,sub_size):
			if chunk1 + chunk2 < sub_size:
				sub_m = cp.array(matrix[chunk1*chunk_data:(chunk1+1)*chunk_data,chunk2*chunk_data:(chunk2+1)*chunk_data,:])
				N3,N4,n2 = sub_m.shape
				w = cp.ones(shape=(N3,N4,n2),dtype=float)*weights
				subsum = w*cp.absolute(sub_m)**2
				subsum = cp.sum(subsum,axis=2)
				subsum = cp.sqrt(subsum)
				subsum = cp.reshape(subsum,(N3,N4,1))
				final_norm[chunk1*chunk_data:(chunk1+1)*chunk_data,chunk2*chunk_data:(chunk2+1)*chunk_data,:] = cp.asnumpy(subsum)
	return final_norm
 
# img1, img2: dim(M,M)
# img1,img2: real!!!! comentary: do not work for complex 
# return mean quadratic diference
def  mse(img1, img2):
	N1,N2 = img1.shape
	err = np.sum((img1 - img2)**2)/(N1*N2)
	#err = err / (img1.shape[0]*img1.shape[1])
	return err

def psnr(img_ini,img_fin):
	return 20*math.log10(np.max(np.max(img_fin))/mse(img_ini,img_fin)) # comentary mse need to be taken outside the object



# z_target: dim M
# z,weights, data: dim n
# size: number , size==M
# s: number, s==N
# division_sigma: real, selection criteria
# return: final_data (TARGET, dim (M,M)), err (sigma_TARGET, dim (M,M)), residual, dim (n,n), P_target (N,N,M), P (N,N,n), std_a
def recurrence2d(z_target,z,weights,data,size,s,division_sigma,chunk_data):
	z = np.array(z)
	z_target = np.array(z_target)
	w = np.array(weights)
	residual = np.array(data)

	sigma_weights = np.divide(1.0, w, where=w!=0, out=np.zeros_like(w))

	sigma2 = np.max(sigma_weights)/division_sigma # 
	
	print("Sigma: ",sigma2)

	final_data = np.zeros(shape=(size),dtype=np.complex128)
	P = np.zeros(shape=(s,s,z.size),dtype=np.complex128)
	P_target = np.zeros(shape=(s,s,size),dtype=np.complex128)
	V = np.ones(shape=(s,s,1),dtype=int) # validation matrix: discard already processed pol, 1:to process, 0:already processed      
	D = np.zeros(z.size,dtype=np.complex128) # previous pol for computing the actual pol
	D_target = np.zeros(size,dtype=np.complex128) # same on target
	err = np.zeros(shape=(size),dtype=float)

	print("Paso 1 ",np.size(err))   
	max_rep = 2
	i = 1.0
	b = 1
	
	# init for polynomial matrix: P,P_target
	# Gram Matrix
	for j in range(0,s):
		for k in range(0,s):
			P[k,j,:] = (z**(k))*(np.conjugate(z)**j)
			P_target[k,j,:] = (z_target**(k))*(np.conjugate(z_target)**j)
			
			no=norm(w,P[k,j,:])
			P[k,j,:]       = np.divide(P[k,j,:],       no,out = np.zeros(np.size(P[k,j,:]),       dtype=np.complex128),where=(no!=0))
			P_target[k,j,:]= np.divide(P_target[k,j,:],no,out = np.zeros(np.size(P_target[k,j,:]),dtype=np.complex128),where=(no!=0))
			
	
	#P = P*np.exp(-z*np.conjugate(z)/(2*(b**2)))
	#P_target = P_target*np.exp(-z_target*np.conjugate(z_target)/(2*(b**2)))
	no_data = norm2x2(w,P,chunk_data)
	P = np.divide(P, no_data, where=no_data!=0, out=np.zeros_like(P))
	P_target = np.divide(P_target, no_data, where=no_data!=0, out=np.zeros_like(P_target))
	
	# GS procedure on polynomials + 1 repetion, iteration goes on contra diagonal for same total degree k
	for k in range(0,s): # total degree, correspond to last column
		#large3 = np.concatenate((large3,np.array([k])),axis=0)
		for j in range(0,k+1): # position in contra diagonal (row degree)
			print(k-j,j)
			# total degree == (k-j)+j == k


			# after several experiments one repeat is sufficient
			for repeat in range(0,max_rep):
				
				if repeat > 0: # for first pass
					# normalizing P, P_Target
					no=norm(w,P[k-j,j,:])                     
					P[k-j,j,:]=np.divide(P[k-j,j,:],no,out = np.zeros(np.size(P[k-j,j,:]),dtype=np.complex128),where=(no!=0))
					P_target[k-j,j,:]=np.divide(P_target[k-j,j,:],no,out = np.zeros(np.size(P_target[k-j,j,:]),dtype=np.complex128),where=(no!=0))
			

				if k==0 and j==0: # for first pass
					# normalizing P, P_Target
					no=norm(w,P[k-j,j,:]) 
					P[k-j,j,:]=np.divide(P[k-j,j,:],no,out = np.zeros(np.size(P[k-j,j,:]),dtype=np.complex128),where=(no!=0))
					P_target[k-j,j,:]=np.divide(P_target[k-j,j,:],no,out = np.zeros(np.size(P_target[k-j,j,:]),dtype=np.complex128),where=(no!=0))
			
					
					D = np.array(P[k-j,j,:]) # store first polynomial
					D_target = np.array(P_target[k-j,j,:])
					V[k-j,j,:] = 0 # this pol was processed

				else: # for the remaining
					if repeat == 0:
						#print(k-j+l,j)
						if j==1 and k>0:
							no_data = norm2x2(w,P,chunk_data)
							no_data[V == 0] = 1
							P = np.divide(P, no_data, where=no_data!=0, out=np.zeros_like(P))
							P_target = np.divide(P_target, no_data, where=no_data!=0, out=np.zeros_like(P_target))

						# GS main operation
						dot_data = dot2x2(w,P*V,D,chunk_data) # dim ((n),(N1,N2,n),(n))  --> return (N1,N2,1)
						P = P - dot_data*D # dim (N1,N2,n) - (N1,N2,1)*(n)
						P_target = P_target - dot_data*D_target # dim (N1,N2,M) - (N1,N2,1)*(M)
						
						#normalization
						no=norm(w,P[k-j,j,:])
						P[k-j,j,:]=np.divide(P[k-j,j,:],no,out = np.zeros(np.size(P[k-j,j,:]),dtype=np.complex128),where=(no!=0))
						P_target[k-j,j,:]=np.divide(P_target[k-j,j,:],no,out = np.zeros(np.size(P_target[k-j,j,:]),dtype=np.complex128),where=(no!=0))
			

						if (j==0):
							# normalization
							no=norm(w,P[k-j,j,:])
							P[k-j,j,:]=np.divide(P[k-j,j,:],no,out = np.zeros(np.size(P[k-j,j,:]),dtype=np.complex128),where=(no!=0))
							P_target[k-j,j,:]=np.divide(P_target[k-j,j,:],no,out = np.zeros(np.size(P_target[k-j,j,:]),dtype=np.complex128),where=(no!=0))
			
						
					if repeat > 0:
						# Could be the same than step "GS main operation"
						for y in range(0,k+1):
							for x in range(0,y+1):
								if (y-x != k-j) or (j!=x):
									dot_data=dot(w,P[k-j,j,:],P[y-x,x,:])
									P[k-j,j,:] = P[k-j,j,:] - dot_data*P[y-x,x,:]
									P_target[k-j,j,:] = P_target[k-j,j,:] - dot_data*P_target[y-x,x,:]
								else:
									break 
							
							#normalization
							no=norm(w,P[k-j,j,:])
							P[k-j,j,:]=np.divide(P[k-j,j,:],no,out = np.zeros(np.size(P[k-j,j,:]),dtype=np.complex128),where=(no!=0))
							P_target[k-j,j,:]=np.divide(P_target[k-j,j,:],no,out = np.zeros(np.size(P_target[k-j,j,:]),dtype=np.complex128),where=(no!=0))

							#P[np.isnan(P)] = 0.0
							#P[np.isinf(P)] = 0.0
							#P_target[np.isnan(P_target)] = 0.0
							#P_target[np.isinf(P_target)] = 0.0
			
				
				P[np.isnan(P)] = 0.0
				P[np.isinf(P)] = 0.0
				P_target[np.isnan(P_target)] = 0.0
				P_target[np.isinf(P_target)] = 0.0
							 
			# updating validation matrix and current polynomial
			V[k-j,j,:] = 0
			D = np.array(P[k-j,j,:])
			D_target = np.array(P_target[k-j,j,:])

			# calculating extrapolation using current polynomials
			M = dot(w,residual.flatten(),P[k-j,j,:])
			final_data = final_data + M*P_target[k-j,j,:]
			residual = residual - M*P[k-j,j,:]
		   
			# error estimation
			err = err + np.absolute(P_target[k-j,j,:])**2
  
	final_data[cp.asnumpy(err)>sigma2]=0
	#plt.figure()
	#plt.plot(final_data.flatten(),color='b')
	#plt.figure()
	#plt.plot(data.flatten(),color='k')
	return final_data,err,residual, P_target, P

M = 1

N1 = 251 * M

S = 20
sub_S = int(S)
ini = 1
#p = 0.1

#namefile_visibilities = "gridded_visibilities_natural_251_HLTau_B6cont.npz"
#namefile_weights = "gridded_weights_natural_251_HLTau_B6cont.npz"
#namefile_visibilities = "gridded_visibilities_natural_251.npz"
#namefile_weights = "gridded_weights_natural_251.npz"


#gridded_visibilities = np.load(namefile_visibilities)
#gridded_visibilities = gridded_visibilities["arr_0"][0,:,:]

#u_t = np.reshape(np.linspace(-ini,ini,251),(1,251))*np.ones(shape=(251,1))
#v_t = np.reshape(np.linspace(-ini,ini,251),(251,1))*np.ones(shape=(1,251))

#z_t = u_t+1j*v_t

#z_t = z_t[np.absolute(gridded_visibilities)!=0]



#gridded_weights = np.load(namefile_weights)
#gridded_weights = gridded_weights["arr_0"][0,:,:]

#gv_sparse = gridded_visibilities[np.absolute(gridded_visibilities)!=0]
#gw_sparse = gridded_weights[np.absolute(gridded_visibilities)!=0]

namefile_fulldata = "hd100546_selfcal_cont_13.npz"

full_data = np.load(namefile_fulldata)

gv_sparse = full_data["arr_6"][0,0,:]
gv_sparse = gv_sparse/np.sqrt(np.sum(gv_sparse**2))
#gv_sparse = np.concatenate((gv_sparse,np.conjugate(gv_sparse)))

gw_sparse = full_data["arr_2"][0,:]
gw_sparse = gw_sparse/np.sqrt(np.sum(gw_sparse**2))
#gw_sparse = np.concatenate((gw_sparse,gw_sparse))

u_data = full_data["arr_0"][0,:]
v_data = full_data["arr_0"][1,:]

#print(np.max(np.absolute(u_data)))
#print(np.max(np.absolute(v_data)))


#round_uv = np.round(u_data + v_data*1j)
#r_uv, pos_uv = np.unique(round_uv,return_index=True)

#u_data = np.round(u_data[pos_uv])
#v_data = np.round(v_data[pos_uv])
#gv_sparse = gv_sparse[pos_uv]
#gw_sparse = gw_sparse[pos_uv]


plt.figure()
plt.plot(gv_sparse.flatten(),color='r')

max_uv = 0

	
#max_uv = 501

#dx = 0.0007310213536 / M
dx = 0.00253827
du = 1/(N1*dx)

umax = N1*du/2

u_sparse = np.array(u_data)/umax
v_sparse = np.array(v_data)/umax

print(np.max(np.absolute(u_sparse)),np.max(np.absolute(v_sparse)))

#u_max = int(max_uv/2)
#u_min = int(-1*max_uv/2)

#u_sparse = (u_data - ((u_max + u_min)*0.5))/((u_max - u_min)*0.5)
#v_sparse = (v_data - ((u_max + u_min)*0.5))/((u_max - u_min)*0.5)

plt.figure()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.scatter(u_sparse, v_sparse, s = 1)

#plt.figure()
#plt.xlim(-1, 1)
#plt.ylim(-1, 1)
#plt.scatter(np.real(z_t.flatten()), np.imag(z_t.flatten()), s = 0.1)

#plt.show()

print(np.size(gv_sparse))

u_target = np.reshape(np.linspace(-ini,ini,N1),(1,N1))*np.ones(shape=(N1,1))
v_target = np.reshape(np.linspace(-ini,ini,N1),(N1,1))*np.ones(shape=(1,N1))

z_target = u_target+1j*v_target
z_sparse = u_sparse+1j*v_sparse
#z_sparse = np.concatenate((z_sparse,-z_sparse))

#pos_sparse = np.argsort(z_sparse)
#z_sparse = np.sort(z_sparse)
#gv_sparse[pos_sparse] = gv_sparse
#gw_sparse[pos_sparse] = gw_sparse

b = 1

z_exp = np.exp(-z_target*np.conjugate(z_target)/(2*b*b))

title="Z exp"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(z_exp))
plt.colorbar(im)
plt.show()


#max_memory = 120000000
max_memory = 120000000
max_data = float(int(max_memory/(S*S)))


divide_data = int(np.size(gv_sparse[np.absolute(gv_sparse)!=0].flatten())/max_data)+1
divide_target = int(N1*N1/max_data)+1

if divide_target > divide_data:
	divide_data = int(divide_target)

if divide_data > int(divide_data):
	divide_data = int(divide_data) + 1

chunk_data = int(((S*S)/divide_data)**(1/2)) + 1
if chunk_data == 0:
	chunk_data = 1

chunk_data = 52
print(chunk_data)


visibilities_model = np.zeros((N1,N1),dtype=np.complex128)


division = 10**(-1) # division_sigma

print("New S:",S)
print("Division:",division)

visibilities_aux = np.zeros(N1*N1,dtype=np.complex128)
weights_aux = np.zeros(N1*N1,dtype=float)

start_time = time.time()

visibilities_mini, err, residual, P_target, P = recurrence2d(z_target.flatten(), z_sparse.flatten(), gw_sparse.flatten(), gv_sparse.flatten(), np.size(z_target.flatten()), S, division,chunk_data)

'''
#aux = np.reshape(np.arange(S),(S,1,1)) + np.reshape(np.arange(S),(1,S,1))
#print(np.sum(np.absolute(visibilities_aux - visibilities_aux_2)))
aux[aux < S] = 1
aux[aux >= S] = 0
idx = aux*np.ones(shape=(S,S,np.size(z_sparse.flatten())))
pp=P[idx == 1]

p1 = np.reshape(pp,(int(S*(S+1)/2),1,np.size(z_sparse.flatten())))
p2 = np.reshape(pp,(1,int(S*(S+1)/2),np.size(z_sparse.flatten())))
w = np.reshape(gw_sparse.flatten(),(1,1,np.size(z_sparse.flatten())))


corr = np.sum(p1*w*np.conjugate(p2),axis=2) # shape (n1n2,n1n2)

corr2=corr-np.diag(np.diag(corr))


fig=plt.figure("corr")
im=plt.imshow(np.absolute(corr))
plt.colorbar(im)

fig=plt.figure("corr2")
im=plt.imshow(np.absolute(corr2))
plt.colorbar(im)
'''
print()
visibilities_mini = np.reshape(visibilities_mini,(N1,N1))

visibilities_model = np.array(visibilities_mini)

plt.figure()
plt.plot(visibilities_model.flatten(),color='g')


weights_model = np.zeros((N1,N1),dtype=float)

sigma_weights = np.divide(1.0, gw_sparse, where=gw_sparse!=0, out=np.zeros_like(gw_sparse))#1.0/gw_sparse
sigma = np.max(sigma_weights)/division
weights_mini = np.array(1/err)
weights_mini[np.isnan(weights_mini)] = 0.0
weights_mini[np.isinf(weights_mini)] = 0.0

#weights_mini[err > sigma] = 0
weights_mini = np.reshape(weights_mini,(N1,N1))
#weights_model[int((gridded_size-sub_size)/2):gridded_size - int((gridded_size-sub_size)/2),int((gridded_size-sub_size)/2):gridded_size - int((gridded_size-sub_size)/2)] = np.array(weights_aux)
weights_model = np.array(weights_mini)

print(time.time() - start_time)


print(visibilities_model.shape)


image_model = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibilities_model*weights_model/np.sum(weights_model.flatten()))))*N1**2
image_model = np.array(image_model.real)

title="Image model (division sigma: "+str(division)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(image_model)
plt.colorbar(im)
#title="Residual model (division sigma: "+str(division)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(image_residual)
#plt.colorbar(im)

title="Visibility model (division sigma: "+str(division)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(np.absolute(visibilities_model))
plt.colorbar(im)


title="Weights model (division sigma: "+str(division)+")"; fig=plt.figure(title); plt.title(title); im=plt.imshow(weights_model)
plt.colorbar(im)

plt.show()

#visibilities_residual = gridded_visibilities - visibilities_model

tittle_1 = "visibility_model_natural_"
tittle_1_fits = "image_model_natural_"
tittle_1_weigths = "weigths_model_natural_"
tittle_1_residual = "residual_model_natural_"
tittle_2 = "S_"+str(S)+"_"+str(S)+"_"
tittle_3 = "division_"+str(division)+"_"
#tittle_4 = "size_"+str(N1)+"_"+str(N1)+"_hd100546_selfcal_cont_13"
tittle_4 = "size_"+str(N1)+"_"+str(N1)+"_original_hd142_b9cont_self_tav"

title_visibilities_result =  tittle_1+tittle_2+tittle_3+tittle_4+".npz"
np.savez(title_visibilities_result, visibilities_model)
#np.savez("test_visibility_model.npz", visibilities_model)

title_weigths_result =  tittle_1_weigths+tittle_2+tittle_3+tittle_4+".npz"
np.savez(title_weigths_result, weights_model)
#np.savez("test_weight_model.npz", weights_model)

title_visibilities_fits =  tittle_1_fits+tittle_2+tittle_3+tittle_4+".fits"
fits.writeto(title_visibilities_fits, image_model, header,overwrite=True)
#fits.writeto(title_visibilities_fits, image_model, header)

#fits.writeto("test_visibility_model.fits", np.absolute(visibilities_model), header,overwrite=True)
#fits.writeto("test_weight_model.fits", weights_model, header,overwrite=True)
#fits.writeto("test_image_model.fits", image_model, header,overwrite=True)

title_residual_fits =  tittle_1_residual+tittle_2+tittle_3+tittle_4+".fits"
#fits.writeto(title_residual_fits, image_residual, header,overwrite=True)
#its.writeto(title_residual_fits, image_residual, header)


#data_process = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(visibilities_aux*weights_aux/np.sum(weights_aux.flatten()))))
#data_process = np.array(data_process.real)*N**2

#fig=plt.figure(); im=plt.imshow(np.absolute(weights_aux))