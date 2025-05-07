
from math import pi, sqrt, floor, log
import numpy as np
import cupy as cp
import os
import nrrd
import random
import sys
from Tools import patch_img, patch_img_3ch
from scipy import ndimage
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler


def crop_center(img,cropx,cropy,cropz):
    
    x,y,z = img.shape
    x = int(x/2.0)*2
    y = int(y/2.0)*2
    z = int(z/2.0)*2
    temp = np.zeros((cropx,cropy,cropz))
    print(img.shape)
    print(temp.shape)
    if x>cropx:
        inx = [x//2-(cropx//2), x//2+(cropx//2)-1]
        outx = [0, cropx-1]
    else:
        inx = [0, x-1]
        outx = [cropx//2-floor(x//2), cropx//2+floor(x//2)-1]

    if y>cropy:
        iny = [y//2-(cropy//2), y//2+(cropy//2)-1]
        outy = [0, cropy-1]
    else:
        iny = [0, y-1]
        outy = [cropy//2-floor(y//2), cropy//2+floor(y//2)-1]

    if z>cropz:
        inz = [z//2-(cropz//2), z//2+(cropz//2)-1]
        outz = [0, cropz-1]
    else:
        inz = [0, z-1]
        outz = [cropz//2-floor(z//2), cropz//2+floor(z//2)-1]

    print(iny)
    print(outy)
    temp[outx[0]:outx[1],outy[0]:outy[1],outz[0]:outz[1]] = img[inx[0]:inx[1],iny[0]:iny[1],inz[0]:inz[1]]
    return temp


def perform_gpu_fft(data):
    L = data.shape
    n = L[0] * L[1] * L[2]
    data_gpu = cp.asarray(data) # Move data to GPU
    fft_result = cp.fft.ifftshift(data_gpu) # Perform FFTSHIFT
    fft_result = cp.fft.fftn(fft_result) # Perform 3D FFT
    fft_result = cp.fft.fftshift(fft_result) # Perform FFTSHIFT
    return cp.asnumpy(fft_result)/sqrt(n) # Move result back to CPU

def perform_gpu_ifft(fft_data):
    L = fft_data.shape
    n = L[0] * L[1] * L[2]
    fft_data_gpu = cp.asarray(fft_data) # Move FFT data to GPU
    fft_data_gpu = cp.fft.fftshift(fft_data_gpu) # Perform FFTSHIFT
    ifft_result = cp.fft.ifftn(fft_data_gpu) # Perform 3D IFFT
    ifft_result = cp.fft.ifftshift(ifft_result) # Perform FFTSHIFT

    return cp.asnumpy(ifft_result)*sqrt(n) # Move result back to CPU


#https://journals.physiology.org/doi/full/10.1152/japplphysiol.01036.2005
#https://pmc.ncbi.nlm.nih.gov/articles/PMC5743233/#FD1
def Sinusoidal_wave(A, f, B, t):
    #Return velocity in cm/s
    return A*np.sin(pi*f*t) + B


#https://www.deltexmedical.com/decision_tree/doppler-specific-parameters/
def Triangular_wave(A, f, B, t):
    #Return velocity in cm/s
    blcyc = 1.0/f
    blph = t % blcyc
    FTc = 330e-3#second

    FT1_ind = np.logical_not(blph < FTc*0.5)
    vel1 = A*blph/(0.5*FTc)
    vel1[FT1_ind] = 0


    FT2_ind = np.logical_not( np.logical_and((blph < FTc), (blph >= FTc*0.5)) )
    vel2 = -1.0*A*blph/(0.5*FTc) + 2.0*A
    vel2[FT2_ind] = 0

    return vel1 + vel2 + B


def phase_error_model(venc, velocity):
    PE = pi*velocity/venc
    return PE

def addGaussianNoise(spect, std):

    shape = spect.shape

    mean = 0.0
    std_dev =std

    # Generate Gaussian noise of the same shape as the image
    real_noise = np.random.normal(loc=mean, scale=std_dev, size=shape)
    imag_noise = np.random.normal(loc=mean, scale=std_dev, size=shape)

    # Create complex noise with the Gaussian noise
    cplx_noise = real_noise + 1j*imag_noise

    # Add the complex noise to the image
    noisy_spect = spect + cplx_noise
    
    return noisy_spect



def addErrors_random(spect, A, f, B, TR, venc):

    shape = spect.shape

    x = np.linspace(0, shape[1]-1, shape[1])
    y = np.linspace(0, shape[2]-1, shape[2])
    xv, yv = np.meshgrid(x,y)

    #Mask out center of k-space
    x = np.linspace(-shape[0]/2, shape[0]/2-1, shape[0])
    y = np.linspace(-shape[1]/2, shape[1]/2-1, shape[1])
    z = np.linspace(-shape[2]/2, shape[2]/2-1, shape[2])
    xv, yv, zv = np.meshgrid(x,y,z)
    vsz = np.sqrt(yv*yv+4*zv*zv)
    vsz = (vsz>20).astype(float)

    ph_random = 2.0 * np.pi * np.random.rand(shape[0], shape[2])

    phaseError = ph_random.reshape((shape[0], 1, shape[2]))

    phaseError = np.tile(phaseError, (1,shape[1],1))

    phaseError = 1j*np.sinh( (phaseError)*1j ) * vsz - (1-vsz)

    spect = spect * phaseError

    return spect

#https://pmc.ncbi.nlm.nih.gov/articles/PMC5743233/
def addErrors(spect, A, f, B, TR, venc):

    shape = spect.shape
    s0, s1, s2 = shape
    duration = 1/f
    durations = duration + duration * (0.9 * np.random.rand(s1 * s2))

    #Owashi, K. P., Capel, C., & Balédent, O. (2023). Cerebral arterial flow dynamics during systole and diastole phases in young and older healthy adults. Fluids and Barriers of the CNS, 20(1), 65.
    duration_systolic = (160.0 + random.uniform(-29.0, 29.0))*1e-3

    x = np.linspace(0, shape[0]-1, shape[0])
    y = np.linspace(0, shape[2]-1, shape[2])
    xv, yv = np.meshgrid(y,x)


    #This index denotes the number of TRs from begining of acquisition
    phaseIndex = yv*shape[2] + xv
    #phaseIndex = xv*shape[1] + yv
    t = phaseIndex*TR
    #t = t % duration
    #t[t>duration_systolic] = 0.0
    t_flat = t.T.flatten(order='F')
    t_ind = 0
    for d in durations:
        max_t = int(np.floor(d / TR))
        if t_ind + max_t > t_flat.size:
            t_flat[t_ind:] = 0
            break
        else:
            # Set values: TR * [1, 2, ..., max_t]
            t_flat[t_ind:t_ind+max_t] = TR * np.arange(1, max_t + 1)
            t_ind += max_t
    # Reshape back to the original shape (using column-major order)
    t = t_flat.reshape(t.T.shape, order='F').T
    # Set values above duration_systolic to 0
    t[t > duration_systolic] = 0.0


    vecolity_at_t = Sinusoidal_wave(A, 1/duration_systolic, B, t)

    #Mask out center of k-space
    x = np.linspace(-shape[0]/2, shape[0]/2-1, shape[0])
    y = np.linspace(-shape[1]/2, shape[1]/2-1, shape[1])
    z = np.linspace(-shape[2]/2, shape[2]/2-1, shape[2])
    xv, yv, zv = np.meshgrid(y,x,z)
    vsz = np.sqrt(yv*yv+4*zv*zv)
    vsz = (vsz>20).astype(float)
    vsz = vsz 


    phaseError = phase_error_model(venc, vecolity_at_t) + pi/2

    phaseError = phaseError.reshape((shape[0], 1, shape[2]))

    phaseError = np.tile(phaseError, (1,shape[1],1))

    phaseError = -1j*np.sinh( (phaseError)*1j ) * vsz + (1-vsz)

    spect = spect * phaseError

    return spect



def addPhaseError(spect, A, f, B, TR, venc):

    shape = spect.shape

    x = np.linspace(0, shape[1]-1, shape[1])
    y = np.linspace(0, shape[2]-1, shape[2])
    xv, yv = np.meshgrid(x,y)

    #This index denotes the number of TRs from begining of acquisition
    phaseIndex = yv*shape[2] + xv
    t = phaseIndex*TR

    vecolity_at_t = Triangular_wave(A,f, B, t)
    #vecolity_at_t = Sinusoidal_wave(A, f, B, t)

    #Mask out center of k-space
    x = np.linspace(-shape[0]/2, shape[0]/2-1, shape[0])
    y = np.linspace(-shape[1]/2, shape[1]/2-1, shape[1])
    z = np.linspace(-shape[2]/2, shape[2]/2-1, shape[2])
    xv, yv, zv = np.meshgrid(x,y,z)
    vsz = np.sqrt(yv*yv+4*zv*zv)
    vsz = (vsz>40).astype(float)


    phaseError = phase_error_model(venc, vecolity_at_t)

    phaseError = phaseError.reshape((shape[0], 1, shape[2]))

    phaseError = np.tile(phaseError, (1,shape[1],1))

    phaseError = phaseError * vsz

    phaseError = np.exp(0.0+1j*phaseError)

    spect = spect * phaseError

    return spect



def perform_augmentation(root_dir_train, rotate_img):

    TR  = 9.6e-3# ms
    venc = 0

    for i, dirname in enumerate(os.listdir(root_dir_train)):
        full_path = os.path.join(root_dir_train, dirname)

        if full_path.find('aug') == -1:
            print(full_path)

            if full_path.find('VENC10') > 0:
                venc = 30
            else:
                venc = 30

            #Rosengarten, B., et al. "Comparison of visually evoked peak systolic and end diastolic blood flow velocity using a control system approach." Ultrasound in medicine & biology 27.11 (2001): 1499-1503.
            peak_systolic = random.uniform(59.0,112.5)#cm/s
            end_diastolic = random.uniform(28.2, 50.8)#cm/s
            flow_mean = end_diastolic#cm/s
            flow_flac = (peak_systolic-end_diastolic)/2.0#cm/s
            #Sapra et al., 2021 Sapra, A., Malik, A., Bhandari, P., 2021. Vital Sign Assessment, StatPearls, Treasure Island (FL).
            freq = random.uniform(1.0,1.667)#Hz
            

            readdata, header_raw = nrrd.read(os.path.join(full_path, 'image.nrrd'))
            readdata_seg, header_seg = nrrd.read(os.path.join(full_path, 'seg.nrrd'))


            if rotate_img:

                print('Rotation')
                theta = random.uniform(-0.0, 0.0)
                phi= random.uniform(-3.0, 3.0)
                rotate_recon_noise = ndimage.rotate(readdata, theta, axes=(0, 1), reshape=False)
                rotate_recon_noise = ndimage.rotate(rotate_recon_noise, phi, axes=(1, 2), reshape=False)

                rotate_recon_seg = ndimage.rotate(readdata_seg, theta, axes=(0, 1), reshape=False)
                rotate_recon_seg = ndimage.rotate(rotate_recon_seg, phi, axes=(1, 2), reshape=False)

                readdata = rotate_recon_noise
                readdata_seg = rotate_recon_seg


            spect = perform_gpu_fft(readdata)
            
            spect_noise = addPhaseError(spect, flow_flac, freq, flow_mean, TR, venc)

            
            recon_noise = perform_gpu_fft(spect_noise)
            
            recon_noise = np.flip(recon_noise)
            
            xshift = random.randint(-16, 16)
            yshift = random.randint(-16, 16)
            zshift = random.randint(-8, 8)

            recon_noise = np.roll(recon_noise, 1+zshift, axis=2)
            recon_noise = np.roll(recon_noise, 1+yshift, axis=1)
            recon_noise = np.roll(recon_noise, 1+xshift, axis=0)

            readdata_seg = np.roll(readdata_seg, zshift, axis=2)
            readdata_seg = np.roll(readdata_seg, yshift, axis=1)
            readdata_seg = np.roll(readdata_seg, xshift, axis=0)

            recon_noise = np.abs(recon_noise)


            readdata_seg = np.where(readdata_seg > 0, 1, 0)

            recon_path = full_path + '_aug'

            print(recon_path)

            if os.path.isdir(recon_path):
                print('Direcotry Exists')
            else:
                os.mkdir(recon_path)

            nrrd.write('%s/image.nrrd'%(recon_path),recon_noise.astype(np.float32),header=header_raw)
            nrrd.write('%s/seg.nrrd'%(recon_path),readdata_seg.astype(np.float32),header=header_raw)



def save_training_dataset(root_dir_train, output_dir):


    for i, dirname in enumerate(os.listdir(root_dir_train)):
        full_path = os.path.join(root_dir_train, dirname)

        readdata, header_raw = nrrd.read(os.path.join(full_path, 'image.nrrd'))
        readdata = readdata.astype(np.float16)
        patched_img = patch_img(readdata,[128,128,32],[64,64,16])
        dim_arr = patched_img[0].shape
        patched_img = np.reshape(patched_img[0], (dim_arr[0]*dim_arr[1]*dim_arr[2],dim_arr[3],dim_arr[4],dim_arr[5]))

        readdata2, header_seg = nrrd.read(os.path.join(full_path, 'seg.nrrd'))
        readdata2 = readdata2.astype(np.float16)
        readdata2[readdata2>0] = 1
        patched_seg = patch_img(readdata2,[128,128,32],[64,64,16])
        dim_arr = patched_seg[0].shape
        patched_seg = np.reshape(patched_seg[0], (dim_arr[0]*dim_arr[1]*dim_arr[2],dim_arr[3],dim_arr[4],dim_arr[5]))
        patched_segN = 1.0-patched_seg

        header_raw['sizes'][0] = 128
        header_raw['sizes'][1] = 128
        header_raw['sizes'][2] = 32

        ind_size = 100
        ind_rand = np.random.permutation(patched_segN.shape[0]-1)[:ind_size]

        output_path = os.path.join(output_dir, dirname)

        if os.path.isdir(output_path):
            print('Direcotry Exists')
        else:
            os.mkdir(output_path)
            
        for sl in range(ind_size):

            raw_image = np.squeeze(patched_img[ind_rand[sl],:,:,:])
            raw_image = np.reshape(raw_image, (header_raw['sizes'][0]*header_raw['sizes'][1]*header_raw['sizes'][2],1))
            normalized_image = StandardScaler().fit_transform(raw_image)
            max_val = np.max(np.abs(normalized_image))
            normalized_image = np.reshape(normalized_image,(header_raw['sizes'][0],header_raw['sizes'][1],header_raw['sizes'][2]))

            mu, sigma = 0, 0.01*max_val # mean and standard deviation
            noise_mat = np.random.normal(mu, sigma, (header_raw['sizes'][0],header_raw['sizes'][1],header_raw['sizes'][2]))

            normalized_image = normalized_image + noise_mat

            nrrd.write('%s/image_%d.nrrd'%(output_path,sl),normalized_image.astype(np.double),header=header_raw)
            nrrd.write('%s/seg_%d.nrrd'%(output_path,sl),np.squeeze(patched_segN[ind_rand[sl],:,:,:]).astype(np.short),header=header_raw)




#perform_augmentation_3axes() function:
#Input:
#   root_dir_train: Directory of training datasets
#   rotate_img: Flag for rotation agumentation
#   random_phase: Flag for PA agumentation
#   pattern_ind: Augmentation pattern index 
#   rand_seed: Seeds for random function
def perform_augmentation_3axes(root_dir_train, rotate_img, random_phase, pattern_ind, rand_seed):

    TR  = 9.6e-3# ms
    venc = 0

    random.seed(rand_seed)

    for i, dirname in enumerate(os.listdir(root_dir_train)):
        full_path = os.path.join(root_dir_train, dirname)

        if full_path.find('aug') == -1:
            print(full_path)

            if full_path.find('VENC10') > 0:
                venc = 10
            else:
                venc = 10

            #Rosengarten, B., et al. "Comparison of visually evoked peak systolic and end diastolic blood flow velocity using a control system approach." Ultrasound in medicine & biology 27.11 (2001): 1499-1503.
            end_diastolic = 21 + random.uniform(-5.4, 5.4)#random.uniform(0, 21 + random.uniform(-5.4, 5.4))#cm/s
            peak_systolic = 57 + random.uniform(-12.3, 12.3)#random.uniform(0, 57 + random.uniform(-12.3, 12.3))#cm/s
            flow_mean = 0.0#cm/s
            flow_flac = random.uniform(0, peak_systolic-end_diastolic)#cm/s
            if pattern_ind%2 == 0:
                flow_flac = peak_systolic-end_diastolic#cm/s
            else:
                flow_flac = 0#cm/s
            #Sapra et al., 2021 Sapra, A., Malik, A., Bhandari, P., 2021. Vital Sign Assessment, StatPearls, Treasure Island (FL).
            freq = random.uniform(1.0,1.667)#Hz
            
            readdata_x, header_raw = nrrd.read(os.path.join(full_path, 'image_x.nrrd'))
            readdata_y, header_raw = nrrd.read(os.path.join(full_path, 'image_y.nrrd'))
            readdata_z, header_raw = nrrd.read(os.path.join(full_path, 'image_z.nrrd'))
            readdata_seg, header_seg = nrrd.read(os.path.join(full_path, 'seg.nrrd'))

            dimk = readdata_x.shape
            orig_4d = np.zeros((dimk[0],dimk[1],dimk[2],3))
            orig_4d[:,:,:,0] = readdata_x
            orig_4d[:,:,:,1] = readdata_y
            orig_4d[:,:,:,2] = readdata_z

            nrrd.write('%s/image_3axes.nrrd'%(full_path),orig_4d.astype(np.float32),header=header_raw)

            #Swap array
            if random.random() < 0.0:
                readdata_x = np.transpose(readdata_x, (0, 2, 1)) 
                readdata_y = np.transpose(readdata_y, (0, 2, 1)) 
                readdata_z = np.transpose(readdata_z, (0, 2, 1)) 
                readdata_seg = np.transpose(readdata_seg, (0, 2, 1))
                #orig_4d = np.transpose(orig_4d, (0, 2, 1, 3))

                readdata_x = ndimage.zoom(readdata_x, zoom=[1, 2, 0.5], order=2)
                readdata_y = ndimage.zoom(readdata_y, zoom=[1, 2, 0.5], order=2)
                readdata_z = ndimage.zoom(readdata_z, zoom=[1, 2, 0.5], order=2)
                readdata_seg = ndimage.zoom(readdata_seg, zoom=[1, 2, 0.5], order=2)
                readdata_seg = (readdata_seg >= 0.5).astype(np.float32)
                #orig_4d = ndimage.zoom(orig_4d, zoom=[1, 2, 0.5, 1], order=2)

                dimk = readdata_x.shape


            if rotate_img:

                print('Rotation')

                scale_x = random.uniform(0.9, 1.1)
                scale_y = random.uniform(0.9, 1.1)
                scale_z = random.uniform(0.9, 1.1)
                rotate_recon_noise_x = ndimage.zoom(readdata_x, (scale_x, scale_y, scale_z))
                rotate_recon_noise_y = ndimage.zoom(readdata_y, (scale_x, scale_y, scale_z))
                rotate_recon_noise_z = ndimage.zoom(readdata_z, (scale_x, scale_y, scale_z))
                rotate_recon_seg = ndimage.zoom(readdata_seg, (scale_x, scale_y, scale_z))

                readdata_x = crop_center(rotate_recon_noise_x,dimk[0],dimk[1],dimk[2])
                readdata_y = crop_center(rotate_recon_noise_y,dimk[0],dimk[1],dimk[2])
                readdata_z = crop_center(rotate_recon_noise_z,dimk[0],dimk[1],dimk[2])
                readdata_seg = crop_center(rotate_recon_seg,dimk[0],dimk[1],dimk[2])

            
            recon_4d = np.zeros((dimk[0],dimk[1],dimk[2],3))
            

            spect = perform_gpu_fft(readdata_x)
            temp = addErrors(spect, flow_flac, freq, flow_mean, TR, venc)
            if random_phase:
                spect_noise = temp
            else:
                spect_noise = spect
            recon_noise = perform_gpu_fft(spect_noise)
            recon_noise = np.flip(recon_noise)
            recon_4d[:,:,:,0] = np.abs(recon_noise)


            spect = perform_gpu_fft(readdata_y)
            temp = addErrors(spect, flow_flac, freq, flow_mean, TR, venc)
            if random_phase:
                spect_noise = temp
            else:
                spect_noise = spect
            recon_noise = perform_gpu_fft(spect_noise)
            recon_noise = np.flip(recon_noise)
            recon_4d[:,:,:,1] = np.abs(recon_noise)

            spect = perform_gpu_fft(readdata_z)
            temp = addErrors(spect, flow_flac, freq, flow_mean, TR, venc)
            if random_phase:
                spect_noise = temp
            else:
                spect_noise = spect
            recon_noise = perform_gpu_fft(spect_noise)
            recon_noise = np.flip(recon_noise)
            recon_4d[:,:,:,2] = np.abs(recon_noise)


            xshift = random.randint(-16, 16)
            yshift = random.randint(-16, 16)
            zshift = random.randint(-8, 8)

            recon_4d = np.roll(recon_4d, 1+zshift, axis=2)
            recon_4d = np.roll(recon_4d, 1+yshift, axis=1)
            recon_4d = np.roll(recon_4d, 1+xshift, axis=0)

            readdata_seg = np.roll(readdata_seg, zshift, axis=2)
            readdata_seg = np.roll(readdata_seg, yshift, axis=1)
            readdata_seg = np.roll(readdata_seg, xshift, axis=0)

            recon_4d = np.abs(recon_4d)


            readdata_seg = np.where(readdata_seg > 0, 1, 0)

            recon_path = full_path + '_aug_p%d'%(pattern_ind)

            print(recon_path)

            if os.path.isdir(recon_path):
                print('Direcotry Exists')
            else:
                os.mkdir(recon_path)

            print(recon_4d.shape)
            nrrd.write('%s/image_3axes.nrrd'%(recon_path),recon_4d.astype(np.float32),header=header_raw)
            nrrd.write('%s/seg.nrrd'%(recon_path),readdata_seg.astype(np.float32),header=header_raw)
            



def save_training_datasetn_3axes(root_dir_train, output_dir, random_seed):

    

    for i, dirname in enumerate(os.listdir(root_dir_train)):
        full_path = os.path.join(root_dir_train, dirname)

        print(full_path)


        readdata, header_raw = nrrd.read(os.path.join(full_path, 'image_3axes.nrrd'))
        readdata = readdata.astype(np.float16)
        dim_ori = readdata.shape
        stride_patch = [2 ** round(log(dim / 8, 2)) for dim in dim_ori[:3]]
        print(stride_patch)
        patched_img = patch_img_3ch(readdata,[128,128,32],stride_patch)
        dim_arr = patched_img[0].shape
        patched_img = np.reshape(patched_img[0], (dim_arr[0]*dim_arr[1]*dim_arr[2],dim_arr[3],dim_arr[4],dim_arr[5],dim_arr[6]))

        readdata2, header_seg = nrrd.read(os.path.join(full_path, 'seg.nrrd'))
        readdata2 = readdata2.astype(np.float16)
        readdata2[readdata2>0] = 1
        patched_seg = patch_img(readdata2,[128,128,32],stride_patch)
        dim_arr = patched_seg[0].shape
        patched_seg = np.reshape(patched_seg[0], (dim_arr[0]*dim_arr[1]*dim_arr[2],dim_arr[3],dim_arr[4],dim_arr[5]))
        patched_segN = 1.0-patched_seg

        header_raw['sizes'][0] = 3
        header_raw['sizes'][1] = 128
        header_raw['sizes'][2] = 128
        header_raw['sizes'][3] = 32
   

        header_seg['sizes'][0] = 128
        header_seg['sizes'][1] = 128
        header_seg['sizes'][2] = 32

        ind_size = 75
        np.random.seed(random_seed + i)
        random.seed(random_seed + i)
        ind_rand = np.random.permutation(patched_segN.shape[0]-1)[:ind_size]
        print(random_seed + i)
        print(ind_rand)

        output_path = os.path.join(output_dir, dirname)

        if os.path.isdir(output_path):
            print('Direcotry Exists')
        else:
            os.mkdir(output_path)
            
        
        for sl in range(ind_size):
            raw_image = np.squeeze(patched_img[ind_rand[sl],:,:,:,:])
            raw_image = np.reshape(raw_image, (header_raw['sizes'][0]*header_raw['sizes'][1]*header_raw['sizes'][2]*header_raw['sizes'][3],1))
            normalized_image = StandardScaler().fit_transform(raw_image)
            max_val = np.max(np.abs(normalized_image))
            normalized_image = np.reshape(normalized_image,(header_raw['sizes'][1],header_raw['sizes'][2],header_raw['sizes'][3],header_raw['sizes'][0]))

            mu, sigma = 0, 0.01*max_val*random.uniform(0.1, 1) # mean and standard deviation
            noise_mat = np.random.normal(mu, sigma, (header_raw['sizes'][1],header_raw['sizes'][2],header_raw['sizes'][3],header_raw['sizes'][0]))

            normalized_image = normalized_image + noise_mat

            flipped_seg = np.squeeze(patched_segN[ind_rand[sl],:,:,:]).astype(np.short)

            if random.random() < 0.5:
                normalized_image = np.flip(normalized_image, axis=0)  # Flip along the RO direction
                flipped_seg = np.flip(flipped_seg, axis=0)  # Flip along the RO direction

            normalized_image = normalized_image.transpose(3,0,1,2)

            nrrd.write('%s/image_%d.nrrd'%(output_path,sl),normalized_image.astype(np.double),header=header_raw)
            nrrd.write('%s/seg_%d.nrrd'%(output_path,sl),flipped_seg,header=header_seg)

        
#The code has been modified from the original, as the original utilized GE's proprietary library.
#This version utilizes a magnitude NRRD input rather than a complex array.
if __name__ == '__main__':

    #Augementation without PA augmentation
    #for p_ind in range(16):
    #    perform_augmentation_3axes(sys.argv[1], True, False, p_ind, 123+p_ind*20)
    #save_training_datasetn_3axes(sys.argv[1], sys.argv[2], 123)
    
    #Augementation with PA augmentation
    for p_ind in range(16):
        perform_augmentation_3axes(sys.argv[1], True, True, p_ind, 123+p_ind*20)
    save_training_datasetn_3axes(sys.argv[1], sys.argv[2], 123)

