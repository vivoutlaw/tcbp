from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable 
import torchvision.models as models

import os
import argparse
import random
import h5py
import time
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.serialization import load_lua
import matplotlib.pyplot as plt 
import scipy.io as sio

sys.path.insert(0, '/home/vsharma/Documents/Audio_Visual_Text/codes_to_extract_features/video-classification-3d-cnn-pytorch/models')
import resnet, resnext

sys.path.insert(0, '/home/vsharma/Documents/Audio_Visual_Text/codes_to_extract_features/pytorch_i3d')
from pytorch_i3d import InceptionI3d

sys.path.insert(0, '/export/data/TrecVid_INS_2018/pyannote-video')
from pyannote.video import Video
import pdb


# Python3
import sys
sys.path.insert(0,'/home/vsharma/temp/audio/build/lib.linux-x86_64-3.5')

#Python2 
#sys.path.insert(0,'/home/vsharma/temp/audio/build/lib.linux-x86_64-2.7')

# For audio preprocessing
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import librosa
import torchaudio

import readline
from rlcompleter import Completer
readline.parse_and_bind("tab: complete")
readline.set_completer(Completer(locals()).complete)


##############
#
# Example :  CUDA_VISIBLE_DEVICES=2 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=0 --end_movie_num=1 
# krenew -b -- screen -Dm -fn
# screen -r
# cd /home/vsharma/Documents/Audio_Visual_Text
# source ~/.bashrc
# CUDA_VISIBLE_DEVICES=0 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=0 --end_movie_num=25
# CUDA_VISIBLE_DEVICES=0 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=25 --end_movie_num=50
# CUDA_VISIBLE_DEVICES=1 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=50 --end_movie_num=75
# CUDA_VISIBLE_DEVICES=1 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=75 --end_movie_num=100
# CUDA_VISIBLE_DEVICES=2 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=100 --end_movie_num=125
# CUDA_VISIBLE_DEVICES=2 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=125 --end_movie_num=150
# CUDA_VISIBLE_DEVICES=3 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=150 --end_movie_num=175
# CUDA_VISIBLE_DEVICES=3 python3.5 Audio_Visual_Place_Temporal.py --start_movie_num=175 --end_movie_num=204
##############

#######################################################
#
# Generates segments of pairs for feature extraction.
#
#######################################################

def generate_start_end_index(last_idx, seg_interval, sub_seg_interval):
    end_idx = last_idx
    duration = end_idx/seg_interval
    ratio = last_idx % seg_interval
    additional_frames_needed = seg_interval - ratio
    x1 = [x for x in range(0,last_idx,seg_interval)]
    x2 = [x for x in range(0,last_idx,sub_seg_interval) if (x % seg_interval)]
    temp_pairs_1 = np.concatenate((x1,x2),axis=0)
    temp_pairs_1.sort()
    temp_pairs_2 = [x+seg_interval for x in temp_pairs_1 if (x+seg_interval)<last_idx]
    if len(temp_pairs_1) == len(temp_pairs_2):
        temp_pairs_1 = temp_pairs_1
        temp_pairs_2 = temp_pairs_2
    else:
        temp_pairs_1 = temp_pairs_1[0:len(temp_pairs_2)+1]
        temp_pairs_2 = np.concatenate((temp_pairs_2,[last_idx]))
    return temp_pairs_1.astype(int), temp_pairs_2.astype(int)


####################################################
#
# Generic feature extractor for a given layer
# Currently used for imagenet and places.
# Reference: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
# Other way: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/2
####################################################

def generic_layer_feat_extractor(model, layer_name, output_embedding):
    # Use the model object to select the desired layer
    layer = model._modules.get(layer_name)
    model.eval()
    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        output_embedding.copy_(o.data)
    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Not removing the hook
    # h.remove()
    return model


####################################################
#
# Soundnet8
#
####################################################


class SoundNet8(nn.Module):

    def __init__(self, features):
        super(SoundNet8, self).__init__()
        self.features = features

    def forward(self, x):
        # only use the features from the pretrained model
        self.features.forward(x)
        conv7_feat = self.features.modules[21].output.cuda().squeeze() # cpu cuda 
        pool5_feat = self.features.modules[17].output.cuda().squeeze() # cpu cuda
        return conv7_feat, pool5_feat


def load_pretrained_soundnet8():
    """Loads pre-trained SoundNet8 model
    """
    MODEL_PATH = '/home/vsharma/Documents/Audio_Visual_Text/models/makarand_soundnet/soundnet8_final_cpu_nosqueeze.t7'
    net = load_lua(MODEL_PATH)
    net.remove()  # remove ParallelTable
    net.remove()  # remove FlattenTable
    net.remove()  # remove ParallelTable
    net.remove()  # remove ConcatTable
    model = SoundNet8(net)
    #model.cuda()
    return model


def load_one_audio(audio_filename):
	'''Given the file path, return normalized proper audio sample
    '''
	rate, sig = wavfile.read(audio_filename)
	# print sig.dtype, sig.max(), sig.min()
	# convert to float, bring to -1..1 range, divide by bitrate
	if sig.dtype == 'uint8':
	    sig = sig.astype('float32')/(2**7) - 1  # originally in 0..255, divide by 128 to get 0..2, subtract 1
	elif sig.dtype == 'int16':
	    sig = sig.astype('float32')/(2**15)  # originally in -32768..32767, divide by 32768 to get -1..1
	elif sig.dtype == 'int32':
	    sig = sig.astype('float32')/(2**31) # original: -2^31..(2^31-1), divide by 2^31 to get -1..1
	else:
	    pdb.set_trace()

	# get rid of stereo
	if sig.ndim > 1:
	    sig = sig.mean(1)

	# final signal wants to be in -256..256, multiply by 2^8
	sig *= 2**8

	# resample to 22,050 Hz: 1 second video
	# sig = resample(sig, 22050)

	# Handles N seconds 
	duration = (sig.shape[0]*1.0 / rate)
	#print(duration)
	sig = resample(sig, int(np.round(22050*duration)))

	# shape is CH=1 x DIM x 1
	sig = torch.from_numpy(sig.astype('float32'))
	sig = sig.contiguous().view(1, -1, 1)

	return sig

def load_audio_librosa(audio_filename):
	y, sr = librosa.load(audio_filename, sr=None)
	duration_audio = librosa.get_duration(y=y, sr=sr)
	y *= 2**8
	y = librosa.resample(y, sr, 22050)

	# shape is CH=1 x DIM x 1
	y = torch.from_numpy(y.astype('float32'))
	y = y.contiguous().view(1, -1, 1)

	return y, duration_audio

def load_audio_all_methods(audio_filename):
	TORCHAUDIO = False
	SCIPY = False
	LIBROSA = False

	if TORCHAUDIO:
		###### TORCHAUDIO
		# HAS NO RESAMPLING OPTION
		sound, sample_rate = torchaudio.load(audio_filename)
		# sound -> torch.float32
		sound = sound.numpy()/(2**31)
		# sound.shape -> torch.Size([XX, 2])
		sound = sound.mean(1)
		sig = sound


	if SCIPY:
		####### SCIPY
		#
		rate, sig = wavfile.read(audio_filename)
		#print(sig.dtype)
		# sig -> dtype('int16')
		sig = sig.astype('float32')/(2**15)
		#print(sig.shape)
		# sig.shape -> (XX, 2)
		sig = sig.mean(1)
		#print ('{} {}'.format(np.min(sig),np.max(sig)))
		sig *= 2**8
		duration_sig = (sig.shape[0]*1.0 / rate)
		#print(duration_sig)
		sig = resample(sig, int(np.round(22050*duration_sig)))
		#print ('{} {}'.format(np.min(sig),np.max(sig)))
		#print(sig.shape)

	if LIBROSA:
		###### LIBROSA
		#
		y, sr = librosa.load(audio_filename, sr=None)
		#print(y.dtype)
		# y -> dtype('float32')
		#print(y.shape)
		# y.shape -> (XX,)
		#print ('{} {}'.format(np.min(y),np.max(y)))
		y *= 2**8
		#duration_y = librosa.get_duration(y=y, sr=sr)
		#print(duration_y)
		y = librosa.resample(y, sr, 22050)
		#print ('{} {}'.format(np.min(y),np.max(y)))
		#print(y.shape)
		sig = y

	return sig


####################################################
#
# Places
# https://github.com/CSAILVision/places365/blob/master/run_placesCNN_unified.py
####################################################

def load_places_model():
	path_to_places = '/home/vsharma/Documents/Audio_Visual_Text/models/places365'
	# th architecture to use
	arch = 'resnet50'
	# load the pre-trained weights
	model_file = '{}/{}_places365.pth.tar'.format(path_to_places,arch)
	model = models.__dict__[arch](num_classes=365)
	checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
	state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
	model.load_state_dict(state_dict)
	return model

####################################################
#
# ResNet50, ResNext101, and I3D models 
# https://github.com/kenshohara/video-classification-3d-cnn-pytorch
# --resnet_shortcut B
# https://github.com/piergiaj/pytorch-i3d/
####################################################

def load_temporal_model(model_name, model_depth):
	verbose = False
	#model_name = 'i3d' # resnext resnet i3d

	if model_name == 'i3d':
		model_path = '/home/vsharma/Documents/Audio_Visual_Text/models/i3d/rgb_imagenet.pt'
		model = InceptionI3d(400, in_channels=3)
		model.load_state_dict(torch.load(model_path))
		arch = model_name
		model.train(False)  # Set model to evaluate mode

	elif (model_name == 'resnet') or (model_name == 'resnext'):
		#model_depth = 50 # 101 50
		arch = '{}-{}'.format(model_name, model_depth)
		model_path = '/home/vsharma/Documents/Audio_Visual_Text/models/resnet3d'
		model_path = '{}/{}-kinetics.pth'.format(model_path,arch)
		
		if arch=='resnet-50':
			model = resnet.resnet50(num_classes=400, shortcut_type='B',
	                                sample_size=112, sample_duration=16,
	                                last_fc=True)
		elif arch=='resnext-101':
			model = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
	                                sample_size=112, sample_duration=16,
	                                last_fc=True)

		model_data = torch.load(model_path)
		assert arch == model_data['arch']

		#model.load_state_dict(model_data['state_dict'])
		state_dict = model_data['state_dict']
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:]  # remove `module.`
			new_state_dict[name] = v
		model.load_state_dict(new_state_dict)

		# Removing the last 2 layers: fc and softmax
		model = nn.Sequential(*list(model.children())[:-2])
		model.eval()

	if verbose:
		print(model)

	return model



################################
#
#
# Main
#
#
################################


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Feature Extraction')
	parser.add_argument('--start_movie_num', default=0, type=int, help='Staring movie index')
	parser.add_argument('--end_movie_num', default=1, type=int, help='Ending movie index')
	args = parser.parse_args()

	load_video_models 		= True
	load_audio_models 		= True
	load_imagenet_models 	= True
	load_places_models 		= True
	extract_feats 			= True

	DIR_PATH = '/cvhci/data/QA/Movie_Description_Dataset/MOVIE_DESCRIPTION_DATASET'
	MOVIE_NAMES = [ file for file in os.listdir(DIR_PATH) if os.path.isdir(os.path.join(DIR_PATH, file))]
	OUTPUT_DIR = '/cvhci/data/QA/Movie_Description_Dataset/Features'
	# Raw Image
	OUTPUT_DIR_RAW_IMAGE = '/cvhci/data/QA/Movie_Description_Dataset/Raw_Image'

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Temporal model input_sizes
	i3d_imagenet_expected_size = 224
	resnet3d_expected_size = 112

	################################
	#
	#
	# ALL transformations and normalisation
	#
	#
	################################

	# Transformation steps
	trn_resnet = transforms.Compose([
		transforms.Resize((resnet3d_expected_size, resnet3d_expected_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.450, 0.422, 0.390], [1, 1, 1])])
		#transforms.Normalize([114.7748/255., 107.7354/255., 99.4750/255.], [1, 1, 1])])

	trn_i3d = transforms.Compose([
		transforms.Resize((i3d_imagenet_expected_size, i3d_imagenet_expected_size)),
		transforms.ToTensor()])

	trn_raw_img = transforms.Compose([
		transforms.Resize((512, 512)),
		transforms.ToTensor()])

	trn_imagenet = transforms.Compose([
	    transforms.Resize((i3d_imagenet_expected_size, i3d_imagenet_expected_size)),
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


	################################
	#
	#
	# ALL models details:
	# ImageNet: ResNet50. 2048x7x7 
	# Places365: ResNet50, 2048x7x7
	# I3d: ImageNet + Kinectics-400. BN-Inception. 1024x8x7x7
	# ResNet3D: ResNet50. 2048x1x7x7 
	# Soundnet: 1024x, 256x
	#
	#
	################################

	if load_video_models:
		# Temporal: ResNet(16frames): 1x2048x1x4x4, ImageNet(64frames): 1x1024x8x7x7
		model_name = 'resnet' 		# resnext resnet i3d
		if (model_name == 'resnet') or (model_name == 'resnext'):
			model_depth =  50 if (model_name == 'resnet') else 101	#  101		50	 
			model_temporal_resnet = load_temporal_model(model_name,model_depth)
			model_temporal_resnet = model_temporal_resnet.to(device)
			model_temporal_resnet.eval()
			
			#x = torch.rand([1,3,16,112, 112]).to(device) # Resnet
			#y = model_temporal_resnet(x) # Resnet: [1,2048,1,4,4]  
			#print(y.shape)

		model_name = 'i3d'
		if  model_name == 'i3d':
			model_depth = 0 # Doesn't matter
			model = load_temporal_model(model_name,model_depth)
			layer_name = 'Mixed_5c' # Mixed_5c --> [1, 1024, 2, 7, 7], avg_pool --> [1, 1024, 1, 1, 1]
			#video_embedding = torch.zeros([1,1024,2,7,7]) # I3D: [1, 1024, 2, 7, 7] 
			video_embedding = torch.zeros([1,1024,8,7,7]) # For 64 frames ->> 8  
			model_temporal_i3d = generic_layer_feat_extractor(model, layer_name, video_embedding)
			model_temporal_i3d = model_temporal_i3d.to(device)
			model_temporal_i3d.eval()

			#x = torch.rand([1,3,64,224, 224]).to(device)
			#model_temporal_i3d(x) # I3D: [1, 1024, 8, 7, 7] 
			#print(video_embedding.shape)

	if load_places_models:
		# Places : 1x2048x7x7
		model_places = load_places_model()
		model_places = nn.Sequential(*list(model_places.children())[:-2])
		model_places = model_places.to(device)
		model_places.eval()

	if load_imagenet_models: 
		# ImageNet : 1x2048x7x7
		model_imagenet = models.resnet50(pretrained=True)
		model_imagenet = nn.Sequential(*list(model_imagenet.children())[:-2])
		model_imagenet = model_imagenet.to(device)
		model_imagenet.eval()

		# Extracting avgpool features
		model_imagenet_fc = models.resnet50(pretrained=True)
		model_imagenet_fc = nn.Sequential(*list(model_imagenet_fc.children())[:-1])
		model_imagenet_fc = model_imagenet_fc.to(device)
		model_imagenet_fc.eval()

	if load_audio_models:
		# Audio : 1024x, 256x
		#print('Load model...')
		model_sound = load_pretrained_soundnet8()
		model_sound = model_sound.to(device)
		model_sound.eval()
		#au = load_one_audio(audio_file_path)
		#au = au.unsqueeze(0) # batch-size 1
		#conv7_feat, pool5_feat = model_sound(au)
		#print('{} {}'.format(conv7_feat.shape,pool5_feat.shape))

	################################
	#
	# Loading video to extract features
	#
	################################

	for m in range(args.start_movie_num, args.end_movie_num):

		INPUT_PATH = os.path.join(DIR_PATH,MOVIE_NAMES[m])
		print(INPUT_PATH)

		for root, dirs, files in os.walk(INPUT_PATH):
			for basefile_name_w_ext in tqdm(files):
				if basefile_name_w_ext.endswith('.avi'): # Checks for avi file format.
					# create output file
					basefile_name = os.path.splitext(basefile_name_w_ext)[0] 
					foldername = root.split('/')[-1]

					target_file_path = os.path.join(OUTPUT_DIR, foldername, basefile_name + '.h5')
					target_file_path_raw_img = os.path.join(OUTPUT_DIR_RAW_IMAGE, foldername, basefile_name + '.mat')

					if not os.path.exists(os.path.dirname(target_file_path)):
						os.makedirs(os.path.dirname(target_file_path))

						# Raw Image 
						os.makedirs(os.path.dirname(target_file_path_raw_img))

					if os.path.exists(target_file_path):
						continue
					
					# process video and audio files
					#video_filename = '/cvhci/data/QA/Movie_Description_Dataset/MOVIE_DESCRIPTION_DATASET/0001_American_Beauty/0001_American_Beauty_01.55.05.110-01.55.21.331.avi'
					#audio_filename = '/cvhci/data/QA/Movie_Description_Dataset/MOVIE_DESCRIPTION_DATASET/0001_American_Beauty/0001_American_Beauty_01.55.05.110-01.55.21.331.wav'
					video_filename = os.path.join(root,basefile_name_w_ext) 		# Video
					audio_filename = os.path.join(root, basefile_name + '.wav')		# Audio

					video = Video(video_filename)
					num_frames = video._nframes
					width = video._width
					height = video._height
					channels = 3
					black_border_removed_indexes = 0 # Used afterwards
					step_size_imagenet = 5

					# Buffer memory to store frames for for fast accessing
					frames_buffer_i3d = torch.zeros([num_frames, channels, i3d_imagenet_expected_size, i3d_imagenet_expected_size], dtype=torch.float32)
					frames_buffer_i3d = frames_buffer_i3d.to(device)

					frames_buffer_resnet = torch.zeros([num_frames, channels, resnet3d_expected_size, resnet3d_expected_size], dtype=torch.float32)
					frames_buffer_resnet = frames_buffer_resnet.to(device)

					# +1 for saving the zeroth index frame
					zeroth_and_last_index = 2 # 1+1
					num_places_imagenet_frames = int(np.floor(num_frames/step_size_imagenet))+zeroth_and_last_index
					frames_buffer_places_imagenet = torch.zeros([num_places_imagenet_frames, channels, i3d_imagenet_expected_size, i3d_imagenet_expected_size], dtype=torch.float32)
					frames_buffer_places_imagenet = frames_buffer_places_imagenet.to(device)

					# Raw Image 
					frames_buffer_raw_image = torch.zeros([num_places_imagenet_frames, channels, 512, 512], dtype=torch.float32)

					#print(int(np.floor(num_frames/step_size_imagenet)))
					SIGNAL_BORDER_DETECTION=True
					k=0
					for idx, (t, frame) in enumerate(video):

						if SIGNAL_BORDER_DETECTION:
							temp_Frame = frame.mean(2)
							temp_Frame = temp_Frame.mean(1)
							# Stores indexes for the part of an image, without top and lower black boarders 
							black_border_removed_indexes = [tf for tf in range(temp_Frame.shape[0]) if not (temp_Frame[tf]==0.)]

							if len(black_border_removed_indexes) < 600: # if less than 600 height, save whole image, example 1014_2012_02.23.30.972-02.23.40.344.avi
								black_border_removed_indexes = [tf for tf in range(temp_Frame.shape[0])]
								SIGNAL_BORDER_DETECTION= True
								#print(idx)
							else:
								SIGNAL_BORDER_DETECTION= False
								#print(black_border_removed_indexes)
								#print(len(black_border_removed_indexes))
								#print(idx)
								#exit()
							#print(black_border_removed_indexes)

						# Removal of black boarders
						img = frame[black_border_removed_indexes,:,:]

						# Scaling between -1:1 for I3D
						img_i3d = (img/255.)*2 - 1
						img_i3d = Image.fromarray(img_i3d, mode='RGB')

						# For Resnet and ImageNet, non scaled version
						img = Image.fromarray(img, mode='RGB')
						#img.save('/home/vsharma/Desktop/dump/{}.jpg'.format(idx))

						# Storing frames in a buffer memory 
						frames_buffer_i3d[idx,:,:,:] = trn_i3d(img_i3d).to(device)
						frames_buffer_resnet[idx,:,:,:] = trn_resnet(img).to(device)

						# Storing every 5th frame
						if idx % step_size_imagenet == 0:
						#	print('{} --> {}'.format(idx,k))
							frames_buffer_places_imagenet[k,:,:,:] = trn_imagenet(img).to(device) 

							# Raw Image 
							frames_buffer_raw_image[k,:,:,:] = trn_raw_img(img)
							k+=1

						# Also storing the last frame, could be repeating when the frames are exact multple of step_size_imagenet=5
						if idx == num_frames-1:
							#print('idx: {}, Num_Frames: {}'.format(idx, num_frames-1))
							frames_buffer_places_imagenet[k,:,:,:] = trn_imagenet(img).to(device)

							# Raw Image 
							frames_buffer_raw_image[k,:,:,:] = trn_raw_img(img)
							k+=1
							#print('Last Frame extracted {}'.format(k))

					################################
					#
					# Variables to store features
					#
					################################
					
					# i3d: 64 frames with stepsize of 32
					s_i3d, e_i3d = generate_start_end_index(num_frames, 64, 32)
					num_pairs_i3d = len(s_i3d)
					video_feats_i3d = np.zeros([num_pairs_i3d,1024,8,7,7], dtype='float32')

					# Resnet: 16 frames with stepsize of 8
					s_resnet, e_resnet = generate_start_end_index(num_frames, 16, 8)
					num_pairs_resnet = len(s_resnet)
					video_feats_resnet = np.zeros([num_pairs_resnet,2048,1,4,4], dtype='float32')

					# Places and ImageNet: 32 frames with stepsize of 32, that means no features are extracted for same frames
					s_places_imagenet, e_places_imagenet = generate_start_end_index(num_places_imagenet_frames, 32, 32)
					num_pairs_places_imagenet = len(s_places_imagenet)
					video_feats_places = np.zeros([num_places_imagenet_frames,2048,7,7], dtype='float32')
					video_feats_imagenet = np.zeros([num_places_imagenet_frames,2048,7,7], dtype='float32')

					# Raw Image 
					frames_buffer_raw_image = frames_buffer_raw_image.numpy()
					video_feats_imagenet_fc = np.zeros([num_places_imagenet_frames,2048,1,1], dtype='float32')


					################################
					#
					# Extract features for all models
					#
					################################

					if extract_feats:
						with torch.no_grad():

							#print('Extracting I3D Features')
							# Extract i3d features
							for i in range(num_pairs_i3d):
								V = frames_buffer_i3d[s_i3d[i]:e_i3d[i],:,:,:] # BxCxTxHxW, T=64
								if len(V) == 64:
									temp_V = V
								else:
									#print('{}/{}'.format(s_i3d[i],e_i3d[i]))
									# Handling last set of images, not multiple of 64, by repeating last_frame
									last_frame = V[len(V)-1,:,:,:]
									last_frame = last_frame.unsqueeze(3).repeat(1, 1, 1, 64-len(V))
									last_frame = last_frame.permute(3,0,1,2)
									temp_V = torch.cat((V,last_frame),dim=0)
								temp_V = temp_V.permute(1,0,2,3).unsqueeze(0)
								model_temporal_i3d(temp_V)
								video_feats_i3d[i,:,:,:,:] = video_embedding.squeeze(0).data.cpu().numpy()
								#print('i3d: {}'.format(feat_V))

							#print('Extracting ResNet-50 Features')
							# Extract Resnet features
							for i in range(num_pairs_resnet):
								V = frames_buffer_resnet[s_resnet[i]:e_resnet[i],:,:,:] # BxCxTxHxW, T=16	
								if len(V) == 16:
									temp_V = V
								else:
									#print('{}/{}'.format(s_resnet[i],e_resnet[i]))
									# Handling last set of images, not multiple of 16, by repeating last_frame
									last_frame = V[len(V)-1,:,:,:]
									last_frame = last_frame.unsqueeze(3).repeat(1, 1, 1, 16-len(V))
									last_frame = last_frame.permute(3,0,1,2)
									temp_V = torch.cat((V,last_frame),dim=0)
								temp_V = temp_V.permute(1,0,2,3).unsqueeze(0)
								feat_V = model_temporal_resnet(temp_V)
								video_feats_resnet[i,:,:,:,:] = feat_V.squeeze(0).data.cpu().numpy()
								#print('Resnet: {}'.format(feat_V))
							#print('{} {}'.format(video_feats_i3d, video_feats_resnet))

							#print('Extracting Places and ImageNet Features')
							# Extract Places and ImageNet features
							for i in range(num_pairs_places_imagenet):
								V = frames_buffer_places_imagenet[s_places_imagenet[i]:e_places_imagenet[i],:,:,:] # BxCxHxW, B=32
								feat_P = model_places(V)
								video_feats_places[s_places_imagenet[i]:e_places_imagenet[i],:,:,:] = feat_P.data.cpu().numpy()
								feat_I = model_imagenet(V)
								video_feats_imagenet[s_places_imagenet[i]:e_places_imagenet[i],:,:,:] = feat_I.data.cpu().numpy()

								# Raw Image 
								feat_I_fc = model_imagenet_fc(V)
								video_feats_imagenet_fc[s_places_imagenet[i]:e_places_imagenet[i],:,:,:] = feat_I_fc.data.cpu().numpy()
								#print('{} {}'.format(feat_P.shape, feat_I.shape))
							#print('{} {}'.format(video_feats_places, video_feats_imagenet))

							#print('Extracting SoundNet Audio Features')
							#temp_A = load_one_audio(audio_filename) # Using scipy resample
							temp_A, duration_audio = load_audio_librosa(audio_filename) # Using librosa
							temp_A = temp_A.unsqueeze(0) # batch-size 1
							feat_conv7, feat_pool5 = model_sound(temp_A)
							feat_conv7, feat_pool5 = feat_conv7.data.cpu().numpy(), feat_pool5.data.cpu().numpy()
							#print('{} {}'.format(feat_conv7.shape, feat_pool5.shape))


							#print('Saving Features')
							'''sio.savemat('/home/vsharma/Desktop/dump/_s.mat',{'feat_conv7':feat_conv7, 
								'feat_pool5':feat_pool5, 'video_feats_resnet':video_feats_resnet,  
								'video_feats_i3d':video_feats_i3d, 
								'video_feats_imagenet':video_feats_imagenet,
								'video_feats_places':video_feats_places,
								'duration_audio':duration_audio,'num_frames':num_frames,
								'num_places_imagenet_frames':num_places_imagenet_frames,'num_pairs_i3d':num_pairs_i3d, 'num_pairs_resnet':num_pairs_resnet})'''
							with h5py.File(target_file_path, 'w') as hf:
								hf.create_dataset('audio_pool5', data=feat_pool5, dtype='float32')
								hf.create_dataset('audio_conv7', data=feat_conv7, dtype='float32')
								hf.create_dataset('video_resnet', data=video_feats_resnet, dtype='float32')
								hf.create_dataset('video_i3d', data=video_feats_i3d, dtype='float32')
								hf.create_dataset('imagenet', data=video_feats_imagenet, dtype='float32')
								hf.create_dataset('places', data=video_feats_places, dtype='float32')
								hf.create_dataset('duration_audio', data=duration_audio, dtype='float32')
								hf.create_dataset('num_frames', data=num_frames, dtype='float32')
								hf.create_dataset('num_places_imagenet_frames', data=num_places_imagenet_frames, dtype='float32')
								hf.create_dataset('num_pairs_i3d', data=num_pairs_i3d, dtype='float32')
								hf.create_dataset('num_pairs_resnet', data=num_pairs_resnet, dtype='float32')

							sio.savemat(target_file_path_raw_img,{'raw_img':frames_buffer_raw_image,'imagenet_fc':video_feats_imagenet_fc})

							#hf = h5py.File('data.h5', 'r')
							#n1 = hf.get('dataset_1')
							#n1 = np.array(n1)
							del frames_buffer_i3d 
							del frames_buffer_resnet 
							del frames_buffer_places_imagenet

							# Raw Image 
							del frames_buffer_raw_image
							torch.cuda.empty_cache()



