from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for, session, current_app, Flask
from app.models import _equalized_conv2d, _equalized_deconv2d, _equalized_linear, PixelwiseNorm, GenInitialBlock, GenGeneralConvBlock
from app.generator import Generator
import json
import os
from sklearn.preprocessing import LabelEncoder
import re
import gc
import os
import cv2
import copy
import time
import pickle
import random
import shutil
import urllib
import pathlib
import datetime
import operator
import warnings
import numpy as np

from PIL import Image
from scipy import linalg
from sklearn.metrics import *
from collections import Counter
from scipy.stats import truncnorm
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder

import torch
import torch as th
import torch.nn as nn
import torch.utils.data
import torchvision as tv
import torch.nn.functional as F
import torchvision.models as models

from torch.optim import Adam
from torch.nn import Parameter

from torchvision.datasets import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
from torchvision.utils import save_image
from torchvision.datasets.folder import *
from torch.nn.functional import interpolate
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Conv2d, BCEWithLogitsLoss, DataParallel, AvgPool2d, ModuleList, LeakyReLU, ConvTranspose2d, Embedding

import base64
import io


views = Blueprint('views', __name__)

def load_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'-> Loading weights from {filename}')
    print('Loading weights')
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    return model

class MarvelGenerator():
    def __init__(self):
        self.gen = Generator(depth=6, latent_size=256)
        GEN_PATH = os.path.join(current_app.root_path, 'marvel_gens', 'genNoGpu.pt')
        self.gen = load_model_weights(self.gen, GEN_PATH)
        self.gen_shadow = copy.deepcopy(self.gen)
        SHADOW_GEN_PATH = os.path.join(current_app.root_path, 'marvel_gens', 'gen_shadowNoGpu.pt')
        self.gen_shadow = load_model_weights(self.gen_shadow, SHADOW_GEN_PATH)

    def generate(self, depth=None, alpha=1, noise=None, n=1, n_plot=0):
        print('Checkpoint 3')
        if depth is None:
            depth = self.depth - 1
        if noise is None:
            noise = th.randn(n, 256).cpu()
#             z = self.truncated_normal(size=(n, self.latent_size - self.num_classes))
#             noise = torch.from_numpy(z).float().cuda()
        
        gan_input = noise
        print('Checkpoint 6')
        if True:
            generated_images = self.gen_shadow(gan_input, depth, alpha).detach().cpu()
        else:
            generated_images = self.gen(gan_input, depth, alpha).detach().cpu()
        print('Checkpoint 7')
#         self.scale(generated_images)
        generated_images.add_(1).div_(2)
        return generated_images

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

class MathGenerator():
    def __init__(self, ANNOTATION_PATH):
        self.classes = [dirname[5:] for dirname in os.listdir(ANNOTATION_PATH)]
        
        le = LabelEncoder().fit(self.classes)
        self.classes = le.inverse_transform(range(len(self.classes)))
        self.num_classes = len(self.classes)
        num_classes = self.num_classes
        self.classesbetternames = []
        for mathclass in self.classes:
            newclass = mathclass.lower()
            newclass = newclass.replace('_', '')
            newclass = newclass.replace(' ', '')
            self.classesbetternames.append(newclass)
        
        


        self.gen = Generator(depth=5, latent_size=256)
        GEN_PATH = os.path.join(current_app.root_path, 'math_gens', 'genNoGpu.pt')
        self.gen = load_model_weights(self.gen, GEN_PATH)
        self.gen_shadow = copy.deepcopy(self.gen)
        SHADOW_GEN_PATH = os.path.join(current_app.root_path, 'math_gens', 'gen_shadowNoGpu.pt')
        self.gen_shadow = load_model_weights(self.gen_shadow, SHADOW_GEN_PATH)
        


    def one_hot_encode(self, labels):
        if not hasattr(self, "label_oh_encoder"):
            self.label_oh_encoder = th.nn.Embedding(self.num_classes, self.num_classes)
            self.label_oh_encoder.weight.data = th.eye(self.num_classes)
        return self.label_oh_encoder(labels.view(-1))

    def generate(self, depth=None, alpha=1, noise=None, race=None, n=1, n_plot=0):
        if depth is None:
            depth = self.depth - 1
        if noise is None:
            noise = th.randn(n, 256 - self.num_classes).cpu()
#             z = self.truncated_normal(size=(n, self.latent_size - self.num_classes))
#             noise = torch.from_numpy(z).float().cuda()
        races = None
        index = self.classesbetternames.index(race)
        indexes = []
        for _ in range(n):
            indexes.append(index)
        indexes = np.array(indexes)
        if races is None:
            races = torch.from_numpy(indexes).long()
        
        label_information = self.one_hot_encode(races).cpu()
        gan_input = th.cat((label_information, noise), dim=-1)
        
        if True:
            generated_images = self.gen_shadow(gan_input, depth, alpha).detach().cpu()
        else:
            generated_images = self.gen(gan_input, depth, alpha).detach().cpu()

#         self.scale(generated_images)
        generated_images.add_(1).div_(2)
        return generated_images

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)


class DogGenerator():
    def __init__(self, ANNOTATION_PATH):
        self.classes = [dirname[10:] for dirname in os.listdir(ANNOTATION_PATH)]
        
        le = LabelEncoder().fit(self.classes)
        self.classes = le.inverse_transform(range(len(self.classes)))
        self.num_classes = len(self.classes)
        num_classes = self.num_classes
        self.classesbetternames = []
        for speciesclass in self.classes:
            newclass = speciesclass.lower()
            newclass = newclass.replace('_', '')
            newclass = newclass.replace('-', '')
            newclass = newclass.replace(' ', '')
            self.classesbetternames.append(newclass)


        self.gen = Generator(depth=5, latent_size=256)
        GEN_PATH = os.path.join(current_app.root_path, 'dog_gens', 'genNoGpu.pt')
        self.gen = load_model_weights(self.gen, GEN_PATH)
        self.gen_shadow = copy.deepcopy(self.gen)
        SHADOW_GEN_PATH = os.path.join(current_app.root_path, 'dog_gens', 'gen_shadowNoGpu.pt')
        self.gen_shadow = load_model_weights(self.gen_shadow, SHADOW_GEN_PATH)
        


    def one_hot_encode(self, labels):
        if not hasattr(self, "label_oh_encoder"):
            self.label_oh_encoder = th.nn.Embedding(self.num_classes, self.num_classes)
            self.label_oh_encoder.weight.data = th.eye(self.num_classes)
        return self.label_oh_encoder(labels.view(-1))

    def generate(self, depth=None, alpha=1, noise=None, race=None, n=1, n_plot=0):
        print('Checkpoint 3')
        if depth is None:
            depth = self.depth - 1
        if noise is None:
            noise = th.randn(n, 256 - self.num_classes).cpu()
#             z = self.truncated_normal(size=(n, self.latent_size - self.num_classes))
#             noise = torch.from_numpy(z).float().cuda()
        races = None
        index = self.classesbetternames.index(race)
        indexes = []
        print('Checkpoint 4')
        for _ in range(n):
            indexes.append(index)
        indexes = np.array(indexes)
        print('Checkpoint 5')
        if races is None:
            races = torch.from_numpy(indexes).long()
        
        label_information = self.one_hot_encode(races).cpu()
        gan_input = th.cat((label_information, noise), dim=-1)
        print('Checkpoint 6')
        if True:
            generated_images = self.gen_shadow(gan_input, depth, alpha).detach().cpu()
        else:
            generated_images = self.gen(gan_input, depth, alpha).detach().cpu()
        print('Checkpoint 7')
#         self.scale(generated_images)
        generated_images.add_(1).div_(2)
        return generated_images

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

@views.route('/')
def home():
    return redirect(url_for('views.about'))

@views.route('/dogs', methods=['GET', 'POST'])
def dogs():
    ANNOTATION_PATH = os.path.join(current_app.root_path, 'Annotation')
    classes = [dirname[10:] for dirname in os.listdir(ANNOTATION_PATH)]
    improvedclasses = []
    for currentclass in classes:
        newclass = currentclass.lower()
        newclass = newclass.replace('_', ' ')
        newclass = newclass.replace('-', ' ')
        improvedclasses.append(newclass)
    improvedclasses.sort()
    if request.method == 'POST':
        print('Checkpoint 1')
        dog_gen = DogGenerator(ANNOTATION_PATH)
        species = request.form.get('species')
        species = species.lower()
        species = species.replace('_', '')
        species = species.replace('-', '')
        species = species.replace(' ', '')
        print('Checkpoint 2')
        if species in dog_gen.classesbetternames:
            generated_images = dog_gen.generate(depth=4, alpha=1, noise=None, race=species, n=64, n_plot=10)
            images = generated_images.clone().numpy().transpose(0, 2, 3, 1)      
            urls = []
            scale_size = 2
            for dogimg in images:
                img = (dogimg*255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((64*scale_size,64*scale_size))
                buff = io.BytesIO()
                pil_img.save(buff, format="JPEG")
                new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
                urls.append('data:image/png;base64,%s' % new_image_string)
            return render_template("dog.html", urls = (urls), validspecies = improvedclasses)
        else:
            flash('Invalid species!', category='error')

    return render_template("dog.html", urls = None, validspecies = improvedclasses)

@views.route('/math-symbols', methods=['GET', 'POST'])
def math_symbols():
    ANNOTATION_PATH = os.path.join(current_app.root_path, 'AnnotationMath')
    classes = [dirname[5:] for dirname in os.listdir(ANNOTATION_PATH)]
    improvedclasses = []
    for currentclass in classes:
        if ('cap' in currentclass):
            currentclass =  "capital " + currentclass.replace('cap', '')
        newclass = currentclass.lower()
        if newclass == "decimal":
            continue
        if newclass == "prime":
            continue
        newclass = newclass.lower()
        newclass = newclass.replace('_', ' ')
        if ('infty' == newclass):
            newclass =  'infinity'
        if ('int' == newclass):
            newclass =  'integral'
        if ('geq' == newclass):
            newclass =  '>='
        if ('leq' == newclass):
            newclass =  '<='
        if ('lt' == newclass):
            newclass =  '<'
        if ('gt' == newclass):
            newclass =  '>'
        if ('div' == newclass):
            newclass =  'division'
        if ('neq' == newclass):
            newclass =  '!='
        if ('pm' == newclass):
            newclass =  '+/-'
        improvedclasses.append(newclass)
    if request.method == 'POST':
        math_gen = MathGenerator(ANNOTATION_PATH)
        species = request.form.get('symbol')
        species = species.lower()
        if ('capital' in species):
            species =  species.replace('capital', '') + "cap"
        species = species.replace('_', '')
        species = species.replace(' ', '')
        if ('infinity' == species):
            species =  'infty'
        if ('integral' == species):
            species =  'int'
        if ('>=' == species):
            species =  'geq'
        if ('<=' == species):
            species =  'leq'
        if ('<' == species):
            species =  'lt'
        if ('>' == species):
            species =  'gt'
        if ('%' == species):
            species =  'div'
        if ('division' in species):
            species =  'div'
        if ('!=' == species):
            species =  'neq'
        if ('+-' == species):
            species =  'pm'
        if ('+/-' == species):
            species =  'pm'
        if ('plusminus' == species):
            species =  'pm'
        if ('summation' == species):
            species =  'sum'

        if species in math_gen.classesbetternames:
            urls = []
            scale_size = 1
            for _ in range(2):
                generated_images = math_gen.generate(depth=4, alpha=1, noise=None, race=species, n=64, n_plot=10)
                images = generated_images.clone().numpy().transpose(0, 2, 3, 1)      
                for mathimg in images:
                    img = (mathimg*255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((64*scale_size,64*scale_size))
                    buff = io.BytesIO()
                    pil_img.save(buff, format="JPEG")
                    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
                    urls.append('data:image/png;base64,%s' % new_image_string)
            return render_template("math.html", urls = (urls), validsymbols = improvedclasses)
        else:
            flash('Invalid symbol!', category='error')

    return render_template("math.html", urls = None, validsymbols = improvedclasses)

@views.route('/marvel-inspired-superheroes', methods=['GET', 'POST'])
def marvel_inspired_superheroes():
    if request.method == 'POST':
        marvel_gen = MarvelGenerator()
        urls = []
        scale_size = 1
        for _ in range(4):
            generated_images = marvel_gen.generate(depth=5, alpha=1, noise=None, n=16, n_plot=10)
            images = generated_images.clone().numpy().transpose(0, 2, 3, 1)      
            for marvelimg in images:
                img = (marvelimg*255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((128*scale_size,128*scale_size))
                buff = io.BytesIO()
                pil_img.save(buff, format="JPEG")
                new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
                urls.append('data:image/png;base64,%s' % new_image_string)
        return render_template("marvel.html", urls = (urls))

    return render_template("marvel.html", urls = None)


@views.route('/how-to-use')
def how_to_use():
    MATH_ANNOTATION_PATH = os.path.join(current_app.root_path, 'AnnotationMath')
    mathclassesoriginal = [dirname[5:] for dirname in os.listdir(MATH_ANNOTATION_PATH)]
    mathclasses = []
    for mathclass in mathclassesoriginal:
        if ('cap' in mathclass):
            mathclass =  "capital " + mathclass.replace('cap', '')
        newclass = mathclass.lower()
        if newclass == "decimal":
            continue
        if newclass == "prime":
            continue
        if ('infty' == newclass):
            newclass =  'infinity'
        if ('int' == newclass):
            newclass =  'integral'
        if ('geq' == newclass):
            newclass =  '>='
        if ('leq' == newclass):
            newclass =  '<='
        if ('lt' == newclass):
            newclass =  '<'
        if ('gt' == newclass):
            newclass =  '>'
        if ('div' == newclass):
            newclass =  'division'
        if ('neq' == newclass):
            newclass =  '!='
        if ('pm' == newclass):
            newclass =  '+/-'
        newclass = newclass.lower()
        newclass = newclass.replace('_', '')
        mathclasses.append(newclass)

    DOG_ANNOTATION_PATH = os.path.join(current_app.root_path, 'Annotation')
    dogclassesoriginal = [dirname[10:] for dirname in os.listdir(DOG_ANNOTATION_PATH)]
    dogclasses = []
    for dogclass in dogclassesoriginal:
        newclass = dogclass.lower()
        newclass = newclass.replace('_', ' ')
        newclass = newclass.replace('-', ' ')
        dogclasses.append(newclass)
    dogclasses.sort()
    return render_template("howtouse.html", validdogs = dogclasses, validsymbols = mathclasses)


@views.route('/about')
def about():
    return render_template('about.html')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'Machine_Learning'
app.secret_key = 'Machine_Learning'
app.register_blueprint(views, url_prefix='/')