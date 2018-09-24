import numpy as np

import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
import pprint
import time
import copy

import threading
import asyncio

import multiprocessing

import itertools



def square(list):
    return [i ** 2 for i in list]
    
def fft(audio_stream):
    def real_fft(im):
        im = np.abs(np.fft.fft(im))
        re = im[0:len(im)//2]
        re[1:] += im[len(im)//2 + 1:][::-1]
        return re
    for l, r in audio_stream:
        yield real_fft(r) + real_fft(l)
        
def gen_beat(audio_stream,limit):
    samples = []
    beat_count = 0
    Tbeat = 1
    Tbeats = []
    Tbeats_lim = 5
    while True:
        l,r = audio_stream.__next__()
        sample = l+r
        samples.append(sample)
        
        if len(samples)==1:
            samples =samples*limit
        if len(samples)>limit:
            samples.pop(0)
            
        en_samples = np.mean(np.mean(square(samples[:-1]), axis = 1))
        sd_samples =  np.mean(np.std(square(samples[:-1]), axis = 1))
        en_sample = np.mean(square(sample))
        beat = (en_sample > (en_samples + sd_samples))
        if beat_count > 0:
            beat_count -= 1
            beat = 0
        if beat:
            beat_count = limit-1
        
        t = time.time()

        if beat:
            Tbeats.append(t)
            if len(Tbeats)>Tbeats_lim:
                Tbeats.pop(0)
            if len(Tbeats)>1:
                #print(Tbeats, np.diff(Tbeats))
                Tbeat = np.mean(np.diff(Tbeats))

                
        yield {"Beat Period" : Tbeat, "Beat": bool(beat)}
        
        
def gen_norm_struct(falloff, struct_stream):
    peak = 0
    
    def flatten(seq):
        flat = []
        for col in seq:
            col_out = []
            if isinstance(col, list):
                for el in col:
                    col_out.append(el)
            flat.extend(col_out)
        return flat
        
    
    while True:
        struct = struct_stream.__next__()
        #print("\n")
        #print("struct")
        #pprint.pprint(struct)
        struct_max = max(flatten(struct))
        #print("max",struct_max)
        #print("peak",peak)
        
        struct_norm = []
        for i,col in enumerate(struct):
            col_norm = []
            for el in col:
                if struct_max>peak:
                    peak = struct_max
                else:
                    peak *= falloff
                    peak += el * (1- falloff)
                if peak == 0:
                    col_norm.append(el)
                else:
                    col_norm.append(el/peak)
            struct_norm.append(col_norm)
                    
                    
        #print("Norm")
        #pprint.pprint(struct_norm)
                    
        yield struct_norm
                
        
def gen_freq_bins(effect_struct,fft_stream):
    peaks = []
    peaks = np.zeros(len(effect_struct.bins))
    
    falloff = 0.98
    num_bins = len(effect_struct.bins)
    
    cap_fft_bins=[]
    cap_fft_bins = np.zeros(len(effect_struct.bins))
    
    
    while True:
        fft = fft_stream.__next__()
        num_freqs = len(fft)

        #fft = fft[num_freqs//2:3*num_freqs//4]
        fft = fft[:3*num_freqs//4]
        fft_bins = (np.histogram(fft, num_bins, weights=fft)[0] /
                     (np.histogram(fft, num_bins)[0]+1))
        fft_bins = np.nan_to_num(fft_bins)
        
        #print(fft_bins)
        
        for i,peak,fft_bin in zip(range(len(effect_struct.bins)),peaks,fft_bins):
            if fft_bin > peak:
                peaks[i]=fft_bin
            else:
                peaks[i] *= falloff
                peaks[i] += fft_bin * (1-falloff)
            if peaks[i]==0:
                cap_fft_bins[i] = fft_bin
            else:
                cap_fft_bins[i] = fft_bin/peaks[i]
                
            yield cap_fft_bins

def gen_freqbar_struct(effect_struct, cap_fft_bins):

    while True:
        cap_fft_bin = cap_fft_bins.__next__()
        struct = []
        for num_amp_bin,cap_fft_bin in zip(effect_struct.bins, cap_fft_bins):
            amp_bins = np.linspace(0,1,num_amp_bin+2)[1:]
            struct.append([cap_fft_bin>b for b in amp_bins])
            
        yield struct
            
def gen_struct_to_pattern(effect_struct, structs):
    
    while True:
        struct = structs.__next__()
        pattern_out = []
        for col in effect_struct.pattern:
            column_out = []
            for row in col:
                c,r = row
                column_out.append(list(struct[c][r]))
            pattern_out.append(column_out)
        yield pattern_out
        
def gen_color_struct(effect_struct,beat_info_stream):
    leng = len(effect_struct.bins)
    widt = max([bn for bn in effect_struct.bins])
    x = np.linspace(0, 1, leng)
    y = np.linspace(0, 1, widt)
    X, Y = np.meshgrid(x, y)
    
    
    t0=time.time()
    
    while True:
        beat_info = beat_info_stream.__next__()
        
        t=time.time()
        
        #data = (np.sin(-(t-t0)/beat_info["Beat Period"]*10+(X ** 2 + Y **2)/8)+1)/2
        data = (np.sin(-(t-t0)/beat_info["Beat Period"]+(5*X ** 2 + Y **2)/8)+1)/2
        #print(data)
        struct = []
        for num_amp_bin,data_col in zip(effect_struct.bins, data):
            struct.append(data_col[:num_amp_bin]) 
        
        yield struct
    
        
def gen_color_freq_struct(q,effect_struct,fft_stream,beat_info_stream):
    
    freq_bins = gen_freq_bins(effect_struct,fft_stream)
    color_structs = gen_color_struct(effect_struct,beat_info_stream)
    
    #cmap = cm.hsv
    #cmap = cm.gray
    #cmap = cm.terrain
    #cmap = cm.jet
    
    cmaps = [cm.magma,cm.plasma,cm.rainbow,cm.hsv,cm.terrain,cm.jet,]
    cmap = cmaps[0]
    
    norm = Normalize(vmin=0, vmax=1)
    
    blend = 0.7
    
    mode = "2"
    
    t0 = time.time()
    cmaps_count = 0
    dt0 = 0
    
    while True:
        t = time.time()
        dt = t-t0
        freq_bin = freq_bins.__next__() # range 0-1
        color_struct = color_structs.__next__() #range -1 to 1
        beat_info = beat_info_stream.__next__()
        

            
        
        try:
            cmd = q.get_nowait()
            print(cmd)
            if cmd["Command"] == "Flash":
                print("Flash command received!")
                flash = True
            else:
                q.put(cmd)
        except:
            pass
            
        #print(int(dt)%10)
            
        if (int(dt)%10)==0:
            if int(dt) != int(dt0):
                cmaps_count += 1
                cmap = cmaps[cmaps_count % len(cmaps)]
                #print("Change")
                dt0 = dt
                
        
        #print("Color_struct")
        #pprint.pprint(color_struct)
        struct = []
        for i,bn in enumerate(effect_struct.bins):
            column = []
            amp_bins = np.linspace(0,1,bn+2)[1:]
            #print(amp_bins)
            for j in range(bn):
                
                #if i==-j+3:
                #    freq_sel = freq_bin[0]
                #if i==-j+2:
                #    freq_sel = freq_bin[1]
                #if i==-j+1:
                #    freq_sel = freq_bin[2] #*freq_sel
                #if (i==0) & (j==0):
                #    freq_sel = freq_bin[3]
                
                if mode == "1":
                    if i==-j+3:
                        freq_sel = freq_bin[0]
                    if i==-j+2:
                        freq_sel = freq_bin[1]
                    if i==-j+1:
                        freq_sel = freq_bin[2] #*freq_sel
                    if (i==0) & (j==0):
                        freq_sel = freq_bin[3]
                        
                    data = np.array((color_struct[i][j]*(1-blend))+(blend)*freq_sel)
                    data = cmap(norm(data))[:3]
                    
                    if beat_info["Beat"]:
                        data = [1.0,1.0,1.0]
                
                elif mode == "2":
                    
                    
                    if i==+3:
                        freq_sel = freq_bin[0]
                    if i==+2:
                        freq_sel = freq_bin[1]
                    if i==+1:
                        freq_sel = freq_bin[2] #*freq_sel
                    if i==0:
                        freq_sel = freq_bin[3]
                     
                    #print(j)  
                    mag = 1-amp_bins[j]
                        
                    data = np.array((color_struct[i][j]*(1-blend)/mag)+(blend)*freq_sel*mag)
                    #data = np.array(np.array(cmap(norm(data)*freq_sel*mag/2)[:3]))
                    data = cmap(norm(data))[:3]
                    
                    if beat_info["Beat"]:
                        data = [1.0,1.0,1.0]
                #data = np.array(color_struct[i][j]*(1-blend)+(blend)*freq_sel)
                    #column.append(cmap(norm( data ))[:3])
                column.append(data)
                #print(data)
            struct.append(column)
            
        
            
        #print("CMAP")
        #pprint.pprint(struct)
        yield struct
        
        #TODO: every additive goes with a blend a,1-a. Every mult needs to be renormalized
        
def gen_flash(q,effect_struct,patterns):
    
    flash_falloff = 0.8
    flash = 0
    
    while True:
        
        
        try:
            cmd = q.get_nowait()
            print(cmd)
            if cmd["Command"] == "Flash":
                print("Flash command received!")
                flash = 1.0
            else:
                q.put(cmd)
        except:
            pass
        
        
        pattern = patterns.__next__()
        
        pattern_out=[]
        for i,col in enumerate(pattern):
            column = []
            for j,el in enumerate(col):
                column.append(np.array((1.0,1.0,1.0))*flash+(1-flash)*np.array(pattern[i][j]))
            pattern_out.append(column)
        
        flash *= flash_falloff
        
        #print("Flash")
        #pprint.pprint(pattern_out)
        
        yield pattern_out
        
def gen_cols(q,effect_struct,patterns):
    
    effects_array = []
    
    while True:
        
        pattern = patterns.__next__()
        nbins = len(pattern)
        
        try:
            cmd = q.get_nowait()
            print(cmd)
            if cmd["Command"] == "Columns":
                print("Column command received!")
                tstart = time.time()
                speed = cmd["Data"] # how many cycles a step lasts
               # effect_array = []
               # 
               # for i in range(nbins//2+1):
               #     effect_step = set()
               #     for j in range(len(pattern[i])):
               #         effect_step.add((i,j))
               #     for j in range(len(pattern[-i-1])):
               #         effect_step.add((nbins-i-1,j))
               #     for d in range(int(speed)):
               #         effect_array.append(effect_step)
                    
                
                effect_array = [{(0,0),(0,1),(0,2),(0,3)},{(1,0),(1,1),(1, 2)},{(2,0),(2,1)},{(3,0)}]
                effect_array.extend(effect_array[::-1])
                
                #effect_array = [{(0,0)},{(1,0)},{(2,0)},{(3,0)},{(4,0)},{(5,0)},{(6,0)},{(1,1)},{(2,1)},{(3,1)},{(4,1)},{(5,1)},{(2,2)},{(3,2)},{(4,2)},{(3,3)}]
                #effect_array.extend(effect_array[::-1])
                
                lefs = len(effects_array)
                lef = len(effect_array)
                effects_out = []
                i=0
                for i in range(min(lefs,lef)):
                    effects_out.append(set())
                    effects_out[i] |= effects_array[i]
                    effects_out[i] |= effect_array[i]
                    
                if lefs > lef:
                    effects_out.extend(effects_array[i:])
                else:
                    effects_out.extend(effect_array[i:])
                effects_array = effects_out
            else:
                q.put()
                
        except:
            pass
        
        
        if effects_array:
            for el in effects_array[0]:
                pattern[el[0]][el[1]] = np.array((1.0,1.0,1.0))
                
            effects_array = effects_array[1:]
        
        yield pattern
        
def gen_cmap(effect_struct,q_ui,struct_stream):
    
    #cmap = cm.jet
    cmap = cm.hsv
    #cmap = cm.gray
    
    norm = Normalize(vmin=0, vmax=1)
    
    while True:
        struct = struct_stream.__next__()
        
        #print("Patt")
        #pprint.pprint(struct)
        
        struct_out = []
        for i,col in enumerate(struct):
            col_out = []
            for el in col:
                col_out.append(cmap(norm(el))[:3])
            struct_out.append(col_out)
            
        yield struct_out
        
        
def gen_color_freq_pattern(effect_struct,q_ui,fft_stream,beat_info_stream):
    
    #I need to call next on _ other wise it won't advance
    
    amps = gen_color_freq_struct(q_ui,effect_struct,fft_stream,beat_info_stream) # not normalized
    #norms =  gen_norm_struct(0.7,amps) #can't normalize something that has already become a color..
    #structs = gen_cmap(effect_struct,q_ui,norms)

    patterns3 = gen_cols(q_ui,effect_struct,amps)
    
    patterns4 = gen_struct_to_pattern(effect_struct, patterns3)
    
        
    return patterns4
        
        
class Effect:
    def __init__(self,bins,pattern,cubes,generator,name):
        self.bins = bins
        self.pattern = pattern
        self.cubes = cubes
        self.generator_func = generator
        self.on = True
        self.name = name
        
        
    def set_state(self,state):
        if state:
            self.on = True
        else:
            self.on = False
            
    def get_state(self):
        return self.on
            
    def generator(self,*args):
        return self.generator_func(self,*args)
            

        

        
        
        