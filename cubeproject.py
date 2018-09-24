

import sys
import numpy as np
from pydmx import Dmx
import pyaudio as pa

import pprint
import time

import threading
import multiprocessing
import queue
import asyncio


import cubeui #before the other local imports!
import effects
import structures

import itertools

from matplotlib import pyplot as plt






            

class CubeProject:
    def __init__(self):
        self.dmxmap = bytearray([0]*513)
        self.cubes = []
        self.dmx = []
        
        
    def start_serial(self):
        self.dmx = Dmx('/dev/tty.usbserial-A6WZD7OJ')
        #self.dmx = Dmx('/dev/tty.usbserial-A6X0G7BT')
        
    def appendcube(self,typ,baseaddr):
       # if len(self.cubes)==0:
       #     baseaddr = 1
       # else:
       #     baseaddr = self.cubes[-1].baseaddress + self.cubes[-1].num_chs
       # 
       # if baseaddr <= 512:
       
        self.cubes.append(structures.Fixture(baseaddr, typ))
            
            #print(baseaddr)
            
    def send_to_dmx(self,strip):
        #print(strip[0:3])
        self.dmxmap = bytearray([0]*513)
        #self.dmxmap = np.zeros((513,),dtype=int)
        for cube,color in zip(self.cubes,strip):
            r,g,b = color
            self.dmxmap[cube.trim_ch]=255
            self.dmxmap[cube.red_ch]=r
            self.dmxmap[cube.green_ch]=g
            self.dmxmap[cube.blue_ch]=b
        self.dmx.render(self.dmxmap)
        
def read_audio(sem_quit,audio_stream, q_audio,num_samples):
    
    #while not sem_quit.acquire(False):
    while not sem_quit.acquire(False):

        
        # Read all the input data. 
        samples = audio_stream.read(num_samples,exception_on_overflow = False) 
        # Convert input data to numbers
        samples = np.frombuffer(samples, dtype=np.int16).astype(np.float)
        #samples = np.fromstring(samples, dtype=np.int16).astype(np.float)
        samples_l = samples[::2]  
        samples_r = samples[1::2]
        q_audio.put([samples_l, samples_r])
        
        
def queue_gen(q):
    while True:
        value = q.get()
        #print(value)
        yield value
        
def queue_gen_nowait(q):
    while True:
        
        try:
            value = q.get_nowait()
        #print(value)
            yield value
        except:
            yield None

def gen_queue(q_l,gen_stream_l):
    
        
    while True:
        for q,stream in zip(q_l,gen_stream_l):
            value = stream.__next__()
            #print(value)
            q.put(value)
        
        
def generators_loop(q_audio,q_plot,q_dmx,q_ui,effect_struct):
    
        
        audio = queue_gen(q_audio)
        
        freqs = effects.fft(audio)
        beat_info = effects.gen_beat(audio, 8)
        corner_stream = effect_struct.generator(q_ui,freqs,beat_info)
        
        cs1,cs2 = itertools.tee(corner_stream,2)
        
        gen_queue([q_plot,q_dmx],[cs1,cs2])
        
        
            
def plotting(q_plot,effect_struct):
    
    ln = len(effect_struct.pattern)
    wd = max([bn for bn in effect_struct.bins]) 
    
    img = []
    for i in range(wd):
        col = []
        for j in range(ln):
            col.append([0.0,0.0,0.0])
        img.append(col)
        
    plt.figure()  
    im = plt.imshow(img, interpolation='none', 
                            origin='bottom', 
                            aspect='auto', # get rid of this to have equal aspect
                            vmin=0,
                            vmax=1,
                            #cmap='jet'
                            )
    
    plt.ion()
    plt.show()
    
    while True:
        
        effect_data = q_plot.get()
        
        #print("Plot")
        #pprint.pprint(effect_data)
        
        for i,cl in enumerate(effect_data):
            for j in range(len(cl)):
                
                img[j][i]=effect_data[i][j]
              
        im.set_data(img) 
        plt.draw()
        plt.pause(0.001)
        
    return
    
    
def dmx_loop(q_data,effect_struct,cube_project):
    
    cube_project.start_serial()
    
    while True:
        
        pattern = q_data.get()
        
        #strip = np.zeros((513,), dtype=int)
        strip = [(0,0,0)]*512
        for i,col in enumerate(pattern):
            for j,el in enumerate(col):
                strip[effect_struct.cubes[i][j]]=(int(el[0]*255),int(el[1]*255),int(el[2]*255))
                
        #print(strip[0:3])
                
        cube_project.send_to_dmx(strip)
                
        
        
                    



        
        
if __name__ == "__main__":
    
    multiprocessing.set_start_method('forkserver')

    q_ui = multiprocessing.Queue()
    q_plot = multiprocessing.Queue()
    q_dmx = multiprocessing.Queue()
    q_audio = multiprocessing.Queue()

    audio_stream = pa.PyAudio().open(format=pa.paInt16, \
                                    channels=1, \
                                    rate=44100, \
                                    input=True, \
                                    # Uncomment and set this using find_input_devices.py
                                    # if default input device is not correct
                                    #input_device_index=2, \
                                    frames_per_buffer=1024)
                                
    
                                    
    # Convert the audio data to numbers, num_samples at a time.
    
    
    #This represents the order of addressing and the type
    cp = CubeProject()
    
    # adding phisical cubes with unique phisical addresses
    
    # or when I append a cube I select the structure
    
    # Corner
    cp.appendcube("4ch",1)
    cp.appendcube("4ch",9)
    cp.appendcube("4ch",17)
    cp.appendcube("4ch",25)    
    
    cp.appendcube("8ch",225)
    cp.appendcube("8ch",233)
    cp.appendcube("8ch",241)
    
    cp.appendcube("8ch",385)
    cp.appendcube("8ch",393)
   # 
    cp.appendcube("8ch",481)
    
    
    fx_audio_corner = effects.Effect(structures.bins_audio_corner,
                           structures.pattern_audio_corner,
                           structures.cubes_corner, 
                           effects.gen_color_freq_pattern,
                           "corner")
        
    
    audio_sem_quit = threading.Semaphore()
    audio_sem_quit.acquire()
    audiothread = threading.Thread(target = read_audio, args = (audio_sem_quit,audio_stream,q_audio,2048))
    audiothread.start()
    
                    
    
    genproc = multiprocessing.Process(None,generators_loop,args=(q_audio,q_plot,q_dmx,q_ui,fx_audio_corner))
    genproc.daemon = True
    genproc.start()
    
    plot=multiprocessing.Process(None,plotting,args=(q_plot,fx_audio_corner))
    plot.daemon = True
    plot.start()
    
    dmx=multiprocessing.Process(None,dmx_loop,args=(q_dmx,fx_audio_corner,cp))
    dmx.daemon = True
    dmx.start()
    
    cubeui.uistart(q_ui,fx_audio_corner)
    audio_sem_quit.release()
    
    genproc.terminate()
    plot.terminate()
    dmx.terminate()
    
    
    
    
    
    
        
        