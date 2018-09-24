import serial, sys, time


DMXOPEN = bytes([126])
DMXCLOSE = bytes([231])
DMXINTENSITY=bytes([6])+bytes([1])+bytes([2])				
DMXINIT1= bytes([3])+bytes([2])+bytes([0])+bytes([0])+bytes([0])
DMXINIT2= bytes([10])+bytes([2])+bytes([0])+bytes([0])+bytes([0])

class Dmx:
    def __init__(self, serialPort):
        try:
            print(serialPort)
            self.serial=serial.Serial(serialPort, baudrate=57600) #115200
            print(self.serial)
        except:
            print("Error: could not open Serial Port")
            sys.exit(0)

        self.serial.write( DMXOPEN+DMXINIT1+DMXCLOSE)
        self.serial.write( DMXOPEN+DMXINIT2+DMXCLOSE)
        
        self.dmxData=bytearray([0]*513)   #128 plus "spacer".
        
    def setChannel(self, chan, intensity):
        if chan > 512 : chan = 512
        if chan < 0 : chan = 0
        if intensity > 255 : intensity = 255
        if intensity < 0 : intensity = 0
        self.dmxData[chan] = intensity
        #print(self.dmxData[0:24])
    
    def blackout(self):
        for i in range(1, 512, 1):
            self.dmxData[i] = 0
    
    def render(self,data):
        sdata=data
        #print(sdata[0:27])
        #print(self.serial)
        res = self.serial.write(DMXOPEN+DMXINTENSITY+sdata+DMXCLOSE)
        #self.serial.flush()
        time.sleep(0.01)
        #print(res)