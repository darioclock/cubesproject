import sys

_FixturesTypes = {
    "8ch": {
        "num_chs" : 8,
        "trim_ch" : 0,
        "red_ch" : 4,
        "green_ch" : 5,
        "blue_ch" : 6
    },
    "4ch": {
        "num_chs" : 4,
        "trim_ch" : 0,
        "red_ch" : 1,
        "green_ch" : 2,
        "blue_ch" : 3
    }
}

class Fixture:
    def __init__(self,baseaddress,typ):
        self.baseaddress = baseaddress
        self.num_chs = _FixturesTypes[typ]["num_chs"]
        self.trim_ch = _FixturesTypes[typ]["trim_ch"]+self.baseaddress
        self.red_ch = _FixturesTypes[typ]["red_ch"]+self.baseaddress
        self.green_ch = _FixturesTypes[typ]["green_ch"]+self.baseaddress
        self.blue_ch = _FixturesTypes[typ]["blue_ch"]+self.baseaddress
        self.virtual_cubes = []
            
        
#needs to correspond to pattern       
cubes_corner = [
                    [3],
                    [2,6],
                    [1,5,8],
                    [0,4,7,9],
                    [1,5,8],
                    [2,6],
                    [3],
                 ]
                 
cubes_stage = [
                    [16],
                    [17],
                    [18,20],
                    [19,21,22],
                 ]

cubes_entrance = [
                    [23,25],
                    [24,26,27],
             ]

bins_audio_corner = [4,3,2,1]
pattern_audio_corner = [  [[3,0]],
                          [[2,0], [2,1]],
                          [[1,0], [1,1], [1,2]], 
                          [[0,0], [0,1], [0,2], [0,3]],
                          [[1,0], [1,1], [1,2]],
                          [[2,0], [2,1]],
                          [[3,0]],
                        ]
                        
bins_generic_address = [1,2,3,4,3,2,1]
pattern_generic_corner = [[[0,0]],
                          [[1,0], [1,1]],
                          [[2,0], [2,1], [2,2]], 
                          [[3,0], [3,1], [3,2], [3,3]],
                          [[4,0], [4,1], [4,2]],
                          [[5,0], [5,1]],
                          [[6,0]],
                        ]
                            

        
        