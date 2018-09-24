from tkinter import *

import matplotlib
matplotlib.use("TkAgg")


def uistart(q_ui,effect_struct):
    window=Tk()
    
    w = 100
    h = 50
    x = 650
    y = 28
    window.geometry('%dx%d+%d+%d' % (w, h, x, y))
    
    def callback():
        print("Toggle!")
        cmd = {"Command": "Columns","Data": 1.0}
        #effect_struct.queue.put(cmd)
        q_ui.put(cmd)

    b = Button(window, text="Toggle", command=callback)
    b.pack()

        
    window.mainloop()