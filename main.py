import logging
import sys 
from tkinter import Tk, messagebox
import eel
import base64

sys.path.append('./emotion-detection')
sys.path.append('./fall-detection')

def show_error(title, msg):
    root = Tk() # create a tkinter window
    root.withdraw()  # hide main window
    messagebox.showerror(title, msg) # show error message
    root.destroy() # destroy the tkinter window
        
if __name__ == '__main__':
    try:
        eel.init('client') # path to project folder 
        eel.start('index.html') # start the web app with the main file index.html
    except Exception as e: 
        err_msg = 'Could not launch a local server' # error message
        logging.error('{}\n{}'.format(err_msg, e.args))
        show_error(title='Failed to initialise server',
                   msg=err_msg) #use tkinter to show error message
        sys.exit()