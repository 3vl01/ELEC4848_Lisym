import tkinter as tk
from subprocess import Popen, PIPE
import sys

# Function to execute the external script
def run_script():
    global process
    global running

    if not running:
        # Start the external script
        process = Popen([sys.executable, 'realtime_clap.py'], stdout=PIPE, stderr=PIPE)
        running = True
        button.config(text='Pause')
    else:
        # Terminate the external script
        process.terminate()
        running = False
        button.config(text='Start')


# Function to resize the widgets
def resize(event):
    # Calculate the new font size based on the height of the window
    new_font_size = int(root.winfo_height() / 15)
    title_label.config(font=('Arial', new_font_size))


    # Calculate the new button size based on the width of the window
    new_button_width = int(root.winfo_width() / 100)
    new_button_height = int(root.winfo_height() / 200)
    button.config(width=new_button_width, height=new_button_height)


    # Resize the icon
    new_icon_size = int(root.winfo_height() / 100)
    icon = original_icon.subsample(new_icon_size, new_icon_size)
    icon_label.config(image=icon)
    icon_label.image = icon  # Keep a reference to the image


# Set up the GUI
root = tk.Tk()
root.title("Lisym -- We hear we care")
root.geometry('500x500')  # Set the size of the window
root.bind('<Configure>', resize)  # Bind the resize function to the Configure event


title_label = tk.Label(root, text='Lisym -- We hear we care', font=('Arial', 30))  # Replace 'Title' with your title
title_label.pack(pady=10)


# Load the icon
original_icon = tk.PhotoImage(file='icon.png')  # Replace 'icon.png' with your icon file path


# Create a label with the icon
icon_label = tk.Label(root, image=original_icon)
icon_label.pack(pady=10)


# Create a label with the title
title_label2 = tk.Label(root, text='Home safety system for elderly', font=('Arial', 15))  # Replace 'Title' with your title
title_label2.pack(pady=10)


# Create a start/pause button
button = tk.Button(root, text='Start', command=run_script, width=5, height=3)
button.pack(pady=20)


# Initialize the process variable and running state
process = None
running = False


# Run the GUI loop
root.mainloop()