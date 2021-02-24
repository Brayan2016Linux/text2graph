#!/usr/bin/python3


import subprocess as sb
import platform
import os, getpass

def is_found(name_directory):
    found = False
    if platform.system() == 'Linux':
        directory = os.path.expanduser('~') + '/.local/'
        for root, dirs, files in os.walk(directory):
            for dt in dirs:
                if dt == name_directory:
                    found = True
    elif platform.system() == 'Windows':
        directory = os.path.expandvars(r'C:\\Users\\$USERNAME\\')
        directory += 'appdata\\local\\'
        for root, dirs, files in os.walk(directory):
            for dt in dirs:
                if dt == name_directory:
                    found = False
    return found

def install_model(model):
    py_command = "python3"
    if is_found(model):
        print("Model %s was installed previously."%model)
    else:
        if platform.system() == 'Windows':
            py_command = "python"
        command = sb.Popen([py_command, "-m", "spacy", "download", model, "--user"])
        output = command.communicate()[0]
        print(output)



