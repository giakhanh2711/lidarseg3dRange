from zipfile import ZipFile
import os

with ZipFile('OpenPCSeg.zip', 'r') as f:
    f.extractall(os.getcwd())