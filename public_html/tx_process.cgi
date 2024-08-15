#!/usr/bin/env python3
import cgi
import cgitb
import sys
import os
import subprocess
import torch

cgitb.enable()

# when true, we don't return a file, but print text to the screen instead
debug=False

# upload file from POST

form = cgi.FieldStorage()
form_file = form['file']
data = sys.stdin.buffer.read()
with open('/tmp/input.wav', 'wb') as f:
    while True:
        chunk = form_file.file.read(100000)
        if not chunk:
            break
        f.write(chunk)
        
if debug:
    print("Content-type: text/html\n\n")
    print("start")

my_env = os.environ.copy()
my_env["CODEC2_DEV"] = "/home/david/codec2-dev"
os.chdir('../radae')
if debug:
    print(os.getcwd())
ota_test = subprocess.check_output(["./ota_test.sh","-x","/tmp/input.wav","--tx_path","/tmp/","-d",],env=my_env, encoding='utf-8').replace('\n','<br>')
if debug:
    print(ota_test)

if not debug:
    # Download processed file

    print("Content-Type: application/octet-stream")
    print("Content-Disposition: attachment; filename=tx.wav")
    print()
    sys.stdout.flush()

    bstdout = open(sys.stdout.fileno(), 'wb', closefd=False)
    file = open('/tmp/tx.wav','rb')
    bstdout.write(file.read())
