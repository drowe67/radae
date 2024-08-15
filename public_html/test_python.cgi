#!/usr/bin/env python3
import cgi
import cgitb
import sys
import os

cgitb.enable()

# upload file from POST

form = cgi.FieldStorage()
form_file = form['file']
data = sys.stdin.buffer.read()
with open('/tmp/tx.wav', 'wb') as f:
    while True:
        chunk = form_file.file.read(100000)
        if not chunk:
            break
        f.write(chunk)
        
#print('Content-Type: text/html\r\n\r\n', end='')
#print('<html><body>')
#print('get <a href=\"/tmp/systemd-private-2fcf85ad243b4da08d79d2e27e0375af-apache2.service-W12wP/tmp/tx.wav\">file</a>')
#print('</body></html>')

# Download processed file

print("Content-Type: application/octet-stream")
print("Content-Disposition: attachment; filename=tx.wav")
print()
sys.stdout.flush()

bstdout = open(sys.stdout.fileno(), 'wb', closefd=False)
file = open('/tmp/tx.wav','rb')
bstdout.write(file.read())
