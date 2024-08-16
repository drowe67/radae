#!/usr/bin/env python3
import cgi
import cgitb
import sys
import os
import subprocess
from pathlib import Path

def choke_on_error(error_str):
    print("Content-type: text/html\n\n")
    print(error_str)
    quit()

def return_file(filename):
    print(f"returning {filename}", file=sys.stderr)
    print(f"Content-Type: application/octet-stream")
    print(f"Content-Disposition: attachment; filename={os.path.basename(filename)}")
    print()
    sys.stdout.flush()
    bstdout = open(sys.stdout.fileno(), 'wb', closefd=False)
    file = open(filename,'rb')
    bstdout.write(file.read())

cgitb.enable()
        
# extract the form fields and pointer to POSTed file
form = cgi.FieldStorage()
form_file = form['file']
form_filename = os.path.basename(form_file.filename)
form_processing = form.getvalue('processing')

if len(form_filename) == 0:
    choke_on_error("No file supplied")

# anything on stderr will appear on apache error log which is very handy for tracing execution
# tail -f /var/log/apache2/error.log
print("\n----------------------------------------------------", file=sys.stderr)
print(f"{form_processing} {form_filename}", file=sys.stderr)        
print("----------------------------------------------------\n", file=sys.stderr)

# upload file from POST
data = sys.stdin.buffer.read()
with open(f"/tmp/{form_filename}", 'wb') as f:
    while True:
        chunk = form_file.file.read(100000)
        if not chunk:
            break
        f.write(chunk)

my_env = os.environ.copy()
my_env["CODEC2_DEV"] = "/home/david/codec2-dev"
os.chdir('../radae')
if form_processing == "tx":
    ota_test = subprocess.check_output(["./ota_test.sh","-x",f"/tmp/{form_filename}","--tx_path","/tmp/","-d",],env=my_env, encoding='utf-8').replace('\n','<br>')
    #print(ota_test, file=sys.stderr)
    return_file('/tmp/tx.wav')
else:
    ota_test = subprocess.check_output(["./ota_test.sh","-r",f"/tmp/{form_filename}","-d",],env=my_env, encoding='utf-8').replace('\n','<br>')
    # zip up files for return
    filename = Path(form_filename).stem
    #print("Content-type: text/html\n\n")
    zip_test = subprocess.check_output(["zip","-j",f"/tmp/{filename}.zip",f"/tmp/{filename}_ssb.wav", f"/tmp/{filename}_radae.wav",
                                        f"/tmp/{filename}_spec.jpg", f"/tmp/{filename}_report.txt"], encoding='utf-8').replace('\n','<br>')
    print(zip_test, file=sys.stderr)
    print("finished!", file=sys.stderr)
    return_file(f"/tmp/{filename}.zip")
