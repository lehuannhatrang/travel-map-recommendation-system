import jwt
from cryptography.fernet import Fernet

from constant import FERNET_KEY
from constant import JWT_SECRET

import shlex
import subprocess
import sys


def make_error(message, status=500):
    headers = {"Content-Type": "application/json"}
    return {"message": message, "success": False}, status, headers


def make_response(data):
    headers = {"Content-Type": "application/json"}
    return data, 200, headers


def get_cipher():
    return Fernet(FERNET_KEY)


def gen_token(username):
    try:
        credential = {
            'sub': username,
        }
        return jwt.encode(
            credential,
            JWT_SECRET,
            algorithm='HS256'
        ).decode('utf-8')
    except Exception as e:
        return e


def verify_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithm='HS256')
    except Exception as e:
        return e


def run_command(command, env=None):
    """
    Runs command and returns stdout
    """
    if env is None:
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True)
    else:
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            env=env)
    output, stderr = [stream.decode(sys.getdefaultencoding(), 'ignore')
                      for stream in process.communicate()]

    # case java
    if ('java ' in command) & (('-cp' in command) | ('-jar' in command)):
        if 'Exception' in stderr:
            raise RuntimeError("Cannot execute {}.\n|Error code is: {}.\n|Output: {}.\n|Stderr: {}"
                               .format(command, process.returncode, output, stderr))
        else:
            return stderr

    if process.returncode != 0:
        raise RuntimeError("Cannot execute {}.\n|Error code is: {}.\n|Output: {}.\n|Stderr: {}"
                           .format(command, process.returncode, output, stderr))
   
    return output
