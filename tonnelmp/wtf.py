import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
import time

PASSWORD = "yowtfisthispieceofshitiiit" # xd

'''
this is equivalent to cryptojs freeman used in the backend
honestly i dont understand why did he add this cuz it basically encrypts the timestamp and thats it :D
'''

def evp(password, salt, key_len=32, iv_len=16):
    dtot = b''
    prev = b''
    while len(dtot) < (key_len + iv_len):
        prev = hashlib.md5(prev + password + salt).digest()
        dtot += prev
    return dtot[:key_len], dtot[key_len:key_len + iv_len]

def generate_wtf():
    timestamp = str(int(time.time()))

    salt = get_random_bytes(8)
    key, iv = evp(PASSWORD.encode('utf-8'), salt)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(timestamp.encode('utf-8'), AES.block_size))
    
    wtf = base64.b64encode(b'Salted__' + salt + ct).decode('utf-8')
    
    return timestamp, wtf
