# import random 
# import os
# user_key = os.urandom(32)
# print(user_key)

import os
import binascii
user_key = os.urandom(32)
print(binascii.hexlify(user_key).decode())
# Output: 944559415bf7529efc573347f3533150e7e4bc9fdf40d6b3d9877f079c34cc7d
