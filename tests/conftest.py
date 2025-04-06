# conftest.py
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1081'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1081'
