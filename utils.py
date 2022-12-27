from re import sub
import time


def camel_case(s):
    s = sub(r"(_|-)+", " ", s).title().replace(" ",
                                               "").replace(",", "").replace("(", "").replace(")", "")
    return ''.join([s[0].lower(), s[1:]])


def timestamp_as_string():
    return str(int(time.time()))
