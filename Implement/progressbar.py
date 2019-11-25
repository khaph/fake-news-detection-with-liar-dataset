from time import sleep
import sys

def display_bar(name, cur, sum):
    sys.stdout.write('\r')
    sys.stdout.write("%s [%-20s] %d%%" % (name, '='*int(cur/sum*20), cur/sum*100))
    sys.stdout.flush()