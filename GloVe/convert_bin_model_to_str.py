#!/usr/bin/py
#import cPickle as pickle
import pickle
f = open("corpus.model","rb")
bin_data = f.read()
graph_data = pickle.loads(bin_data)
print(graph_data)
