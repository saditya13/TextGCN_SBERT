import os
# import sys
# if len(sys.argv) != 2:
# 	sys.exit("no window size")
# w = sys.argv[1]
#print("execution started")
#window_size = [5, 8, 11, 15, 18]
#for i in window_size:


# print("dataset olid ")
# os.system('python remove_words.py olid')
# os.system('python build_graph.py olid')
# os.system('python train.py olid')

# print("dataset 20ng ")
# os.system('python remove_words.py 20ng')
# os.system('python build_graph.py 20ng')
# os.system('python train.py 20ng')


# print("dataset ohsumed ")
# os.system('python remove_words.py ohsumed')
# os.system('python build_graph.py ohsumed')
# os.system('python train.py ohsumed')


# print("dataset wnut ")
# os.system('python remove_words.py wnut')
# os.system('python build_graph_sbert.py wnut')
# os.system('python train.py wnut')

# print("dataset md ")
# os.system('python remove_words.py md')
# os.system('python build_graph_sbert.py md cos 0.4')
# os.system('python train.py md')

print("dataset mr ")
os.system('python remove_words.py mr')
os.system('python build_graph_sbert.py mr cos 0.4')
os.system('python train.py mr')


# print("dataset 2018 ")
# os.system('python remove_words.py 2018')
# os.system('python build_graph_sbert.py 2018 cos 0.3')
# os.system('python train.py 2018')