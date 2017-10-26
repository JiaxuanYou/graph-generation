# import cPickle
import pickle as pkl
import sys

# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('unicode')
print(sys.getdefaultencoding())
# a = torch.ones(2,3)
# b = torch.ones(3,2)
#
# a_batch = torch.ones(2,3,2,3)
# b_batch = torch.ones(3,2)
#
#
# # print(torch.matmul(a,b))
# print(torch.matmul(a_batch,b_batch))
# # print(a)

dataset = 'cora'


names = ['x', 'tx', 'allx', 'graph']
objects = []
for i in range(len(names)):
    load = pkl.load(open("dataset/ind.{}.{}".format(dataset, names[i]), 'rb'),encoding='latin1')
    print('loaded')
    objects.append(load)
    # print(load)

print(type(objects[0]))
#
# with open('dataset/ind.citeseer.graph', 'r') as f:
#     a = pickle.load(f)
#     print(a)
#
# NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'graph']
# OBJECTS = []
# for i in range(len(NAMES)):
#     with open('dataset/ind.{}.{}'.format(DATASET,NAMES[i]), 'rb') as f:
#         OBJECTS.append(cPickle.load(f))
# x, y, tx, ty, graph = tuple(OBJECTS)