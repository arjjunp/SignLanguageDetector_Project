WORKSPACE_PATH = 'D:/pro3/Tensorflow/workspace'
SCRIPTS_PATH = 'D:/pro3/Tensorflow/scripts'
APIMODEL_PATH = 'D:/pro3/Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
#labels Dictionary
labels = [
{'name':'Hello', 'id':1},
{'name':'Yes', 'id':2},
{'name':'No', 'id':3},
{'name': 'Love', 'id':4},
{'name': 'More', 'id':5},
{'name': 'Name', 'id':6},
{'name': 'Quiet', 'id':7},
{'name': 'Play', 'id':8},
{'name': 'Time', 'id':9},
{'name': 'Friends', 'id':10}
]
with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
for label in labels:
f.write('item { \n')
f.write('\tname:\'{}\'\n'.format(label['name']))
f.write('\tid:{}\n'.format(label['id']))
f.write('}\n')
!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l
{ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/train.record'}
!python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l
{ANNOTATION_PATH + '/label_map.pbtxt'} -o {ANNOTATION_PATH + '/test.record'}
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
!mkdir {'D:\pro3\Tensorflow\workspace\models\\'+CUSTOM_MODEL_NAME}
!copy
{PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pip
27
eline.config'} {MODEL_PATH+'/'+CUSTOM_MODEL_NAME}

