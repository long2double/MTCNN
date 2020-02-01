# MTCNN-tf
# pnet
python gen_pnet_data.py
python gen_landmark.py --name lfw --input_size 12
python gen_landmark.py --name wider --input_size 12
python gen_lfw_wider_imglist.py --input_size 12
python gen_pnet_imglist.py --input_size 12 --name wider
python gen_tfrecords.py --input_size 12 --name wider
python train.py —-input_size 12

# rnet
python gen_hard_example.py —-input_size 12
python gen_landmark.py  --name lfw --input_size 24
python gen_landmark.py  --name wider --input_size 24
python gen_lfw_wider_imglist.py --input_size 24
python gen_tfrecords.py --input_size 24 --name wider
python train.py —-input_size 24

# onet
python gen_hard_example.py —-input_size 24
python gen_landmark.py --name lfw --input_size 48
python gen_landmark.py --name wider --input_size 48
python gen_lfw_wider_imglist.py --input_size 48
python gen_tfrecords.py --input_size 48 --name wider
python train.py —-input_size 48
