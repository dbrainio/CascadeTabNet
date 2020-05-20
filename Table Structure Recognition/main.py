from border import border
from mmdet.apis import inference_detector, show_result, init_detector
import cv2
from Functions.blessFunc import borderless
import lxml.etree as etree
import glob

############ To Do ############
image_path = '/home/avens/data/tables/3A070FF1CE656A2E161E5CEC6ECCCB48.jpg'
xmlPath = '/mnt/hdd1/users/avens/logdir/tables_output'

config_fname = "/home/avens/projects/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py"
checkpoint_path = "/home/avens/data/tables_checkpoints/"
epoch = 'epoch_36.pth'
##############################


model = init_detector(config_fname, checkpoint_path+epoch)

# List of images in the image_path
imgs = glob.glob(image_path)
for i in imgs:
    print(i)
    result = inference_detector(model, i)
    print(result)
    res_border = []
    res_bless = []
    res_cell = []
    root = etree.Element("document")
    ## for border
    for r in result[0][0]:
        if r[4]>.85:
            res_border.append(r[:4].astype(int))
    ## for cells
    for r in result[0][1]:
        if r[4]>.85:
            r[4] = r[4]*100
            res_cell.append(r.astype(int))
    ## for borderless
    print(result[0][2])
    for r in result[0][2]:
        if r[4]>.85:
            res_bless.append(r[:4].astype(int))
    print("bless")
    print(res_bless)
    print("border")
    ## if border tables detected 
    if len(res_border) != 0:
        ## call border script for each table in image
        print("for each table")
        for res in res_border:
            print("for res")
            try:
                print("try")
                cur_img = cv2.imread(i)
                print("img")
                root.append(border(res,cur_img))
                print("border")
            except Exception as e:
                print(e)
                print('ex caught')
                pass
    print("border fin")
    if len(res_bless) != 0:
        if len(res_cell) != 0:
            for no,res in enumerate(res_bless):
                cur_img = cv2.imread(i)
                root.append(borderless(res,cur_img,res_cell))
    print("myfile open")
    myfile = open(xmlPath+i.split('/')[-1][:-3]+'xml', "w")
    myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    myfile.write(etree.tostring(root, pretty_print=True,encoding="unicode"))
    myfile.close()