#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:12:45 2023

@author: tawsinua
"""
import fitz
#from PyPDF2 import PdfReader
import cv2
from io import BytesIO

from PIL import Image
import base64
import numpy as np
from PIL import Image
import cv2
import io
import warnings
#from pdf2image import convert_from_bytes
import re
from datetime import datetime
from collections import namedtuple
warnings.filterwarnings("ignore")
from dateutil import parser
import os
from os import listdir
from os.path import isfile, join
from yunet import YuNet
import csv


def visualize(image, results, img_name, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    #print(output.shape)
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    #if fps is not None:
     #   cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        #print(bbox)
        #output = cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        cropped_img = output[bbox[1]-20:bbox[1]+bbox[3]+20, bbox[0]-20:bbox[0]+bbox[2]+20]
        #print(cropped_img.shape)
        #cropped_depth = depth[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        #cropped_img = cv2.resize(cropped_img, dsize=(250,250))
        #cropped_depth = cv2.resize(cropped_depth, dsize=(224,224))
        conf = det[-1]
        #output=cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        '''for idx, landmark in enumerate(landmarks):
            #output=cv.circle(output, landmark, 2, landmark_color[idx], 2)
            head_tail_rgb = os.path.split(img_name)
            #head_tail_depth = os.path.split(depth_name)
            
            cv2.imwrite('./Face_data_1/' + str(head_tail_rgb[1]),cropped_img)
            #cv.imwrite('./Bonafide_depth/' + str(head_tail_depth[1]),cropped_depth)'''

    return cropped_img









backends = [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_CUDA]
targets = [cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16]
help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
# file path you want to extract images from
conf_threshold=0.9
model='./packageData/face_detection_yunet_2022mar.onnx'
nms_threshold=0.3
top_k=5000
backend=backends[0]
target=targets[0]


detector = YuNet(modelPath=model,
                  #inputSize=[500, 500],
                  confThreshold=conf_threshold,
                  nmsThreshold=nms_threshold,
                  topK=top_k,
                  backendId=backend,
                  targetId=target)






def predict(pdf, file_name):
    result = []
    
    count=0
    
    chip=0
    chip_image = []
    cropped_image = []
    pages_index = []
    images_index = []
    images_ext = []
    
    #pdfpages = pdf.getNumPages()
    for page_index in range(len(pdf)):
        page0 = pdf[page_index]
        #image_list = extract_images_from_pdf_page(page0)
        # get image list
        image_list = page0.get_images()


        # printing number of images found in this page
        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)
        for image_index, img in enumerate(image_list, start=1):
            # get the XREF of the image
            base_image = pdf.extract_image(img[0])
            
            #print(base_image)
            
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image)
            height, width, _ = image.shape
        #print(len(image_list))
        #number += len(image_list)
        
        #for pdf_image in image_list:
            #img = Image.open(pdf_image.data)
            #image = np.array(img)
            #height, width, _ = image.shape

                
            if(round(float(height/width), 1) == 1.3):
                if (height >= 100 or width >= 100):
                    chip = 1
                    chip_image.append(image)
            
            
            else:
                if(chip):
                    continue
                if (height >= 100 or width >= 100):
                    # Inference
                    #detector.setInputSize([w, h])
                    #image = cv2.resize(image, dsize=(500,500))
                    #h, w, _ = image.shape
                    # Inference
                    detector.setInputSize([width, height])
                    
                    
                    results = detector.infer(image)
                    #print(results.shape)
                    if results is not None:
                        # Print results
                        #print('{} faces detected.'.format(results.shape[0]))
                        '''for idx, det in enumerate(results):
                            print('{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(
                                idx, *det[:-1])
                            )'''
                
                        # Draw results on the input image
                        #cropped_img = visualize(image,  results, file_name)
                        cropped_img = image
                        cropped_image.append(cropped_img)
                    else:
                        print('No Face Detected')
                        continue
                    
    max_width=0          
    if(chip_image):
        for i in range(len(chip_image)):
            height, width, _ = chip_image[i].shape
            if(width > max_width):
                largest_image = chip_image[i]
                max_width = width                
                image =  Image.fromarray(chip_image[i])
                #buffered = BytesIO()
                #image.save(buffered, format="JPEG")
                #img_str = base64.b64encode(buffered.getvalue())
                path='output_img/'+file_name
                isExist = os.path.exists(path)
                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(path)
         
                file_name_tmp = path + '/' +file_name + '_' + str(count) +'.jpg'
            
                image.save(file_name_tmp)
                count=count+1
        
        #result.append(image)

            
            

    elif(cropped_image):
        for i in range(len(cropped_image)):
            height, width, _ = cropped_image[i].shape
            if(width > max_width):
                largest_image = cropped_image[i]
                max_width = width                
                image =  Image.fromarray(cropped_image[i])
                path='output_img/'+file_name
                isExist = os.path.exists(path)
                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(path)
                #buffered = BytesIO()
                file_name_tmp = path + '/' + file_name+ '_' + str(count) +'.jpg'
                #print(file_name)
                image.save(file_name_tmp)
                count=count+1
                #img_str = base64.b64encode(buffered.getvalue())
            
        #result.append(image)
       
    
    #print(len(result))
    #json = {"Result": list(result)}
    
    
    return file_name_tmp
        
     

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image



def MRZ_check(blocks):
    
    for block in blocks:

        try:
            string = block[4]#.replace("\n", " ")
            pattern = "MRZ"

            #Will return all the strings that are matched
            MRZ = re.findall(pattern, string)
            if(MRZ):
                #strings.append(string)
                #print(string)
                
                return string
            #print(dates)
        except Exception:
            continue
        
    
    


def Info_check(blocks):
    strings = []
    PID = ""
    for block in blocks:

        try:
            string = block[4].replace("\n", " ")
            #print(string)

            if(string.split(" ")[0].isnumeric()):
                #print(string)
                #if(len(string)==11):
                PID = string.split(" ")[0]
                print(PID)
                    
                    
            #The regex pattern that we created
            pattern = "\d{2}[/.]\d{2}[/.]\d{4}"

            #Will return all the strings that are matched
            dates = re.findall(pattern, string)
            if(dates):
                strings.append(string)
            #print(dates)
        except Exception:
            continue
   

    
        if len(strings) == 3:
            
            
            
            if(strings[1][:-1].split(" ")[-1].isnumeric()):
            
                User_info = {
                    
                    "Entry_info" : strings[0],
                    "Customer_info" : strings[1][:-1],
                    "Document_info" : strings[2]
                
                    }
            else:
                User_info = {
                    
                    "Entry_info" : strings[0],
                    "Customer_info" : strings[1][:-1]+ ' ' + PID,
                    "Document_info" : strings[2]
                
                    }
            
            '''for key, value in User_info.items():
                print(key, ":", value)'''
                
            return User_info
    
    




def compare_dates(date1, date2):

    if date1 > date2:
        return 1
    else:
        return 0
            
    


def csv_export(User_info, MRZ_str, Id, img_path, start_flag, chip_flag):
    with open('Pdf_Extractor.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        
        personal_ID = '_' + str(User_info["Customer_info"].split(' ')[-1])
        
        Document_ID = '_' + str(User_info["Document_info"].split(' ')[1])
        
        Document_type = str(User_info["Document_info"].split(' ')[0])
        
        #print(Document_type)
        
        
        date= str(User_info["Document_info"].split(' ')[-2])
        
        #print(date)
        
        day, month, year = date.split('.')
        
        #print(day)
        
        
        dt1 = datetime(int(year), int(month), int(day))
        today = datetime.now()
        
        #print(dt1)
        #print(today)
        Validity = compare_dates(dt1, today)
        
        
        
        
        #print(personal_ID)
        
        if start_flag==0:
            writer.writerow(["Personal_ID (NOR-ID/D-Number)", "Entry_info", "Customer_info", "Document_type", "Document_ID",  "MRZ", "Image_path", "Chip_flag", "Validity"])
            

        
        writer.writerow([personal_ID, User_info["Entry_info"], User_info["Customer_info"], Document_type, Document_ID,  MRZ_str[5:], img_path, int(chip_flag), str(Validity)])
        
        

        
def RFID_check(blocks):
    #image = block.get_images()
    
    for block in blocks:

        try:
            string = block[4].replace("\n", "")
            #print(string)
            pattern_1 = "RFID pasfoto"
            pattern_2 = "RFID passfoto"

            #Will return all the strings that are matched
            #RFID = 
            
            if(re.findall(pattern_1, string) or re.findall(pattern_2, string)):
                #strings.append(string)
                print(string)
                
                return 1
            
            #print(dates)
        except Exception:
            continue
    return 0
    
    
    
        
        
           

# Defining main function
def pdf_extract(str_path):
    
    mypath = str_path
    
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    #print(len(onlyfiles))
    
    start_flag = 0
    
    Id=1
    
    
    
    
    for file in onlyfiles:
        chip_flag = 0
        print("____________________________________")
        # open the file
        file_name = file.split(".")[0]
        #print(file_name)
        pdf_file = fitz.open(mypath + '/' + file)
        
        img_path = predict(pdf_file, file_name)
        
        
        
        #print(mypath + '/' + file)
        
        if(len(pdf_file)>1):
        
        
            for page_index in range(len(pdf_file)):
            
                
        
                
                    
                    page = pdf_file[page_index]
                    #page_1 = pdf_file[0]
                    
                    #page_2 = pdf_file[1]
                    
                    #page_last = pdf_file[-1]
                
                    #blocks_1 = page_1.get_text("blocks")  # extract sorted words
                    
                    
                    #blocks_2 = page_2.get_text("blocks")
                    
                    #blocks_last = page_last.get_text("blocks")
                    
                    blocks = page.get_text("blocks")
                    
                    #print(blocks_last)
                    
                    if(chip_flag==0):
                        
                        chip_flag = RFID_check(blocks)
                    
                    
                    
                        
                    
                    #print(blocks_last)
                    if(page_index == 0):
                        User_info = Info_check(blocks)
                        #print(blocks)
                    if(page_index == 1):    
                        MRZ_str = MRZ_check(blocks)
                    
                    
                    
                    
                    start_flag=1
                    
                    Id=Id+1
            csv_export(User_info, MRZ_str, Id, img_path, start_flag, chip_flag)    
                
                
        else:
            
            #page_last = pdf_file[-1]
            #blocks_last = page_last.get_text("blocks")
            #chip_flag = RFID_check(blocks_last)
            with open('Pdf_Extractor.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                
                #personal_ID = '_' + str(User_info["Customer_info"].split(' ')[-2])
                
                #print(personal_ID)
                
                if start_flag==0:
                    writer.writerow(["Personal_ID (NOR-ID/D-Number)", "Entry_info", "Customer_info", "Document_type", "Document_ID",  "MRZ", "Image_path", "Chip_flag", "Validity"])
                    

                
                writer.writerow(['######', '######', '######', '######', '######', '######', img_path, 0, '######'])
                start_flag =1
                
    
                
                
            
    

        #doc = fitz.open('test_pdf/Keesing_report_gl.pdf')
        
        
        
        
        
    
    
    
    

    
    
  
  
# Using the special variable 
# __name__
#if __name__=="__main__":
#    main()
