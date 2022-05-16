from skimage.metrics import structural_similarity as ssim

import sys


import numpy as np
import math
import cv2
from cv2 import INTER_AREA, addWeighted, threshold
import random
from sqlalchemy import null
from dataclasses import dataclass
from typing import Any



blockSize = 8
cropThreshold = 0.02
valid_chars = 'abcdefghijklmnopqrstuvwxyz '
guessable_characters = 'abcdefghijklmnopqrstuvwxyz '
max_length = 20
offsetThreshold = 0.10
redacted_image = None
blank_background = None
best_guess = ''
best_guess_img = None
curr_Offsetx =0
curr_Offsety =0

def gatherResults(guess,totalScore,score,imageData, offset_x, offset_y):
    best_score =1
    if (totalScore < best_score):
        best_score = totalScore
        best_guess = guess
        #best_guess_img =imageData
        cv2.imshow('current guess img', imageData)
        #cv2.imshow('best-preview-image', best_guess_img)
        cv2.imshow('redacted_image', redacted_image)
        #print("Best Guess:",best_guess)
        #print("best_score:",totalScore)
    


    
    print("current_guess:",guess)
    print("offset_x:",offset_x,", Offset y:",offset_y )
    print(" ")
    cv2.waitKey(50)

def offsetDiscovery(offset_x,offset_y, success):
    curr_Offsetx =offset_x 
    curr_Offsety =offset_y 
    print("offset_x:",curr_Offsetx,", Offset y:",curr_Offsety )



def randomAlphabet():
    """returns a string of the shuffled alphabet"""
    randomchars = ''.join(random.sample(valid_chars,len(valid_chars)))
    print(randomchars)
    return randomchars

def createImg(width, height, rgb_color=(255, 255, 255)):
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image


def getBlueMargin(image):

    margin = 0
    center = 0
    found = False
    newImg = image
    height = newImg.shape[0]

    width = newImg.shape[1]
    
    
    for x in range(0,width):
        blue,green,red = newImg[42][x]
        if (found == False and blue == 255 and green != 255 and red != 255):
            found = True
            margin = x
            
    found = False
    topBlue = 0
    botBlue = 0
    for y in range(0,height):
    

        blue,green,red = newImg[y][margin]
        if (found == False and blue == 255 and green != 255 and red != 255):
            
            found = True
            topBlue = y

            
        if (found == True and blue == 255 and green == 255 and red == 255):
            found = False
            botBlue = y

        
    center = (topBlue + botBlue) / 2

    return [margin, center]

def getMargins(image):
    newImg = image
    dimensions = newImg.shape
    height = newImg.shape[0]
    width = newImg.shape[1]

    rowsize = width * 4

    hitRed = False
    left_edge = 0
    for y in range(0,int(height)):
        for x in range(0,width):
            red,green,blue = newImg[y][x],newImg[y][x],newImg[y][x]
            if (x > left_edge and hitRed == False and (green > 0)):
                hitRed = True
                left_edge = x

    return left_edge

def getLeftEdge(image):
    newImg = image
    dimensions = newImg.shape
    height = newImg.shape[0]
    width = newImg.shape[1]
    crowsize = width * 4
    left_edge = width

    for y in range(0,height):
        for x in range(0,width):
            blue,green,red = newImg[y][x]
            if (x < left_edge and green != 255 and red != 255 and blue != 255):
                left_edge = x
            


    if (left_edge == width):
        return 0
    

    return left_edge

def createBlank(h,w):
    c = (255,255,255)
    image = np.zeros((h,w,3), )
    color = tuple(reversed(c))
    image[:] = color #set all elements to 255
    return image


@dataclass
class Request:
    command: str
    redacted_image: np.ndarray
    totalLength: float
    previousimage: np.ndarray
    text:str
    charSet: list
    offset_x: float
    offset_y: float
    totalScore: float
    score: float
    imageData: np.ndarray
    tooBig:bool
    
def makeGuess(command, guess, previousimage, offset_x, offset_y):

  request = Request(0,0,0,0,0,0,0,0,0,0,0,0)
  request.command = command
  request.redacted_image = redacted_image
  request.totalLength = max_length
  request.text = guess
  request.previousimage = previousimage
  request.charSet = guessable_characters
  request.offset_x = offset_x
  request.offset_y = offset_y
                
  result = redact(request)
  return result


def decode(encoded_image):
    comparisons = np.empty((0,3), float)                 
    for i in range(0, blockSize):
        for j in range(0, blockSize):
            lowest_error = 1
            for q in range(0, len(valid_chars)):
                initial_guess = makeGuess("guess-text", valid_chars[q], None, i, j) 
                #gatherResults(initial_guess.guess,initial_guess.totalScore,initial_guess.score,initial_guess.imageData, initial_guess.offset_x, initial_guess.offset_y)
                if(initial_guess.score < lowest_error and initial_guess.score >0):
                    gatherResults(initial_guess.guess,initial_guess.totalScore,initial_guess.score,initial_guess.imageData, initial_guess.offset_x, initial_guess.offset_y)
                    lowest_error = initial_guess.score

            if(lowest_error < offsetThreshold):
                #gatherResults(initial_guess.guess,initial_guess.totalScore,initial_guess.score,initial_guess.imageData, initial_guess.offset_x, initial_guess.offset_y)
                comparisons = np.concatenate( ( comparisons, [[lowest_error, i, j]] ) , axis=0)
    comparisons = np.sort(comparisons, axis=0)
    
    for j in range(0, comparisons.shape[0]):
        error_list = np.empty((0,2), float)
        for i in range(0, len(valid_chars)):
            initial_guess = makeGuess("guess-text",valid_chars[i], None, comparisons[j,1], comparisons[j,2])

            if(initial_guess.score < offsetThreshold):
                gatherResults(initial_guess.guess,initial_guess.totalScore,initial_guess.score,initial_guess.imageData, initial_guess.offset_x, initial_guess.offset_y)
                error_list = np.concatenate( ( error_list, [[initial_guess.score, valid_chars[i]]] ) , axis=0)

        error_list = np.sort(error_list)
        for i in range(0, error_list.shape[0]):
            decodeRecursive(error_list[i,1], 0, comparisons[j,1], comparisons[j,2])

def decodeRecursive(letters_guess, error, xoff, yoff):
    if(len(letters_guess) == max_length):
        return  
    comparisons = np.empty((0,2), float)
    initial_estimate = makeGuess("guess-text", letters_guess, None, xoff, yoff)
    gatherResults(initial_estimate.guess,initial_estimate.totalScore,initial_estimate.score,initial_estimate.imageData, initial_estimate.offset_x, initial_estimate.offset_y)
    if(initial_estimate.tooBig != True):
        for i in range(0, len(valid_chars)):
            next_letter_depth = letters_guess + valid_chars[i]
            #print("next_letter_depth",next_letter_depth)
            next_guess = makeGuess("guess-text",next_letter_depth, initial_estimate.imageData, xoff, yoff) 
            gatherResults(next_guess.guess,next_guess.totalScore,next_guess.score,next_guess.imageData, next_guess.offset_x, next_guess.offset_y)

            case_threshold = offsetThreshold
            if(valid_chars[i]== " "):
                case_threshold = 0.5 
            if(next_guess.score < case_threshold):
                gatherResults(next_guess.guess,next_guess.totalScore,next_guess.score,next_guess.imageData, next_guess.offset_x, next_guess.offset_y)
                comparisons = np.concatenate( ( comparisons, [[next_guess.score, next_letter_depth]] ) , axis=0)
               

    comparisons = np.sort(comparisons)
    for i in range(0, comparisons.shape[0]):
        decodeRecursive(comparisons[i,1], comparisons[i,0], xoff, yoff)


def combine_two_color_images(image1, image2):# https://stackoverflow.com/questions/48979219/cv2-composting-2-images-of-differing-size

    foreground, background = image1, image2
    foreground_dimensions = foreground.shape
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    alpha =0.5
    
    blended_portion = cv2.addWeighted(foreground,
                alpha,
                background[int(alpha):int(foreground_height) , int(alpha):int(foreground_width)],
                1 - alpha,
                0,
                background)
    background[:foreground_height,:foreground_width,:] = blended_portion
    return background

def combine_two_color_images_with_anchor(image1, image2, anchor_y, anchor_x):
    foreground = image1
    background = image2
    # Check if the foreground is inbound with the new coordinates and raise an error if out of bounds
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    background_height = background.shape[0]
    background_width = background.shape[1]

    if foreground_height+anchor_y > background_height or foreground_width+anchor_x > background_width:
        raise ValueError("The foreground image exceeds the background boundaries at this location")
    
    alpha =1

    # do composite at specified location
    start_y = anchor_y
    start_x = anchor_x
    end_y = anchor_y+foreground_height
    end_x = anchor_x+foreground_width
    blended_portion = cv2.addWeighted(foreground,
                alpha,
                background[start_y:end_y, start_x:end_x,:],
                1 - alpha,
                0,
                background)
    background[start_y:end_y, start_x:end_x,:] = blended_portion
    return background

def redact(message):
    global redacted_image
    length = len(message.text)
    totalLength = message.totalLength

    image = redacted_image #loads

    result = 0
    textImg = createImg(400, 120, rgb_color=(255, 255, 255))

    font = cv2.FONT_HERSHEY_COMPLEX
    
    textImg = cv2.putText(textImg ,message.text, (8,42), font, 1,(0,0,0), 2 ,cv2.LINE_AA)
    textImg = cv2.putText(textImg ,'||', (len(message.text)*20 + 11, 42), font, 1,(255,0,0), 4 ,cv2.LINE_AA)

    dimensions = redacted_image.shape
    Rheight = redacted_image.shape[0]
    Rwidth = redacted_image.shape[1]
    #previousimage = message.previousimage
    previousimage = message.previousimage

    newImg = textImg
    dimensions = newImg.shape
    height = newImg.shape[0]
    width = newImg.shape[1]
    imageData = newImg

    offset_x = message.offset_x
    offset_y = message.offset_y



    margins = getBlueMargin(imageData)

    blueMargin = margins[0]
    imageCenter = margins[1]

    newImg = imageData[int(offset_y):int(height-offset_y), int(offset_x):int(blueMargin-offset_x)]

    imageCenter -= offset_y
    height = newImg.shape[0]
    width = newImg.shape[1]

    w, h = 8, 5
    averagePixels = np.zeros((math.ceil(height / blockSize), math.ceil(width / blockSize), 4), np.uint8)
    APheight = averagePixels.shape[0]
    APwidth = averagePixels.shape[1]
    remainder = blockSize - (width % blockSize)

    if (remainder < blockSize):
        white = (255, 255, 255)
        blank_canvass = createImg(width + remainder, height, rgb_color=white)
        newImg = combine_two_color_images_with_anchor( newImg,blank_canvass, 0, 0)

        

        


    original = newImg

    dimensions = original.shape
    Oheight = original.shape[0]
    Owidth = original.shape[1]



    
    for y in range(0,Oheight):
        for x in range(0,Owidth):
            
            
            upper_left_x = np.floor((x / blockSize) * blockSize)
            
            upper_left_y = np.floor((y / blockSize) * blockSize)
            
            rowsize = Owidth
            conv_x = int(upper_left_x/blockSize)
            conv_y = int(upper_left_y/blockSize)
            
            if (averagePixels[conv_y][conv_x][0] == 0):
                red = 0
                green = 0
                blue = 0
                
                pixelCount = 0
                for i in range(0,blockSize):
                    for j in range(0,blockSize):
                        
                        redIndex_x =upper_left_x + i
                        redIndex_y =upper_left_y + j
                        dataLength = Oheight*Owidth
                        
                        if (redIndex_x < Owidth and redIndex_y < Oheight):

                            
                            b,g,r = original[int(upper_left_y + j)][int(upper_left_x + i)]
                            red += r
                            pixelCount += 1

                        
                        greenIndex_x =upper_left_x + i
                        greenIndex_y =upper_left_y + j
                        if (redIndex_x < Owidth and redIndex_y < Oheight):
                            b,g,r = original[int(upper_left_y + j)][int(upper_left_x + i)]
                            green += g
                        

                        
                        blueIndex_x =upper_left_x + i
                        blueIndex_y =upper_left_y + j
                        if (redIndex_x < Owidth and redIndex_y < Oheight):
                            b,g,r = original[int(upper_left_y + j)][int(upper_left_x + i)]
                            blue += b


                        
                averagePixels[conv_y][conv_x][0] = red / pixelCount
                averagePixels[conv_y][conv_x][1] = green / pixelCount
                averagePixels[conv_y][conv_x][2] = blue / pixelCount
                


            
            original[y][x] = averagePixels[conv_y][conv_x][0],averagePixels[conv_y][conv_x][1],averagePixels[conv_y][conv_x][2]
            
                        
                        
    freshly_pixelated = original

    dimensions = freshly_pixelated.shape
    Fheight = freshly_pixelated.shape[0]
    Fwidth = freshly_pixelated.shape[1]
    
    left_edge = getLeftEdge(freshly_pixelated)


    #Step 1) Crop image to the same size as the original and adjust brightness to be identical
    threshold = 0.02
    percent_tried = length / totalLength
    #    We need to vertically crop the guess image down to the size of the answer
    #      but also keep the cropping along blocksize boundaries
    adjustedCenter = imageCenter - (imageCenter % blockSize) + 4
    newWidth =Fwidth
    if(Fwidth==32 and left_edge ==0):
        newWidth=Fwidth-blockSize
    newGuess = freshly_pixelated[8:Rheight+8, left_edge:newWidth]
    NGheight = newGuess.shape[0]
    NGwidth = newGuess.shape[1]

    cropped_redacted_image = redacted_image

    dimensions = cropped_redacted_image.shape
    CRheight = cropped_redacted_image.shape[0]
    CRwidth = cropped_redacted_image.shape[1]

    guess_image = newGuess

    dimensions = guess_image.shape
    Gheight = guess_image.shape[0]
    Gwidth = guess_image.shape[1]


    if(message.text =='ba'):
        cv2.imshow('guess_image', guess_image)
        cv2.imshow('cropped_redacted_image', cropped_redacted_image)
        cv2.waitKey(10000)
    # Step 2) Find the area where our new image changed (compared to the previous guess)
    left_boundary = 0
    right_boundary = 0
    NoneType = type(None)
    
    if (type(message.previousimage) == NoneType):
      right_boundary = Gwidth
      
    else:
      
      prev_image = previousimage

      # Scale up the previous image to make it the same size as our new guess
      prev_image_scaled = blank_background
      
      prev_image_scaled = combine_two_color_images_with_anchor(prev_image, prev_image_scaled, 0, 0)


      dimensions = prev_image_scaled.shape
      PIheight = prev_image_scaled.shape[0]
      PIwidth = prev_image_scaled.shape[1]

      if (Gwidth <= PIwidth ):
        #prev_image_scaled.crop(0, 0, width, prev_image_scaled.bitmap.height)
        prev_image_scaled = prev_image_scaled[0:PIheight, 0:Gwidth ]
        

      

      # This is the changed area. The diff image is red where it was different

      grayguess_image = cv2.cvtColor(guess_image, cv2.COLOR_BGR2GRAY)
      grayprev_image_scaled = cv2.cvtColor(prev_image_scaled, cv2.COLOR_BGR2GRAY)

      diff_boundedPercent,diff=ssim(grayguess_image, grayprev_image_scaled, full = True)

      left_boundary = getMargins(diff)

      #    If the images are identical, then use just the size of the image that changed
      #      This can happen when we guess a bunch of spaces in a row and nothing has changed
      if (left_boundary == 0):
        left_boundary = PIwidth
      
    

    # Step 3) Crop our image down to just the area that changed
    
 
    newImg = guess_image[0:Gheight, left_boundary:(Gwidth-left_boundary)]

    NIheight = newImg.shape[0]
    NIwidth = newImg.shape[1]

    
    if (blueMargin > CRwidth):
        white = (255, 255, 255)
        blank_canvass = createImg(CRwidth*2, CRheight, rgb_color=white)

        image = combine_two_color_images_with_anchor(cropped_redacted_image, blank_canvass, 0, 0)
        cropped_redacted_image = image

    
    # Crop the answer image down to the same size as the guess image

    cropped_redacted_image = cropped_redacted_image[0:CRheight, left_boundary:NIwidth]

    CR2height = cropped_redacted_image.shape[0]
    CR2width = cropped_redacted_image.shape[1]

    # Step 4) Crop the right-most edge off both the guess and answer
    #  This is because there's a large error on that last block due to the next letter bleeding over.
    # Adjust the blue margin over because we cropped a bunch since the last measurement
    adjustedBlueMargin = ((blueMargin-left_boundary)-left_edge)-offset_x
    if (NIwidth > adjustedBlueMargin):

      newImg = newImg[0:int(Gheight), 0:int(adjustedBlueMargin)]

      cropped_redacted_image = cropped_redacted_image[0:int(CR2height), 0:int(adjustedBlueMargin)]

      CR3height = cropped_redacted_image.shape[0]
      CR3width = cropped_redacted_image.shape[1]
      #print("CR3height ",CR3height )
      #print("CR3width ",CR3width )
      #print("cropped_redacted_image3" )
      
    
    if(message.text =='ba'):
        cv2.imshow('newImg', newImg)
        cv2.imshow('cropped_redacted_image', guess_image)
        cv2.waitKey(10000)
   # Step 5) Report the similarity score for just that area


    graynewImg = cv2.cvtColor(newImg.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    graycropped_redacted_image = cv2.cvtColor(cropped_redacted_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    

    diff_boundedPercent,diff=ssim(graycropped_redacted_image, graynewImg,full = True)

    # Step 6) Report the similarity score for the whole image
    #    Match up the sizes of the images so we can diff them
    scaled_guess_image = guess_image
    SGheight = scaled_guess_image.shape[0]
    SGwidth = scaled_guess_image.shape[1]
    if (SGwidth < Rwidth):
      white = (255, 255, 255)
      blank_canvass = createImg(Rwidth, Rheight, rgb_color=white)

      scaled_guess_image = combine_two_color_images_with_anchor(scaled_guess_image, blank_canvass, 0, 0)

      SGheight = scaled_guess_image.shape[0]
      SGwidth = scaled_guess_image.shape[1]

      grayscaled_guess_image = cv2.cvtColor(scaled_guess_image, cv2.COLOR_BGR2GRAY)
      grayredacted_image = cv2.cvtColor(redacted_image, cv2.COLOR_BGR2GRAY)

      diff_finalPercent=ssim(grayscaled_guess_image, grayredacted_image)


    dataURI = scaled_guess_image



    request = Request(0,0,0,0,0,0,0,0,0,0,0,0)
    request.command = message.command
    request.guess = message.text
    request.totalScore = diff_finalPercent
    request.score = diff_boundedPercent
    request.imageData = dataURI
    request.offset_x = offset_x
    request.offset_y = offset_y
    if(Rwidth < SGwidth):
        request.tooBig = True
    else:
        request.tooBig = False
    
    return request

    

if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2020, Project: data decoder "undecoderent"')
    print('==================================================')

    # ===================================
    # example:
    # python undecoderent.py obfuscated.png unobfuscated_result.png
    # ===================================
    path_file_obfuscated_image = "newguess.png"
    path_file_unobfuscated_result_image = "unobfuscated.png"
    path_file_reobfuscated_result_image = "reobfuscated.png"
    
    obfuscated_image = cv2.imread(path_file_obfuscated_image)
    unobfuscated_image = cv2.imread(path_file_unobfuscated_result_image)
    reobfuscated_image = cv2.imread(path_file_reobfuscated_result_image)
    dimensions = obfuscated_image.shape
    height = obfuscated_image.shape[0]
    width = obfuscated_image.shape[1]
    redacted_image = obfuscated_image
    ##cv2.imshow('composited image', redacted_image)
    nemo = cv2.cvtColor(redacted_image, cv2.COLOR_BGR2RGB)

    #print(type(redacted_image))
    white = (255, 255, 255)
    blank_background = createImg(width, height, rgb_color=white)
    nemo = cv2.cvtColor(blank_background, cv2.COLOR_BGR2RGB)

    message = 0
    img_offset_x =0
    img_offset_y =0
    length = 0
    totalLength =12
    #makeGuess("guess-text", guess, previousimage, offset_x, offset_y):
    #result = redact(message, obfuscated_image, img_offset_x, img_offset_y, length, totalLength)
    #makeGuess("redact-text", "badeedq", None, 0, 0)
    decode(obfuscated_image)
    #cv2.imwrite('new.png', result)
    
    # ===== interpolate an intermediate frame at t, t in [0,1]
    #unobfuscated_image = decode(obfuscated_image=obfuscated_image)
    #reobfuscated_image = encode(unobfuscated_image)