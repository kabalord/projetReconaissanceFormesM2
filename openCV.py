import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read the image for erosion
img1= cv2.imread("/Users/walterroaserrano/Desktop/M22021/907PythonPourIA/hebbache/GRAINS-1.JPG",0)
m,n= img1.shape #Acquire size of the image
plt.imshow(img1, cmap="gray")


# Define the structuring element
# k= 11,15,45 -Different sizes of the structuring element
k=15
SE= np.ones((k,k), dtype=np.uint8)
constant= (k-1)//2


#Define new image
imgErode= np.zeros((m,n), dtype=np.uint8)


#Erosion without using inbuilt cv2 function for morphology
for i in range(constant, m-constant):
  for j in range(constant,n-constant):
    temp= img1[i-constant:i+constant+1, j-constant:j+constant+1]
    product= temp*SE
    imgErode[i,j]= np.min(product)

plt.imshow(imgErode,cmap="gray")
cv2.imwrite("Eroded3.png", imgErode)



#Erosion using cv2 inbuilt function to obtain structuring element and perform erosion
SE1= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
imgErodenew= cv2.erode(img1,SE1,1)
plt.imshow(imgErodenew,cmap="gray")




#=========================#




#Read the image for dilation
img2= cv2.imread("/Users/walterroaserrano/Desktop/M22021/907PythonPourIA/hebbache/GRAINS-1.JPG",0)
p,q= img2.shape
plt.imshow(img2, cmap="gray")
#cv2.imwrite("text.png", img2)


img_new= cv2.imread("/Users/walterroaserrano/Desktop/M22021/907PythonPourIA/hebbache/GRAINS-1.JPG",0)
cv2.imwrite("text.png",  img_new)

#Define new image for dilation
imgDilate= np.zeros((p,q), dtype=np.uint8)


#Define the structuring element 
SED= np.array([[0,1,0], [1,1,1],[0,1,0]])
constant1=1

#Dilation
for i in range(constant1, p-constant1):
  for j in range(constant1,q-constant1):
    temp= img2[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
    product= temp*SED
    imgDilate[i,j]= np.max(product)

plt.imshow(imgDilate,cmap="gray")
cv2.imwrite("Dilated.png", imgDilate)


#=========================#


#Use of opening and closing for morphological filtering
#Perform the following operation on the noisy fingerprint image
# [(((AoB)d B) e B)]

#AoB= (A e B) d B
#o=opening, e=erosion,d=dilation
# Here inbuilt function of erosion and dilation from cv2 module is used.
#To form the structuring element also, inbuilt function from cv2 is used


#Function for erosion
def erosion(img, SE):
  imgErode= cv2.erode(img,SE,1)
  return imgErode


#Function for dilation
def dilation(img, SE):
  imgDilate= cv2.dilate(img,SE,1)
  return imgDilate



#Read the image for dilation
img= cv2.imread("/Users/walterroaserrano/Desktop/M22021/907PythonPourIA/hebbache/GRAINS-1.JPG",0)
img_finger=cv2.imwrite("finger.png", img)
SE= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) #Define the structuring element using inbuilt CV2 function
AeB= erosion(img,SE) #Erode the image
AoB= dilation(AeB, SE) #Dilate the eroded image. This gives opening oepration
AoBdB= dilation(AoB,SE) #dilate the opened image followed by ersoion. This will give closing of the openeed image
AoBdBeB= erosion(AoBdB, SE)






plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(3,2,2)
plt.title("E(A,B)")
plt.imshow(AeB, cmap="gray")
plt.subplot(3,2,3)
plt.title("O(A, B)")
plt.imshow(AoB, cmap="gray")
plt.subplot(3,2,4)
plt.title("D(O(A,B), B)")
plt.imshow(AoBdB, cmap="gray")
plt.subplot(3,2,5)
plt.title("C((O(A,B),B),B)")
plt.imshow(AoBdBeB, cmap="gray")
cv2.imwrite("finger_filtered.png", AoBdBeB)

#==============================#

#erosion CV2
imageErode= erosion(img,SE)
plt.imshow(imageErode, cmap="gray")


#dilatation CV2
imageDilate= dilation(img,SE)
plt.imshow(imageDilate, cmap="gray")











