import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np

image_org = mpimg.imread('C:/Users/user/Desktop/test/image1.jpg')

# open file
f = open('test.txt', mode='w')
l = open('label.txt',mode='w')
R = open('new_label.txt',mode='w')

#print(image_org.shape)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

gray = rgb2gray(image_org)
size = gray.shape # Gray = R*0.299 + G*0.587 + B*0.114 ; size: 1104x1399
m = size[0] #rows
n = size[1] #columns
threshold = 150

# gray2binary
for i in range(m):
    for j in range(n):
        if gray[i,j] > threshold:
            gray[i,j] = 0
            f.write(str(int(gray[i,j])))
        else:
            gray[i,j] = 1
            f.write(str(int(gray[i,j])))
    f.write('\n')


image = gray

label = np.ones([m,n])
new = 0

# link array
link = []
id = 0 # link index also present object number

# neighbor and current_poit=image[i,j]
def neighbor(i,j,label):
    # left
    left = label[i-1,j]
    # above
    above = label[i,j-1]
    neighbor_array = [left,above]
    return neighbor_array

# first pass
for row in range(m):
    for column in range(n):
        # no object
        if image[row,column] == [0] :
            label[row, column] = 0
            l.write(str(int(label[row,column])))
        # object
        else : # check neighbor label
            current_neighbor = neighbor(row,column,label)

            # current is new label
            if current_neighbor == [0,0]:
                new= new + 1
                label[row, column] = new
                l.write(str(int(label[row, column])))

            # neighbor got label
            else :
                # only one neighbor labeling => choose the large one (the only label)
                if np.min(current_neighbor) == 0 or current_neighbor[0] == current_neighbor[1] :
                    label[row,column] = np.max(current_neighbor)
                    l.write(str(int(label[row, column])))

                else:
                    label[row,column] = np.min(current_neighbor)
                    l.write(str(int(label[row, column])))
                    if id == 0:
                        link.append(current_neighbor)
                        id = id +1
                        #print(link)
                    else:
                        check = 0
                        for k in range(id) :
                            # 交集
                            tmp = set(link[k]).intersection(set(current_neighbor))
                            if len(tmp) != 0 :
                                link[k] = set(link[k]).union(current_neighbor)
                                np.array(link)
                                check = check + 1
                                #print(link)
                        if check == 0:
                            id = id +1
                            np.array(link)
                            link.append(set(current_neighbor))
    l.write('\n')


# second pass
for row in range(m):
    for column in range(n):
        for x in range(id):
            if (label[row, column] in link[x]) and label[row, column] !=0 :
                label[row, column] = min(link[x])
# update file
for row in range(m):
    for column in range(n):
        R.write(str(int(label[row, column])))
    R.write('\n')

plt.figure(figsize=(30,10))
plt.subplot(1,3,1), plt.title('gray')
plt.imshow(image , cmap='binary'), plt.axis('off')
plt.subplot(1,3,2), plt.title('label')
plt.imshow(label ) # 顯示圖片
plt.subplot(1,3,3), plt.title('original')
plt.imshow(image_org ,cmap = 'gray')

plt.axis('off') # 不顯示座標軸
plt.show()
