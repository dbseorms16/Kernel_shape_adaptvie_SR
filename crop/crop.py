from glob import glob
import cv2

imgs = glob('./*')
for i in imgs:
    if 'crop' in i:
        continue
    img = cv2.imread(i)
    # print(img.shape)
    # w, h, c = img.shape
    
    # new = cv2.resize(img, (w//4, h//4), cv2.INTER_CUBIC)
    # print(img.shape)
    
    # new = cv2.resize(new, (w, h), cv2.INTER_CUBIC)
    # print(img.shape)
    
    if '108005' in i:
        print(i)
        x = 300
        a = 100
        y = 100
        b = 100
        new = img[y:y+a, x:x+b]
        w, h, c = new.shape
        new = cv2.resize(new, (w*2, h*2), cv2.INTER_NEAREST)
        s = i.split('-')
        print(s)
        print(new.shape)
        cv2.imwrite(s[0]+s[1]+'crop.png', new)
    
    if '002' in i:
        x = 600
        a = 100
        y = 100
        b = 100
        print(new.shape)
        new = img[y:y+a, x:x+b]
        w, h, c = new.shape
        new = cv2.resize(new, (w*2, h*2), cv2.INTER_NEAREST)
        s = i.split('-')
        cv2.imwrite(s[0]+s[1]+'crop.png', new)
        