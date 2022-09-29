import cv2 as cv
import numpy as np


hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

def Angle(x):           #시야각 +우 -좌
    Aov = np.pi / 4     #화각설정
    Aow = 650           #너비
    '''
    Distance = (Aow / 2) / np.tan(Aov)  # 거리
    Atan = (x - (Aow / 2)) / Distance  # 탄젠트값
    Radian = np.arctan(Atan)  # 라디안값
    Ang = Radian * 180 / np.pi  # 각도변환
    '''
    Ang = np.arctan((x-(Aow/2))/(Aow/2)/np.tan(Aov)) * 180 / np.pi
    return (Ang)
def Sumx(x):
    sx.append(x)
    if len(sx) > 10:    #값 조정
        sx.pop(0)
    return (sum(sx) // len(sx))
def Sumy(y):
    sy.append(y)
    if len(sy) > 10:    #값 조정
        sy.pop(0)
    return (sum(sy) // len(sy))


def nothing(x):
    pass
def mouse_callback(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3, threshold, H_
    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.

    if event == cv.EVENT_LBUTTONDOWN:
        print(img_color[y, x])
        color = img_color[y, x]
        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]


        threshold = cv.getTrackbarPos('threshold', 'img_result')
        H_ = cv.getTrackbarPos('H', 'img_result')

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv[0] < 10:
            print("case1")
            lower_blue1 = np.array([hsv[0]-H_+180, threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], threshold, threshold])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0]+H_-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-H_, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([hsv[0]+H_, 255, 255])
            lower_blue2 = np.array([hsv[0]-H_, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-H_, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0])
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)


            #     print(i, i+10)
            #     print(i-10, i)

        #print(hsv[0])
        #print("@1", lower_blue1, "~", upper_blue1)
        #print("@2", lower_blue2, "~", upper_blue2)
        #print("@3", lower_blue3, "~", upper_blue3)


cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)

cv.namedWindow('img_result')
cv.createTrackbar('threshold', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('threshold', 'img_result', 50)
cv.createTrackbar('H', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('H', 'img_result', 10)

cap = cv.VideoCapture(0)

global centerX, centerY
centerX=0
centerY=0
sx=[0]
sy=[0]
while(True):
    #img_color = cv.imread('2.jpg')
    ret, img_color = cap.read()
    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3


    kernel = np.ones((11,11), np.uint8)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)
    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)

    num0fLabels, img_label, stats, centriod = cv.connectedComponentsWithStats(img_mask)
    centerX=0 #값 초기화
    centerY=0
    for idx, centriod in enumerate(centriod):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue
        if np.any(np.isnan(centriod)):
            continue
        x,y,width,height,area = stats[idx]
        centerX, centerY = int(centriod[0]), int(centriod[1])
        #print(centerX, "\t", centerY, "\t", Angle(centerX))    #중앙값 출력
        if area > 10: #테스트하면서 조정
            cv.circle(img_result, (centerX, centerY), 10, (0,0,255), 10)
            cv.rectangle(img_result, (x,y), (x+width, y+height), (0,0,255))
            cv.circle(img_result, (Sumx(centerX), Sumy(centerY)), 10, (0, 255, 0), 10)
            print(Sumx(centerX), "\t", Sumy(centerY), "\t", Angle(Sumx(centerX)))  # Sum출력
        '''
        sumx.append(centerX)
        if len(sumx) > 10:
            sumx.pop(0)
        summx = sum(sumx) // len(sumx)
        sumy.append(centerY)
        if len(sumy) > 10:
            sumy.pop(0)
        summy = sum(sumy) // len(sumy)
        '''


    '''
    
    sumx.append(centerX)
    if len(sumx)>10:
        sumx.pop(0)
    summx = sum(sumx) // len(sumx)
    sumy.append(centerY)
    if len(sumy) > 10:
        sumy.pop(0)
    summy = sum(sumy) // len(sumy)
    print(summx, "\t", summy,"\t",Angle(summx))
    cv.circle(img_result, (summx, summy), 10, (0, 255, 0), 10)
    '''
    #print(centerX, "\t", centerY, "\t", Angle(centerX))         #중앙값 출력
    cv.imshow('img_color', img_color)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)



    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break


cv.destroyAllWindows()
'''
중앙값+각도 넘기기         선배
다수가 인식되면        논의 필요          2
인식 실패시 값?       논의 필요           1
무시하는 사이즈        테스트 필요         2
조도에 따른 색깔       테스트 필요         HSV로 어느정도 해결
인식되는 값으로 색깔값 갱신     논의 필요   1
X좌표 값에 따른 각도                     1
최적의 threshold값    테스트 필요        2
650
'''
'''
Aov=np.pi/4                     #화각
Aow=650                         #너비
Distance=(Aow/2)/np.tan(Aov)    #거리
Atan=(centerX-(Aow/2))/Distance #탄젠트값
Radian=np.arctan(Atan)          #라디안값
Ang=Radian*180/np.pi            #각도변환
print(Radian,"\t",Ang)
'''