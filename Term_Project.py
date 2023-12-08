import cv2
import numpy as np

video = cv2.VideoCapture('fire_image_3.jpg')  # 영상 읽기
while True:
    src,frame = video.read()
    frame = cv2.resize(frame,(480,480)) # 영상 크기 설정
    frame = cv2.GaussianBlur(frame,(5,5),0) # 노이즈 제거
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) # 색 변환 BGR -> HSV
    lower = np.array([0,100,100])
    upper = np.array([40,255,255])
    mask = cv2.inRange(frame_hsv,lower,upper) # 불꽃 검출 마스크 설졍
    result = cv2.bitwise_and(frame,frame,mask=mask) # 불꽃 부분 검출을 위해 원 영상과 마스크 영상 비트 연산 수행

    src_bgr = cv2.cvtColor(result,cv2.COLOR_HSV2BGR) # 윤관선 검출을 위해 1차 적으로 HSV -> BGR로 변환
    src_gray = cv2.cvtColor(src_bgr,cv2.COLOR_BGR2GRAY) # 2차 적으로 BGR -> GRAYSCALE로 변환
    _,src_bin = cv2.threshold(src_gray,0,255,cv2.THRESH_OTSU) # 영상 이진화
    cnt,labels,stats,centroids = cv2.connectedComponentsWithStats(src_bin) # 윤곽선 검출을 위해 레이블링 함수 이용
    dst_bgr = cv2.cvtColor(src_gray,cv2.COLOR_GRAY2BGR) # 다시 원상태로 돌리기 위해 GRAYSCALE -> BGR로 변환
    dst_hsv = cv2.cvtColor(dst_bgr,cv2.COLOR_BGR2HSV) # 최종 BGR -> HSV로 변환

    for i in range(1,cnt): # 객체 정보를 위해 반복문 설정.
        (x,y,w,h,area) = stats[i]
        if area < 20:  # 노이즈 제거
            continue

        cv2.rectangle(dst_hsv,(x,y,w,h),(0,255,255)) # 최종 hsv 영상의 불꽃 주위에 노란색 사각형 생성

    # 오리지널 영상과 마스크, 최종 화재 검출 결과 실행
    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Result',dst_hsv)

    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
video.release()
