import cv2
import numpy as np

def task1():
    image = cv2.imread("images/variant-4.jpeg")  # Замените 4 на ваш номер варианта
    blue_channel = image[:, :, 0]
    cv2.imshow("Blue Channel", blue_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task2():
    cap = cv2.VideoCapture("sample.mp4")
    while cv2.waitKey(1) != 27:  # Esc для выхода
        ret, frame = cap.read()
        if not ret:
            break
        
        # Извлекаем только черный канал
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Преобразуем в бинарное изображение для поиска контуров
        ret, thresh = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
             
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x, center_y = x + w // 2, y + h // 2
            
            # Проверка, попадает ли метка в правую половину экрана
            if center_x > frame.shape[1] // 2:
                cv2.putText(frame, "Right Half", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Left Half", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow("Frame", frame)
    
    cap.release()
    cv2.destroyAllWindows()

def additional_task():
    fly = cv2.imread("fly64.png")
    fly_height, fly_width = fly.shape[:2]
    
    cap = cv2.VideoCapture("sample.mp4")
    while cv2.waitKey(1) != 27:  # Esc для выхода
        ret, frame = cap.read()
        if not ret:
            break
        
       # Извлекаем только черный канал
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Преобразуем в бинарное изображение для поиска контуров
        ret, thresh = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x, center_y = x + w // 2, y + h // 2
            
            # Наложение мухи на кадр
            top_left_x = center_x - fly_width // 2
            top_left_y = center_y - fly_height // 2
            
            for i in range(fly_height):
                for j in range(fly_width):
                    if fly[i, j].all() != 0:  # Проверка на черный цвет (фон мухи)
                        frame[top_left_y + i, top_left_x + j] = fly[i, j]
        
        cv2.imshow("Frame with Fly", frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    task1()
    task2()
    additional_task()
