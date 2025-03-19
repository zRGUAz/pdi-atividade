import cv2
import numpy as np

cap = cv2.VideoCapture("q1/q1A.mp4")

def detect_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Intervalos de cores ajustados para melhor detecção
    color_ranges = {
        "red": [(np.array([0, 120, 70]), np.array([10, 255, 255])), (np.array([170, 120, 70]), np.array([180, 255, 255]))],
        "blue": [(np.array([90, 80, 50]), np.array([130, 255, 255]))],
        "green": [(np.array([40, 40, 40]), np.array([80, 255, 255]))]
    }
    
    detected_shapes = []
    
    for color, ranges in color_ranges.items():
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Remove ruídos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Evita ruídos
                x, y, w, h = cv2.boundingRect(cnt)
                detected_shapes.append((color, (x, y, w, h)))
    
    return detected_shapes

def find_largest_shape(shapes):
    largest_shape = None
    max_area = 0
    
    for shape in shapes:
        _, (x, y, w, h) = shape
        area = w * h
        if area > max_area:
            max_area = area
            largest_shape = shape
    
    return largest_shape

def check_collision(shapes):
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            _, (x1, y1, w1, h1) = shapes[i]
            _, (x2, y2, w2, h2) = shapes[j]
            
            # Verifica interseção entre bounding boxes
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                return True
    return False

def check_overlap(largest, shapes):
    if largest:
        _, (lx, ly, lw, lh) = largest
        for _, (x, y, w, h) in shapes:
            if (lx != x or ly != y) and not (lx + lw <= x or lx >= x + w or ly + lh <= y or ly >= y + h):
                return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    shapes = detect_shapes(frame)
    largest_shape = find_largest_shape(shapes)
    collision_detected = check_collision(shapes)
    overlap_detected = check_overlap(largest_shape, shapes)
    
    # Desenhar bounding boxes sem o texto
    for color, (x, y, w, h) in shapes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # Destacar maior forma
    if largest_shape:
        _, (x, y, w, h) = largest_shape
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Mostrar mensagens
    if collision_detected:
        cv2.putText(frame, "COLISAO DETECTADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    if overlap_detected:
        cv2.putText(frame, "ULTRAPASSAGEM DETECTADA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
    # Exibe resultado
    cv2.imshow("Feed", frame)
    
    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()
