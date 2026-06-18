import cv2
import numpy as np
import imutils
from collections import OrderedDict
from scipy.spatial import distance as dist

# ---------------------------
# Simple Centroid Tracker
# ---------------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=40, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# ---------------------------
# Main
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    ct = CentroidTracker(maxDisappeared=40, maxDistance=60)
    totalCount = 0
    trackable = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=800)
        H, W = frame.shape[:2]

        fgMask = backSub.apply(frame)
        _, fgMask = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        rects = []

        for c in cnts:
            if cv2.contourArea(c) < 600:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            rects.append((x, y, x + w, y + h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        objects = ct.update(rects)

        lineY = int(H * 0.6)
        cv2.line(frame, (0, lineY), (W, lineY), (0, 0, 255), 2)

        for (objectID, centroid) in objects.items():
            if objectID not in trackable:
                trackable[objectID] = {"counted": False, "y": centroid[1]}

            prevY = trackable[objectID]["y"]
            trackable[objectID]["y"] = centroid[1]

            if not trackable[objectID]["counted"]:
                if prevY < lineY and centroid[1] >= lineY:  # crossed line
                    totalCount += 1
                    trackable[objectID]["counted"] = True
                    print(f"Vehicle #{totalCount} counted")

            cv2.putText(frame, f"ID {objectID}", (centroid[0]-10, centroid[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2.putText(frame, f"Total Vehicles: {totalCount}", (10, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Live Vehicle Counter", frame)
        cv2.imshow("Mask", fgMask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
