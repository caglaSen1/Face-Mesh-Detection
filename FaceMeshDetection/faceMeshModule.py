import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils  # help us draw on our faces
        self.mpFaceMesh = mp.solutions.face_mesh
        #self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,
                                 min_detection_confidence=self.minDetectionCon,
                                 min_tracking_confidence=self.minTrackCon)

        # specifications
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, frame, draw=True):
        # Define face connections manually

        '''
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Jawline
            (5, 6), (6, 7), (7, 8), (8, 9),  # Left eyebrow
            (10, 11), (11, 12), (12, 13), (13, 14),  # Right eyebrow
            (15, 16), (17, 18), (19, 20),  # Nose
            (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),  # Left eye
            (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34),  # Right eye
            (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45),
            (45, 46), (46, 47),  # Outer lips
            (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58),
            (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67)  # Inner lips
        ]
        '''

        self.frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.frameRGB)
        faces = []
        if self.results.multi_face_landmarks:  # if sth is detected
            for faceLms in self.results.multi_face_landmarks:  # faceLms - landmarks of one face
                if draw:
                    self.mpDraw.draw_landmarks(frame, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                      self.drawSpec, self.drawSpec)  # img, landmark list, connections, landmark drawing spec, connection drawing spec
                face = []
                for id, lm in enumerate(faceLms.landmark):  # lm - each of the landmark of landmarks of one face
                    # print(lm)  # lm has x, y and z position
                    h, w, ch = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)  # we have our values in term of pixels
                    print(id, x, y)
                    face.append([x, y])
                faces.append(face)

        return frame, faces


def main():
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # writer - a frame repository
    writer = cv2.VideoWriter("video_kaydı.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
    # "video_record.mp4" --> name of the video to record
    # cv2.VideoWriter_fourcc(*"mp4v") --> 4-character codec used to compress frames
    # 20 --> frame per second
    # (width, height) --> size of the video recorder

    detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

    pTime = 0  # previous time

    while True:
        success, frame = cap.read()
        frame, faces = detector.findFaceMesh(frame, False)
        # if len(faces) != 0:
            # print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # cTime - current time
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)  # Here, (10, 50) represents the (x, y) coordinates where the text will be placed on the frame.

        cv2.imshow("Frame", frame)
        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()