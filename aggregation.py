import numpy as np
import cv2


class Aggregator:
    def __init__(self):
        pass

    def aggregate(self, clip_aedat, start, end, size_kz= (480, 640)):
        image = np.zeros(size_kz)
        agg = clip_aedat[clip_aedat['timestamp'] < end]
        agg = agg[agg['timestamp'] >= start]

        # agregacja
        for x, y, z in zip(agg['x'], agg['y'], agg['polarity']):
            if z == 1:
                image[y][x] += 1
            elif z == 0:
                image[y][x] -= 1

        return image

    def processing(self, image):
        # normalizacja i filtrowanie
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = (np.abs(image - 0.5) * 255).astype(np.uint8)
        img = cv2.filter2D(img, -1, (1.0 / kernel.sum()) * kernel)

        # filtr medianowy
        img = cv2.medianBlur(img, 9)

        # morfologie
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # dylatacja
        img = cv2.dilate(img, kernel)

        # Apply thresholding
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        return thresh

    @staticmethod
    def blobbing(image, min_threshold = 1, max_threshold = 255, filter_by_ares = True, minArea = 1):
        # Detect blobs
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold

        # Filter by Area.
        params.filterByArea = filter_by_ares
        params.minArea = minArea

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)

        pts = cv2.KeyPoint_convert(keypoints)

        return pts

    @staticmethod
    def rescale_coords(x, w, y, h, source_res:tuple, target_res:tuple):
        """

        :param x: coord x
        :param w: szerokość
        :param y: coord y
        :param h: wysokośc
        :param source_res:
        :param target_res:
        :return: scaled_x , scaled_w, scaled_y, scaled_h
        """
        # 480 X 640; 1080 x 1980

        # przeskalowanie współrzędnych do współrzednych filmu mp4
        scaled_x = int(x * target_res[1] / source_res[1])
        scaled_y = int(y * target_res[0] / source_res[0])

        scaled_h = int(h * target_res[0] / source_res[0])
        scaled_w = int(w * target_res[1] / source_res[1])

        return scaled_x, scaled_w, scaled_y, scaled_h

    @staticmethod
    def cut_frames_from_mp4(clip_mp4_path, start_time, t_sec):
        """

        :param clip_mp4_path: ścieżka do klipu mp4
        :param start_time: sekunda rozpoczęcia wycinki klatek
        :param t_sec: czas trwania wycinka
        :return:
        """
        video = cv2.VideoCapture(clip_mp4_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        # Calculate the desired start and end frame numbers
        start_frame_number = int(start_time * fps)
        end_frame_number = start_frame_number + int(fps * t_sec)

        # Go to the start frame
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

        # Extract the frames
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if video.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame_number:
                break
            frames.append(frame)

        # Release the video

        video.release()

        return frames

    @staticmethod
    def resize_and_append(agg_frame, mp4_frames, size:tuple):
        resized = []
        # resized
        resized_agg = cv2.resize(agg_frame, size)
        for frame in mp4_frames:
            resized.append([resized_agg, cv2.resize(frame, size)])

        return resized

    def find_and_rescale(self, clip_aedat_path, clip_mp4_path, aggreg_duration, time_diff,
                         size_kz=(480, 640), size_mp4=(1080, 1980)):

        output = {}
        clip_aedat = np.load(clip_aedat_path)

        # #
        k = 1000000 # mikrosekund w sekundzie
        # #
        images = {}
        # start klipu
        start = clip_aedat['timestamp'][0]

        # długośc klipu aedat4
        clip_length = clip_aedat['timestmap'][-1] - start

        # ile klatek można zagregować
        n_frames = clip_length // aggreg_duration * k

        # pierwsza wartość końcowa
        end = start + aggreg_duration * k

        for i in range(n_frames):
            image = self.aggregate(clip_aedat, start, end)

            image = self.processing(image)

            pts = self.blobbing(image)

            # ROI i cropped image
            x, y, h, w = cv2.boundingRect(pts)
            cropped = image[y:y + h, x:x + w]

            # przeskalowane koordynaty
            scaled_x, scaled_w, scaled_y, scaled_h = self.rescale_coords(x,w,y,h, size_kz, size_mp4)

            # dodaj do listy wycięcie z kz
            # images.append(cropped)

            # klatki z mp4 odpowiadające zagregowanym klatkom kz
            frames = self.cut_frames_from_mp4(clip_mp4_path, (i+1) * aggreg_duration + time_diff, aggreg_duration)

            # cropping
            frames = [frame[scaled_y:scaled_y+scaled_h, scaled_x:scaled_h+scaled_w] for frame in frames]

            images_64x64 = self.resize_and_append(cropped, frames, (64, 64))
            images_32x32 = self.resize_and_append(cropped, frames, (32, 32))

            images[cropped] = [images_64x64, images_32x32]
            ####

            # przejdź do nastepnych wycięć
            start = end
            end += aggreg_duration * k

        return images


