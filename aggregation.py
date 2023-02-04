import numpy as np
import cv2
import os


class Aggregator:
    def __init__(self, cls):
        self.cls = cls

    def aggregate(self, clip_aedat, start, end, size_kz=(480, 640)):
        image = np.zeros(size_kz) + 0.5
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

        # plt.imshow(thresh)

        return thresh

    @staticmethod
    def blobbing(image, min_threshold=1, max_threshold=255, filter_by_areas=True, minArea=1):
        # Detect blobs
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold

        # Filter by Area.
        params.filterByArea = filter_by_areas
        params.minArea = minArea

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)

        pts = cv2.KeyPoint_convert(keypoints)

        return pts

    @staticmethod
    def rescale_coords(x, w, y, h, source_res: tuple, target_res: tuple):
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
    def resize_and_append(agg_frame, mp4_frames, size: tuple):
        resized_mp4 = []
        # resized
        resized_agg = cv2.resize(agg_frame, size)
        # result = np.repeat(resized_agg[np.newaxis, ...], len(mp4_frames), axis=0)
        # print(len(mp4_frames))
        for frame in mp4_frames:
            # resized.append([resized_agg, cv2.resize(frame, size)])
            resized_mp4.append(cv2.resize(frame, size))

        return resized_agg, np.stack(resized_mp4)

    def find_and_rescale(self, clip_numpy_path, clip_mp4_path, aggreg_duration, time_diff,
                         size_kz=(480, 640), size_mp4=(1080, 1980)):
        """
        Agreguje, klatki z kz i szuka odpowiadających klatek z mp4
        :param clip_numpy_path: ścieżka klipu np
        :param clip_mp4_path:  ścieżka klipu mp4
        :param aggreg_duration: czas agregacji w sekundach
        :param time_diff: opóźnienie w sekundach
        :param size_kz: rozdzielczośc kz
        :param size_mp4: rozdzielczość mp4
        :return: Zwraca zbiory obrazów dla danego klipu
        """

        # outputy
        images_64_64_kz = []
        images_64_64_kz_y = []

        images_64_64_mp4 = []
        images_64_64_mp4_y = []

        images_32_32_kz = []
        images_32_32_kz_y = []

        images_32_32_mp4 = []
        images_32_32_mp4_y = []

        clip = np.load(clip_numpy_path)

        # #
        k = 1000000  # mikrosekund w sekundzie
        # #
        # images = {}
        # start klipu
        start = clip['timestamp'][0]

        # długośc klipu aedat4
        clip_length = clip['timestamp'][-1] - start

        # ile klatek można zagregować
        n_frames = int(clip_length / (aggreg_duration * k))
        print(f'..Do agregacji {n_frames} klatki')

        # pierwsza wartość końcowa
        end = start + aggreg_duration * k

        for i in range(n_frames):
            print(f'..Agregacja {i} klatki')
            image = self.aggregate(clip, start, end)
            print(f'..Processing {i} klatki')
            image = self.processing(image)

            print(f'..Blobbing {i} klatki')
            pts = self.blobbing(image)
            if len(pts) != 0:
                # ROI z kz i cropped image
                x, y, h, w = cv2.boundingRect(pts)
                cropped = image[y:y + h, x:x + w]
            else:
                print('Nie znaleziono blobów')
                continue

            print(f'..Skalowanie koordynat {i} klatki')
            # przeskalowane koordynaty
            scaled_x, scaled_w, scaled_y, scaled_h = self.rescale_coords(x, w, y, h, size_kz, size_mp4)

            print(f'..Wycinanie klatek z mp4 dla {i} klatki')
            # klatki z mp4 odpowiadające zagregowanym klatkom kz
            frames = self.cut_frames_from_mp4(clip_mp4_path, i * aggreg_duration + time_diff, aggreg_duration)

            # cropping klatek mp4 przeskalowanymi koordynatami
            frames = [frame[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w] for frame in frames]
            if len(frames) == 0:
                print('Brak klatek.')
                continue
            # print(len(frames))

            # RESIZE
            print(f'..Resize {i} klatki')
            resized_kz_64, images_mp4_64x64 = self.resize_and_append(cropped, frames, (64, 64))
            resized_kz_32, images_mp4_32x32 = self.resize_and_append(cropped, frames, (32, 32))
            #

            #### outputy

            # lista obrazków z kz 64x64
            images_64_64_kz.append(resized_kz_64)
            images_64_64_kz_y.append(i)

            # lista klatek z mp4 64x64
            images_64_64_mp4.append(images_mp4_64x64)
            for x in range(len(images_mp4_64x64)):
                images_64_64_mp4_y.append(i)

            # lista obrazków z kz 32x32
            images_32_32_kz.append(resized_kz_32)
            images_32_32_kz_y.append(i)

            # lista klatek z mp4 32x32
            images_32_32_mp4.append(images_mp4_32x32)
            for x in range(len(images_mp4_32x32)):
                images_32_32_mp4_y.append(i)

            ####

            # przejdź do nastepnych wycięć
            start = end
            end += aggreg_duration * k

        # v stack żeby uzyskać odpowiedni rozmiar
        images_64_64_mp4 = np.vstack(images_64_64_mp4)

        images_32_32_mp4 = np.vstack(images_32_32_mp4)

        return np.stack(images_64_64_kz), \
               np.stack(images_64_64_kz_y), \
               images_64_64_mp4, \
               np.stack(images_64_64_mp4_y), \
               np.stack(images_32_32_kz), \
               np.stack(images_32_32_kz_y), \
               images_32_32_mp4, \
               np.stack(images_32_32_mp4_y)


def search_directory(numpy_directory: str, mp4_directory: str, count_start, count_stop, aggregator: Aggregator,
                     agg_duration=1, time_diff=1):
    """

    :param numpy_directory: folder z numpyowskimi klipami
    :param mp4_directory: folder z klipami mp4
    :param aggregator: agregator użyty
    :param agg_duration: czas trwania agregacji z kz, domyślnie 1 sec
    :param time_diff: czas opóźnienia między kamerami, powinno być 0 sec ale czasami 1 sec potrzebna
    :return: zwraca 4 zbiory obrazów, zagregowane kz dla 64x64, 32x32 i klatki z klipów mp4 dla 64x64 i 32x32.
    Dodatkowo zwraca odpowiadające im zbiory y które określają przynależność do zagregowanych przedziałów dla klipów np.
    0 - przedział od 0 do 1 sekundy
    1 - przedział od 1 do 2 sekundy itd.
    """
    numpy_paths = []
    mp4_paths = []
    # tworzenie 4 zbiorów wraz z odpowiadającymi im czasami agregacji
    x_64_64_kz_data, y_64_64_kz_data, x_64_64_mp4_data, y_64_64_mp4_data, \
    x_32_32_kz_data, y_32_32_kz_data, x_32_32_mp4_data, y_32_32_mp4_data = [], [], [], [], [], [], [], []

    for filename in os.listdir(numpy_directory):
        file_path = os.path.join(numpy_directory, filename)
        numpy_paths.append(file_path)

    for filename in os.listdir(mp4_directory):
        file_path = os.path.join(mp4_directory, filename)
        mp4_paths.append(file_path)

    # n = len(numpy_paths)
    n = count_stop - count_start
    i = 0
    for np_path, mp4_path in zip(numpy_paths[count_start:count_stop], mp4_paths[count_start:count_stop]):
        print(f'Agregacja dla pliku {np_path} i {mp4_path}.\nJeszcze {n - i - 1} plików.. ')
        x_64_64_kz, y_64_64_kz, x_64_64_mp4, y_64_64_mp4, x_32_32_kz, y_32_32_kz, x_32_32_mp4, y_32_32_mp4 = \
            aggregator.find_and_rescale(np_path, mp4_path, agg_duration, time_diff)

        x_64_64_kz_data.append(x_64_64_kz)
        y_64_64_kz_data.append(y_64_64_kz)

        x_32_32_kz_data.append(x_32_32_kz)
        y_32_32_kz_data.append(y_32_32_kz)

        x_64_64_mp4_data.append(x_64_64_mp4)
        y_64_64_mp4_data.append(y_64_64_mp4)

        x_32_32_mp4_data.append(x_32_32_mp4)
        y_32_32_mp4_data.append(y_32_32_mp4)

        i += 1

    # konwersja do numpy array

    # stack
    x_64_64_kz_data = np.vstack(x_64_64_kz_data)
    y_64_64_kz_data = np.hstack(y_64_64_kz_data)

    x_32_32_kz_data = np.vstack(x_32_32_kz_data)
    y_32_32_kz_data = np.hstack(y_32_32_kz_data)

    x_64_64_mp4_data = np.vstack(x_64_64_mp4_data)
    y_64_64_mp4_data = np.hstack(y_64_64_mp4_data)

    x_32_32_mp4_data = np.vstack(x_32_32_mp4_data)
    y_32_32_mp4_data = np.hstack(y_32_32_mp4_data)

    return x_64_64_kz_data, \
           y_64_64_kz_data, \
           x_32_32_kz_data, \
           y_32_32_kz_data, \
           x_64_64_mp4_data, \
           y_64_64_mp4_data, \
           x_32_32_mp4_data, \
           y_32_32_mp4_data
