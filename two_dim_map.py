import cv2
import numpy as np
from configparser import ConfigParser
import os
import my_function


class TwoDimensionalMap:
    def __init__(self, name):
        self.name = name
        self.width = None
        self.height = None
        self.source = None
        self.original = None
        self.matrix = None
        self.present = []
        self.past = []
        self.double_past = []
        self.imaginary_line = []
        cv2.namedWindow(self.name)

    def setSource(self, src):
        self.original = src
        self.source = src
        self.width = src.shape[1]
        self.height = src.shape[0]
        # cv2.imshow(self.name, self.source)
        # cv2.waitKey(1)

    def setWH(self, width, height):
        self.width = width
        self.height = height
        self.original = np.ones(shape=(height, width, 3), dtype=np.uint8) * 255
        self.source = np.ones(shape=(height, width, 3), dtype=np.uint8) * 255
        # cv2.imshow(self.name, self.source)
        # cv2.waitKey(1)

    def setTransformation(self, src_points, dst_points):
        if np.array(src_points).shape != (4, 2):
            raise ValueError('Source Reference Points shape should be (4, 2)')
        if np.array(dst_points).shape != (4, 2):
            raise ValueError('Destination Reference Points shape should be (4, 2)')
        self.matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32),
                                                  np.array(dst_points, dtype=np.float32))

    def transformPoint(self, point):
        if type(point) is tuple and len(point) == 2:
            point = [point]
        elif type(point) is tuple and len(point) != 2:
            raise ValueError('Point should be tuple (x,y)')
        result = []
        for p in point:
            x_temp = (self.matrix[0, 0] * p[0] + self.matrix[0, 1] * p[1] + self.matrix[0, 2]) / (
                    self.matrix[2, 0] * p[0] + self.matrix[2, 1] * p[1] + self.matrix[2, 2])
            y_temp = (self.matrix[1, 0] * p[0] + self.matrix[1, 1] * p[1] + self.matrix[1, 2]) / (
                    self.matrix[2, 0] * p[0] + self.matrix[2, 1] * p[1] + self.matrix[2, 2])
            result.append((int(x_temp), int(y_temp)))
        return result

    def update(self, xywh, obj_id, obj_cls):
        if xywh is None and obj_id is None and obj_cls is None:
            return True
        if len(xywh) != len(obj_id) or len(xywh) != len(obj_cls):
            raise ValueError('Unmatching size!')
        # Check double ID
        double_id = bool([x for x in np.bincount(obj_id) if x > 1])
        if double_id:
            raise ValueError('Double ID!')
        temp = []
        iter_obj = iter(zip(xywh, obj_id, obj_cls))
        for (a, b, c) in iter_obj:
            temp.append([a, b, c])
        self.present = temp
        self._track()
        return True

    def show(self):
        cv2.imshow(self.name, self.source)
        if cv2.waitKey(1) == ord('q'):
            raise StopIteration

    def saveConfigMatrix(self, fname, path='config/'):
        config = ConfigParser()
        if os.path.isfile(os.path.join(path, config + '.ini')):
            config.read(os.path.join(path, fname))
            if not 'default' in config:
                config['default'] = {}
        temp = [str(x) for x in self.matrix.reshape(6)]
        config['default']['matrix'] = ' '.join(temp)
        with open(os.path.join(path, fname)) as configfile:
            config.write(configfile)
        return True

    def loadConfigMatrix(self, fname, path='config/'):
        config = ConfigParser()
        config.read(os.path.join(path, fname))
        temp = config.get('default', 'matrix').split(' ')
        temp = [int(x) for x in temp]
        self.matrix = np.array(temp).reshape((3, 3))
        return True

    def saveConfigLine(self, fname, path='config/'):
        config = ConfigParser()
        if os.path.isfile(os.path.join(path, config + '.ini')):
            config.read(os.path.join(path, fname))
            if not 'default' in config:
                config['default'] = {}
        # temp = [str(x) for x in self.matrix.reshape(6)]
        # config['default']['matrix'] = ' '.join(temp)
        # with open(os.path.join(path, fname)) as configfile:
        #     config.write(configfile)
        return True

    def loadConfigLine(self, fname, path='config/'):
        config = ConfigParser()
        config.read(os.path.join(path, fname))
        temp = config.get('default', 'line').split('\n')
        # temp = [int(x) for x in temp]
        # self.matrix = np.array(temp).reshape((3, 3))
        return True

    def drawCircle(self, point, color=(255, 255, 0), transformed=False):
        if transformed:
            new_point = point
        else:
            new_point = self.transformPoint(point)
        cv2.circle(self.source, new_point[0], 5, color, -1)

    def _track(self):
        # for search
        past_index_to_remove = []
        for num, i in enumerate(np.array(self.present)[:, 1]):
            if i in np.array(self.past)[:, 1]:
                idx = list(np.array(self.past)[:, 1]).index(i)
                present_point = self.present[num][0]
                past_point = self.past[idx][0]
                past_index_to_remove.append(idx)
                # Draw line?
                # Check if the present point is in the certain box
                self._check_intersection([present_point, past_point])
            elif i in np.array(self.double_past)[:, 1]:
                idx = list(np.array(self.double_past)[:, 1]).index(i)
                present_point = self.present[num][0]
                past_point = self.double_past[idx][0]
                self._check_intersection([present_point, past_point])
        for i in past_index_to_remove:
            self.past.pop(i)
        self.double_past = self.past
        self.past = self.present
        # for search - end

    def _check_intersection(self, line):
        if self.imaginary_line:
            for num, i in enumerate(self.imaginary_line):
                # Check if intersect
                # if True, then increase counter for imaginary_line idx and BREAK
                None
