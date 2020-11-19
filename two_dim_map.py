import cv2
import numpy as np
from configparser import ConfigParser
import os
import my_function
import time
import requests
import json


do_post_http = False


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
        self.imaginary_line_color = (0, 255, 255)
        self.cross_counter = None
        self.sql = None
        cv2.namedWindow(self.name)

    def setSQL(self, host, database, user, password, table):
        self.sql = my_function.MySQL(host, database, user, password, table)

    def closeSQL(self):
        self.sql.close()

    def setSource(self, src):
        self.original = src
        self.source = np.copy(src)
        self.width = src.shape[1]
        self.height = src.shape[0]
        if len(self.imaginary_line):
            for l in self.imaginary_line:
                cv2.line(self.source, (l[0], l[1]), (l[2], l[3]), self.imaginary_line_color, 3)
        # cv2.imshow(self.name, self.source)
        # cv2.waitKey(1)

    def setWH(self, width, height):
        self.width = width
        self.height = height
        self.original = np.ones(shape=(height, width, 3), dtype=np.uint8) * 255
        self.source = np.ones(shape=(height, width, 3), dtype=np.uint8) * 255
        if len(self.imaginary_line):
            for l in self.imaginary_line:
                cv2.line(self.source, (l[0], l[1]), (l[2], l[3]), self.imaginary_line_color, 3)
        # cv2.imshow(self.name, self.source)
        # cv2.waitKey(1)

    def setTransformation(self, src_points, dst_points):
        if np.array(src_points).shape != (4, 2):
            raise ValueError('Source Reference Points shape should be (4, 2)')
        if np.array(dst_points).shape != (4, 2):
            raise ValueError('Destination Reference Points shape should be (4, 2)')
        self.matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32),
                                                  np.array(dst_points, dtype=np.float32))
        self.matrix = np.array(self.matrix)

    def setLine(self, point1, point2):
        if type(point1) != tuple or type(point2) != tuple:
            raise ValueError('point type must be tuple.')
        elif len(point1) != 2 or len(point2) != 2:
            raise ValueError('point must be (x, y).')
        self.imaginary_line.append([point1[0], point1[1], point2[0], point2[1]])

    def transformPoint(self, point):
        if type(point) is tuple and len(point) == 2:
            point = [point]
        elif type(point) is tuple and len(point) != 2:
            raise ValueError('Point should be tuple (x,y)')
        if len(self.matrix) == 0:
            raise ValueError('Transformation matrix is empty')
        elif self.matrix.shape != (3, 3):
            raise ValueError('Matrix size error! Current size: ' + str(self.matrix.shape))
        result = []
        for p in point:
            x_temp = (self.matrix[0, 0] * p[0] + self.matrix[0, 1] * p[1] + self.matrix[0, 2]) / (
                    self.matrix[2, 0] * p[0] + self.matrix[2, 1] * p[1] + self.matrix[2, 2])
            y_temp = (self.matrix[1, 0] * p[0] + self.matrix[1, 1] * p[1] + self.matrix[1, 2]) / (
                    self.matrix[2, 0] * p[0] + self.matrix[2, 1] * p[1] + self.matrix[2, 2])
            result.append((int(x_temp), int(y_temp)))
        return result

    def update(self, xyxy, obj_id, obj_cls):
        if xyxy is None and obj_id is None and obj_cls is None:
            return True
        if len(xyxy) != len(obj_id) or len(xyxy) != len(obj_cls):
            raise ValueError('Unmatching size!')
        # Check double ID
        double_id = bool([x for x in np.bincount(obj_id) if x > 1])
        if double_id:
            raise ValueError('Double ID!')
        temp = []
        xywh = my_function.xyxy_to_xywh_v2(xyxy)
        iter_obj = iter(zip(xywh, obj_id, obj_cls))
        for (a, b, c) in iter_obj:
            temp.append([a, b, c])
        if self.cross_counter is None:
            self.cross_counter = np.zeros((len(self.imaginary_line), 8), dtype=np.int)
        self.present = temp
        self._track()
        return True

    def show(self):
        cv2.imshow(self.name, self.source)
        # if cv2.waitKey(1) == ord('q'):
        #     None  # raise StopIteration

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
        self.imaginary_line = [int(x.split(' ')) for x in temp]
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
        if len(self.past):
            # print('Present ID: ', np.array(self.present)[:, 1])
            # print('Past ID: ', np.array(self.past)[:, 1])
            for num, i in enumerate(np.array(self.present)[:, 1]):  # For every present objects
                if i in np.array(self.past)[:, 1]:  # If exist in past objets
                    idx = list(np.array(self.past)[:, 1]).index(i)
                    present_point = (self.present[num][0][0], self.present[num][0][1])
                    past_point = (self.past[idx][0][0], self.past[idx][0][1])
                    past_index_to_remove.append(idx)
                    # Draw line?
                    cv2.line(self.source, present_point, past_point, (0, 0, 255), 3)
                    # Check if the present point is in the certain box
                    res, n = self._check_intersection([past_point, present_point])
                    if res:
                        print('Intersect Detected: ' + str(n) + ' with ID: ' + str(i))
                        # todo: update log (timestamp, imaginary line number, vehicle type)
                        log = {'timestamps': time.time(),
                               'imaginary_line': int(n),
                               'object_id': int(i),
                               'object_class': int(self.present[num][2])
                               }
                        headers = {'Content-Type': 'application/json'}
                        if do_post_http:
                            response = requests.post('http://192.168.0.143:5000/api/traffic', data=json.dumps(log),
                                                     headers=headers)
                            print('post success. ', response)
                        if self.sql is not None:
                            self.sql.insert_crossing(time.time(), n, i, self.present[num][2])
                        cv2.line(self.source, (self.imaginary_line[n][0], self.imaginary_line[n][1]),
                                 (self.imaginary_line[n][2], self.imaginary_line[n][3]), (0, 0, 255), 3)
                        self.cross_counter[n, self.present[num][2]] += 1
                        # print('Line: ', self.imaginary_line[n], 'Point: ', past_point, present_point)
                        # while cv2.waitKey() != ord('r'):
                        #     None
                elif len(self.double_past) and i in np.array(self.double_past)[:, 1]:
                    idx = list(np.array(self.double_past)[:, 1]).index(i)
                    present_point = (self.present[num][0][0], self.present[num][0][1])
                    past_point = (self.double_past[idx][0][0], self.double_past[idx][0][1])
                    cv2.line(self.source, present_point, past_point, (0, 0, 255), 3)
                    res, n = self._check_intersection([past_point, present_point])
                    if res:
                        print('Intersect Detected: ' + str(n) + ' with ID: ' + str(i))
                        # todo: update log (timestamp, imaginary line number, vehicle type)
                        if self.sql is not None:
                            self.sql.insert_crossing(time.time(), n, i, self.present[num][2])
                        cv2.line(self.source, (self.imaginary_line[n][0], self.imaginary_line[n][1]),
                                 (self.imaginary_line[n][2], self.imaginary_line[n][3]), (0, 0, 255), 3)
                        self.cross_counter[n, self.present[num][2]] += 1
                        # while cv2.waitKey() != ord('r'):
                        #     None
        for i in sorted(past_index_to_remove, reverse=True):
            self.past.pop(i)
        self.double_past = self.past
        self.past = self.present
        # for search - end

    def _check_intersection2(self, line):
        if self.imaginary_line:
            intersection = []
            for num, i in enumerate(self.imaginary_line):
                # Check if intersect
                # if True, then increase counter for imaginary_line idx and BREAK
                # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
                p = np.array([i[0], i[1]])
                r = np.array([i[2] - i[0], i[3] - i[1]])
                q = np.array(line[0])
                s = np.array(line[1] - line[0])
                rxs = np.cross(r, s)
                if rxs == 0:
                    intersection.append(False)  # Collinear or Parallel
                else:
                    t = np.cross((q - p), s) / rxs
                    u = np.cross((q - p), r) / rxs
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection.append(True)  # Intersect (two line segments meet at the point p+tr = q+us
                    else:
                        intersection.append(
                            False)  # Do not intersect (two line segments are not parallel but do not intersect)
            if len(intersection) == len(self.imaginary_line):
                return intersection
            else:
                raise ValueError('check intersection failed.')
        else:
            raise ValueError('no imaginary line detected.')

    def _check_intersection(self, line):
        if self.imaginary_line:
            intersection = []
            for num, i in enumerate(self.imaginary_line):
                # Check if intersect
                # if True, then increase counter for imaginary_line idx and BREAK
                # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
                p = np.array([i[0], i[1]])
                r = np.array([i[2], i[3]]) - np.array([i[0], i[1]])
                q = np.array(line[0])
                s = np.array(line[1]) - np.array(line[0])
                rxs = np.cross(r, s)
                if rxs == 0:
                    continue  # Collinear or Parallel
                else:
                    t = np.cross((q - p), s) / rxs
                    u = np.cross((q - p), r) / rxs
                    if 0 < t <= 1 and 0 < u <= 1:
                        return True, num  # Intersect (two line segments meet at the point p+tr = q+us
                    else:
                        continue  # Do not intersect (two line segments are not parallel but do not intersect)
            return False, -1
        else:
            raise ValueError('no imaginary line detected.')
