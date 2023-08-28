#!/usr/bin/env
# -*- coding:utf-8 -*-

from __future__ import division
# import json
import math
from json import encoder
import numpy as np
import pandas as pd

encoder.FLOAT_REPR = lambda o: format(o, '.2f')


class base_station(object):
    def __init__(self, lat, lon, dist):
        self.lat = lat
        self.lon = lon
        self.dist = dist


class point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def print(self):
        print(f"({self.x}, {self.y})")


class circle(object):
    def __init__(self, point, radius):
        self.center = point
        self.radius = radius

    def print(self):
        print(f"radius: {self.radius}, center: ", end='')
        self.center.print()


class json_data(object):
    def __init__(self, circles, inner_points, center):
        self.circles = circles
        self.inner_points = inner_points
        self.center = center


def serialize_instance(obj):
    d = {}
    d.update(vars(obj))
    return d


def get_two_points_distance(p1, p2):
    return math.sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2))


def get_two_circles_intersecting_points(c1, c2):
    p1 = c1.center
    p2 = c2.center
    r1 = c1.radius
    r2 = c2.radius

    d = get_two_points_distance(p1, p2)
    # if to far away, or self contained - can't be done
    if d >= (r1 + r2) or d <= math.fabs(r1 - r2):
        return None

    a = (pow(r1, 2) - pow(r2, 2) + pow(d, 2)) / (2 * d)
    h = math.sqrt(pow(r1, 2) - pow(a, 2))
    x0 = p1.x + a * (p2.x - p1.x) / d
    y0 = p1.y + a * (p2.y - p1.y) / d
    rx = -(p2.y - p1.y) * (h / d)
    ry = -(p2.x - p1.x) * (h / d)
    return [point(x0 + rx, y0 - ry), point(x0 - rx, y0 + ry)]


def get_all_intersecting_points(circles):
    points = []
    num = len(circles)
    for i in range(num):
        j = i + 1
        for k in range(j, num):
            res = get_two_circles_intersecting_points(circles[i], circles[k])
            if res:
                points.extend(res)
    return points


def is_contained_in_circles(point, circles):
    for i in range(len(circles)):
        if (get_two_points_distance(point, circles[i].center) > (circles[i].radius)):
            return False
    return True


def get_polygon_center(points):
    center = point(0, 0)
    num = len(points)
    for i in range(num):
        center.x += points[i].x
        center.y += points[i].y
    center.x /= num
    center.y /= num
    return center


def exe_trilateration(distance_row):
    # point_arr = [point(0.0, 2.0), point(0.0, 1.0), point(0.0, 0.0), point(1.5, 0.0),
    #             point(3.0, 0.0), point(3.0, 1.0), point(3.0, 2.0), point(1.5, 2.0)]

    point_arr = [point(0.0, 2.0), point(0.0, 0.0), point(3.0, 0.0), point(3.0, 2.0)]
    # point_arr = [point(0.0, 2.0), point(0.0, 0.0), point(3.0, 0.0)]


    circle_arr = [circle(point_arr[i], distance_row[2*i]) for i in range(4)]

    # for i in range(8):
    #     circle_arr[i].print()

    inner_points = []
    for p in get_all_intersecting_points(circle_arr):
        if is_contained_in_circles(p, circle_arr):
            inner_points.append(p)

    # for p in inner_points:
    #     p.print()
    # print()
    center = get_polygon_center(inner_points)

    # print("location: ", end='')
    # center.print()

    return center.x, center.y


# if __name__ == '__main__':

    # p1 = point(0.81, 1.2)
    # p2 = point(1.21, 0.69)
    # p3 = point(0.87, 0.84)
    #
    # c1 = circle(p1, 0.70)
    # c2 = circle(p2, 0.51)
    # c3 = circle(p3, 0.63)

    # p1 = point(0.0, 0.0)
    # p2 = point(1.0, 0.0)
    # p3 = point(0.0, 1.0)
    # p4 = point(1.0, 1.0)
    #
    # c1 = circle(p1, 0.90)
    # c2 = circle(p2, 0.7)
    # c3 = circle(p3, 0.83)
    # c4 = circle(p4, 0.54)
    #
    # circle_list = [c1, c2, c3, c4]
    #
    # inner_points = []
    # for p in get_all_intersecting_points(circle_list):
    #     if is_contained_in_circles(p, circle_list):
    #         inner_points.append(p)
    #
    # center = get_polygon_center(inner_points)
    # # in_json = json_data([c1, c2, c3], [p1, p2, p3], center)
    #
    # # out_json = json.dumps(in_json, sort_keys=True,
    # #                       indent=4, default=serialize_instance)
    # #
    # # with open("data.json", 'w') as fw:
    # #     fw.write(out_json)
    #
    # c1.print()
    # c2.print()
    # c3.print()
    # c4.print()
    #
    # print("location: ", end='')
    # center.print()
