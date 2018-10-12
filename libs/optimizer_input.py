# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd  # dataframe
import shapely.geometry as slg  # geometry
import numpy as np


class Infos:
    def __init__(self, input_floor_plan_dict, input_setup_dict, output_file_path, save_output, save_cache, save_log, settings):
        self.output_repo = output_file_path
        self.input_floor_plan_dict = input_floor_plan_dict
        self.input_setup_dict = input_setup_dict
        self.floor_plan = FloorPlan(input_floor_plan_dict, input_setup_dict, settings)
        self.save_output = save_output
        self.save_cache = save_cache
        self.save_log = save_log


class FloorPlan:

    def __init__(self, input_floor_plan_dict, input_setup_dict, settings):

        # load json plan with rooms, fixed items, load-bearing walls...
        self.load_dicts(input_floor_plan_dict, input_setup_dict)

        self.vertex_name = range(len(self.points))
        self.Vertex = pd.DataFrame(self.points, index=self.vertex_name, columns=['X', 'Y'])
        self.walls_nbr = len(self.externalWalls)
        self.plan_polygon = self.plan
        self.plan_area = self.plan.area

        print('Plan Area', self.plan_area)

        # Definition of fixed_items data frame
        # ['D1', 'FD', 'D2', 'W1', 'W2', 'W3', 'W4', 'W5']
        self.fi_name = self.give_name_to_fixed_items(self.fi_types)

        # External spaces
        print('externalSpaces_windows', self.externalSpaces_windows)
        externalSpaces_windowsNames = []
        for i, windowlist in enumerate(self.externalSpaces_windows):
            externalSpaces_windowsNames.append([])
            for j, windowNumber in enumerate(windowlist):
                externalSpaces_windowsNames[i].append(self.fi_name[windowNumber])
        print('externalSpaces_windowsNames', externalSpaces_windowsNames)

        # Fixed items
        self.fi_nbr = len(self.fi_types)
        self.fixed_items = pd.DataFrame(self.fi_types, index=self.fi_name, columns=['Type'])
        self.fixed_items['FirstName'] = self.fi_name
        self.fixed_items['Vertex1'] = self.fi_vertex1  # plan vertex
        self.fixed_items['Vertex2'] = self.fi_vertex2  # plan vertex
        self.fixed_items['coef1'] = self.fi_coef1
        self.fixed_items['coef2'] = self.fi_coef2

        # fixed_items polygons reconstruction
        fi_point1 = []
        fi_point2 = []
        fi_wallnbr = []
        self.fi_polys = []
        for i in range(self.fi_nbr):
            for l in range(len(self.points)):
                if self.fi_vertex1[i] == self.vertex_name[l] and self.fi_vertex2[i] == self.vertex_name[
                    (l + 1) % self.walls_nbr]:
                    fi_wallnbr.append(l)
                    print('ok : ', self.fixed_items['FirstName'][i], 'wall : ', l)
                    point1_x = ((1000 - self.fi_coef1[i]) * self.points[l][0] + self.fi_coef1[i] *
                                self.points[(l + 1) % self.walls_nbr][0]) / 1000
                    point1_y = ((1000 - self.fi_coef1[i]) * self.points[l][1] + self.fi_coef1[i] *
                                self.points[(l + 1) % self.walls_nbr][1]) / 1000
                    fi_point1.append((point1_x, point1_y))
                    point2_x = ((1000 - self.fi_coef2[i]) * self.points[l][0] + self.fi_coef2[i] *
                                self.points[(l + 1) % self.walls_nbr][0]) / 1000
                    point2_y = ((1000 - self.fi_coef2[i]) * self.points[l][1] + self.fi_coef2[i] *
                                self.points[(l + 1) % self.walls_nbr][1]) / 1000
                    fi_point2.append((point2_x, point2_y))

                    v = np.array((self.points[l][1] - self.points[(l + 1) % self.walls_nbr][1],
                                  self.points[(l + 1) % self.walls_nbr][0] - self.points[l][0]),
                                 dtype='f')  # normal vector
                    v /= np.linalg.norm(v)
                    v *= self.fi_width[i]
                    polygon = slg.Polygon(((point1_x, point1_y), (point2_x, point2_y),
                                           (point2_x + v[0], point2_y + v[1]), (point1_x + v[0], point1_y + v[1])))
                    self.fi_polys += [polygon]
                else:
                    if self.fi_vertex1[i] == self.vertex_name[l] and self.fi_vertex2[i] == self.vertex_name[
                        (l + 1) % len(self.points)]:
                        fi_wallnbr.append("inside")
                        print('ok : ', self.fixed_items['FirstName'][i], 'wall : ', 'inside')
                        point1_x = ((1000 - self.fi_coef1[i]) * self.points[l][0] + self.fi_coef1[i] *
                                    self.points[(l + 1) % len(self.points)][0]) / 1000
                        point1_y = ((1000 - self.fi_coef1[i]) * self.points[l][1] + self.fi_coef1[i] *
                                    self.points[(l + 1) % len(self.points)][1]) / 1000
                        fi_point1.append((point1_x, point1_y))
                        point2_x = ((1000 - self.fi_coef2[i]) * self.points[l][0] + self.fi_coef2[i] *
                                    self.points[(l + 1) % len(self.points)][0]) / 1000
                        point2_y = ((1000 - self.fi_coef2[i]) * self.points[l][1] + self.fi_coef2[i] *
                                    self.points[(l + 1) % len(self.points)][1]) / 1000
                        fi_point2.append((point2_x, point2_y))

                        v = np.array(
                            (self.points[l][1] - self.points[(l + 1) % len(self.points)][1],
                             self.points[(l + 1) % len(self.points)][0] - self.points[l][0]),
                            dtype='f')  # normal vector
                        v /= np.linalg.norm(v)
                        v *= self.fi_width[i]
                        polygon = slg.Polygon(
                            ((point1_x, point1_y), (point2_x, point2_y), (point2_x + v[0], point2_y + v[1]),
                             (point1_x + v[0], point1_y + v[1])))
                        self.fi_polys += [polygon]
        print(self.fixed_items)
        print(fi_wallnbr)
        self.fixed_items['WallNumber'] = fi_wallnbr
        self.fixed_items['Polygon'] = self.fi_polys
        self.fixed_items['WallPoint1'] = fi_point1
        self.fixed_items['WallPoint2'] = fi_point2
        print(self.fixed_items)

        list_to_be_drop = []
        for i in range(self.fi_nbr):
            for j in range(self.fi_nbr):
                if i < j:
                    if self.fixed_items['Type'][i] == 'duct' and self.fixed_items['Type'][j] == 'duct':
                        poly_inter = self.fixed_items['Polygon'][i].intersection(self.fixed_items['Polygon'][j].buffer(settings.epsilon))
                        if poly_inter:
                            new_poly = self.fixed_items['Polygon'][i].union(self.fixed_items['Polygon'][j])
                            self.fixed_items.ix[i,'Polygon'] = new_poly
                            print('line to be droped', self.fi_name[j])
                            list_to_be_drop.append(self.fi_name[j])
                            break
        if list_to_be_drop:
            for line in list_to_be_drop:
                self.fixed_items.drop(line, inplace=True)
                self.fi_nbr = len(self.fixed_items['Type'])

        print('WITHOUT Duct', self.fixed_items)

        # fixed items adjacency matrix
        fi_adjacency = []
        fi_position = []
        for i in range(self.fi_nbr):
            fi_adjacency.append([])
            fi_position.append([])
            for j in range(self.fi_nbr):
                if i == j:
                    fi_adjacency[i].append(1)
                else:
                    fi_adjacency[i].append(0)
        last_fi = None
        memo_first_fi = None
        position = 1
        first_wall = True
        for i in range(self.walls_nbr):
            list_wall_fi = []
            list_dist_fi = []
            dist = []
            # To sort the equipment into the wall
            for f in range(self.fi_nbr):
                if self.fixed_items.ix[f, 'WallNumber'] == i:
                    list_wall_fi.append(f)
                    vertex = slg.Point(self.vertice[i][0], self.vertice[i][1])
                    fi_point = slg.Point(self.fixed_items.ix[f, 'WallPoint1'][0],
                                         self.fixed_items.ix[f, 'WallPoint1'][1])
                    dist.append(vertex.distance(fi_point))
                    list_dist_fi.append(vertex.distance(fi_point))
            dist.sort()
            # To find which fi is next to the other
            for k, d in enumerate(dist):
                for j, fi in enumerate(list_wall_fi):
                    if d == list_dist_fi[j]:
                        if k == 0 and first_wall:
                            memo_first_fi = fi
                            last_fi = fi
                            fi_position[fi] = position
                            position += 1
                            first_wall = False
                        else:
                            current_fi = fi
                            fi_position[fi] = position
                            position += 1
                            fi_adjacency[current_fi][last_fi] = 1
                            fi_adjacency[last_fi][current_fi] = 1
                            last_fi = current_fi
        fi_adjacency[last_fi][memo_first_fi] = 1
        fi_adjacency[memo_first_fi][last_fi] = 1

        # To sort fixed items by position
        self.fixed_items['Position'] = fi_position
        self.fixed_items['Sorting_value'] = [pos if isinstance(pos, int) else np.nan for pos in fi_position]
        self.fixed_items = self.fixed_items.sort_values(by='Sorting_value', na_position='last')
        #self.fixed_items = self.fixed_items.drop('Sorting_value')
        print(self.fixed_items)
        print(list(self.fixed_items.index))

        self.fi_name = list(self.fixed_items.index)  # TODO : regarder si cette ligne change quelque chose

        # Definition of rooms data frame
        # ['livingKitchen', 'wc', 'bathroom', 'bedroom1', 'bedroom2', 'entrance']
        self.rooms = []
        nbr_wc = 0
        nbr_bathroom = 0
        nbr_wcbathroom = 0
        nbr_bedroom = 0
        for i in range(len(self.rooms_types)):
            if self.rooms_types[i] == 'wc':
                nbr_wc += 1
                self.rooms.append(self.rooms_types[i] + str(nbr_wc))
            elif self.rooms_types[i] == 'bathroom':
                nbr_bathroom += 1
                self.rooms.append(self.rooms_types[i] + str(nbr_bathroom))
            elif self.rooms_types[i] == 'wcBathroom':
                nbr_wcbathroom += 1
                self.rooms.append(self.rooms_types[i] + str(nbr_wcbathroom))
            elif self.rooms_types[i] == 'bedroom':
                nbr_bedroom += 1
                self.rooms.append(self.rooms_types[i] + str(nbr_bedroom))
            else:
                self.rooms.append(self.rooms_types[i])

        self.rooms_nbr = len(self.rooms_types)

        print(self.rooms)
        print(self.fi_name)

        self.RoomsDf = pd.DataFrame(self.rooms_types, index=self.rooms, columns=["Type"])
        self.RoomsDf['RequiredMinArea'] = self.rooms_min_area
        self.RoomsDf['RequiredMaxArea'] = self.rooms_max_area
        self.RoomsDf['RequiredArea'] = (self.RoomsDf.RequiredMaxArea + self.RoomsDf.RequiredMinArea) / 2
        self.RoomsDf['Size'] = self.rooms_variant

        self.externalSpaces = pd.DataFrame(self.externalSpaces_type, columns=['Type'])

        self.externalSpaces['Area'] = self.externalSpaces_area
        self.externalSpaces['Windows'] = externalSpaces_windowsNames
        print(self.externalSpaces)

        externalSpacesList = []
        externalSpacesTypeList = []
        for i, fi in enumerate(self.fixed_items.index):
            externalSpacesList.append([])
            externalSpacesTypeList.append([])
            for j, ext in enumerate(self.externalSpaces.index):
                if fi in self.externalSpaces.ix[j, 'Windows']:
                    externalSpacesList[i].append(ext)
                    externalSpacesTypeList[i].append(self.externalSpaces.ix[j, 'Type'])
        self.fixed_items['ExternalSpaces'] = externalSpacesList
        self.fixed_items['ExternalSpaceType'] = externalSpacesTypeList

        print(self.fixed_items)
        print(self.RoomsDf)

    @staticmethod
    def give_name_to_fixed_items(fi_types):
        """
        Creates names to identify each fixed items
        (it is really bad because it only works with specific fixed_items types)
        :param fi_types: a list of the type of each fixed items
        :return: a list of names
        """
        fi_names = []
        nbr_duct = 0
        # nbr_frontdoor = 0
        nbr_window = 0

        for i in range(len(fi_types)):
            if fi_types[i] == 'duct':
                nbr_duct += 1
                fi_names.append('D' + str(nbr_duct))
            elif fi_types[i] == 'window' or fi_types[i] == 'doorWindow':
                nbr_window += 1
                fi_names.append('W' + str(nbr_window))
            elif fi_types[i] == 'frontDoor':
                fi_names.append('FD')

        return fi_names


    def load_dicts(self, input_floor_plan_dict, input_setup_dict):
        # loading data
        apartment = input_floor_plan_dict["apartment"]
        rooms = input_setup_dict["setup"]
        points = apartment["vertices"]

        # points list TODO: create list_dict_to_tuples function + bad name should be all_vertices
        points_list = [(points[point_index]["x"], points[point_index]["y"])
                       for point_index in range(len(points))]
        self.points = points_list

        # Plan vertex list TODO: bad name, should be external_vertices
        plan_vertex = [(points[point_index]["x"], points[point_index]["y"])
                       for point_index in apartment["externalWalls"]]
        self.vertice = plan_vertex

        # Plan : polygon of external walls TODO: bad name, should be external_pol
        self.plan = slg.Polygon(plan_vertex)

        # rooms
        self.rooms_types = [str(room["type"]) for room in rooms]
        self.rooms_min_area = [int(room["requiredArea"]["min"]) for room in rooms]
        self.rooms_max_area = [int(room["requiredArea"]["max"]) for room in rooms]
        self.rooms_variant = [str(room["variant"]) for room in rooms]

        # walls
        self.externalWalls = apartment["externalWalls"]

        # load-bearing walls
        lbw_multiLineString = slg.MultiLineString([((points[wall[0]]["x"], points[wall[0]]["y"]),
                                                    (points[wall[1]]["x"], points[wall[1]]["y"]))
                                                   for wall in apartment["loadBearingWalls"]])
        self.load_bearing_walls = lbw_multiLineString

        self.fi_types = [item["type"] for item in apartment["fixedItems"]]
        self.fi_vertex1 = [item["vertex1"] for item in apartment["fixedItems"]]
        self.fi_vertex2 = [item["vertex2"] for item in apartment["fixedItems"]]
        self.fi_coef1 = [item["coef1"] for item in apartment["fixedItems"]]
        self.fi_coef2 = [item["coef2"] for item in apartment["fixedItems"]]
        self.fi_width = [item["width"] for item in apartment["fixedItems"]]

        self.externalSpaces_type = [extSpace["type"] for extSpace in apartment["externalSpaces"]]
        self.externalSpaces_area = [extSpace["area"] for extSpace in apartment["externalSpaces"]]
        self.externalSpaces_windows = [extSpace["windows"] for extSpace in apartment["externalSpaces"]]
        self.externalSpaces_polygon = [extSpace["polygon"] for extSpace in apartment["externalSpaces"] if "polygon" in extSpace.keys()]


class AlgoSettings:
    def __init__(self):
        self.DistFromTheWall = 180
        self.epsilon = 5
        self.epsilon_angle = 0.035  # 2 degrés
        self.DoorSize = 90