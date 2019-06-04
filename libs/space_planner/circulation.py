# coding=utf-8
"""
Circulation module

used to detect isolated rooms and generate a path to connect them

TODO : deal with load bearing walls by defining locations where they can be crossed

"""

import logging
import math
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple, Any, Type, Union, Optional, Callable, Set
from functools import reduce

from libs.io.plot import plot_save
from libs.utils.graph import GraphNx, EdgeGraph
from libs.plan.category import LINEAR_CATEGORIES
from libs.utils.geometry import parallel, opposite_vector, move_point

if TYPE_CHECKING:
    from libs.plan.plan import Space, Plan, Vertex
    from libs.mesh.mesh import Edge
    from libs.specification.specification import Specification, Item

PathsDict = Dict[str, Dict[int, List[List[Union['Vertex', 'Edge']]]]]

DirectionsDict = Dict[int, Dict['Edge', float]]
ScoreArea = Optional[Callable[[float, float, float], float]]

CORRIDOR_WIDTH = 90


class PathInfo():
    """
    class storing information on the circulation path
    """

    def __init__(self, edge_path: List[Tuple['Edge', float]] = None,
                 departure_space: List['Space'] = None,
                 arrival_spaces: List['Space'] = None,
                 departure_penetration: Optional['Edge'] = None,
                 arrival_penetration: Optional['Edge'] = None):
        self.edge_path = edge_path or []
        self.departure_space = departure_space or []
        self.arrival_spaces = arrival_spaces or []
        self.connected_spaces = arrival_spaces or []
        self.departure_penetration = departure_penetration
        self.arrival_penetration = arrival_penetration


class CostRules(Enum):
    """
    The costs of each edge in the path
    """
    water_room_less_than_two_ducts = 10e5
    water_room_default = 1000
    window_room_less_than_two_windows = 10e10
    window_room_default = 5000
    circulation_along_window = 5000
    default = 0


def score_space_area(space_area: float, min_area: float, max_area: float) -> float:
    """
    Scores the space area
    :param space_area:
    :param min_area:
    :param max_area:
    :return:
    """
    sp_score = 0
    if space_area < min_area:
        sp_score = (((min_area - space_area) / min_area) ** 2) * 100
    elif space_area > max_area:
        sp_score = (((space_area - max_area) / max_area) ** 2) * 100
    return sp_score


class Circulator:
    """
    Circulator Class
    contains utilities to detect isolated rooms and connect them to circulation spaces
    """

    def __init__(self, plan: 'Plan', spec: 'Specification', cost_rules: Type[Enum] = CostRules):
        self.plan = plan
        self.spec = spec

        self.paths: PathsDict = {'edge': {level: [] for level in self.plan.levels}}
        self.directions: DirectionsDict = {level: {} for level in self.plan.levels}
        self.paths_info: List[PathInfo] = []
        self.updated_areas = {space: space.cached_area() for space in self.plan.spaces if
                              space.mutable}

        self.cost = 0
        self._reachable_edges = {space: [] for space in self.plan.spaces}
        self._path_calculator = PathCalculator(plan=self.plan, cost_rules=cost_rules)
        self._path_calculator.build()
        self._space_graph = GraphNx()

    def connect(self,
                space_items_dict: Optional[Dict[int, Optional['Item']]] = None,
                score_function: ScoreArea = score_space_area):
        """
        MAIN METHOD OF CIRCULATOR CLASS
        detects isolated rooms and generate a path to connect them
        :return:
        """
        self._init_reachable_edges()
        self._add_circulation_spaces()
        self._add_all_other_spaces()
        self._set_directions(space_items_dict, score_function)
        self._set_penetrations()
        self._remove_redundant_paths()

    def _remove_redundant_paths(self):
        """
        removes redundant paths from self.paths_info
        path_i is considered to be redundant with path_j if the list of rooms connected by
        path_i is included in the list of rooms connected by path_j
        :return:
        """

        def _get_connected_rooms(_edge_path: List['Edge'], _path_info: PathInfo) -> Set:
            """
            gets the set of rooms connected by _edge_path
            :param _edge_path:
            :param _path_info:
            :return:
            """
            connected_rooms = []
            for edge in edge_path:
                sp = self.plan.get_space_of_edge(edge)
                sp_pair = self.plan.get_space_of_edge(edge.pair)
                if sp and sp.mutable:
                    connected_rooms.append(sp)
                if sp_pair and sp_pair.mutable:
                    connected_rooms.append(sp_pair)
            for room in _path_info.departure_space:
                connected_rooms = connected_rooms + [room] if room else connected_rooms
            for room in _path_info.arrival_spaces:
                connected_rooms = connected_rooms + [room] if room else connected_rooms
            return set(connected_rooms)

        list_tuple_connected_rooms = []
        for p, path_info in enumerate(self.paths_info):
            edge_path = [t[0] for t in path_info.edge_path]
            # index of the path_info is stored for later removal
            list_tuple_connected_rooms.append((p, _get_connected_rooms(edge_path, path_info)))

        # tuples sorted by sets length
        list_tuple_connected_rooms.sort(key=lambda t: len(t[1]))
        #redundant paths removal
        for i, tuple_i in enumerate(list_tuple_connected_rooms[:-1]):
            for j, tuple_j in enumerate(list_tuple_connected_rooms[i + 1:]):
                if tuple_i[1] <= tuple_j[1]:  # check set_i is contained by set_j
                    del (self.paths_info[tuple_i[0]])
                    break

    def _set_penetrations(self):
        """
        defines whether a a circulation path shall penetrate, or not, with the spaces it connects
        :return:
        """

        def _get_penetration_edge(_tuple: Tuple['Edge', float], _spaces: List['Space'],
                                  start: bool = True):
            """
            if a penetration in the space is needed to ensure a proper circulation,
            returns the edge through which the path should penetrate in the space
            :param _tuple:
            :param _spaces:
            :param start:
            :return:
            """
            if not _spaces:
                return
            growing_direction = _tuple[1]
            if start:
                growing_direction = -growing_direction
            connecting_edge = _tuple[0].pair if start else _tuple[0]

            if growing_direction > 0:
                next_edge_pair = connecting_edge.next_ortho().pair
            else:
                next_edge_pair = connecting_edge.pair.previous_ortho().pair

            for _space in _spaces:
                if next_edge_pair and not _space.has_edge(next_edge_pair):
                    penetration_edge = connecting_edge.aligned_edge or connecting_edge.continuous_edge
                    if start:
                        penetration_edge = penetration_edge.pair
                    return penetration_edge
            return None

        for path_info in self.paths_info:
            current_path = path_info.edge_path
            if not current_path:
                continue

            path_info.departure_penetration = _get_penetration_edge(current_path[0],
                                                                    path_info.departure_space)

            path_info.arrival_penetration = _get_penetration_edge(current_path[-1],
                                                                  path_info.arrival_spaces,
                                                                  start=False)

    def _set_directions(self,
                        space_items_dict: Optional[Dict[int, Optional['Item']]] = None,
                        score_function: ScoreArea = score_space_area):
        """
        Set the growing direction for each path
        :param space_items_dict: a dictionary matching each space with a specification item
        :param score_function: a score function to evaluate the modification of the space area
        :return: a dictionary containing the scores of each modified spaces
        """

        for p, path_info in enumerate(self.paths_info):
            self.paths_info[p] = self._set_growing_direction(path_info, space_items_dict,
                                                             score_function)

    def _draw_path(self,
                   set_1: List['Edge'],
                   set_2: List['Edge'],
                   level: int) -> Tuple['List[Vertex]', float]:
        """
        Finds the shortest path between two sets of edges
        :return list of vertices on the path and cost of the path
        """
        graph = self._path_calculator.levels_graphs[level]
        path, cost = graph.get_shortest_path(set_1, set_2)

        return path, cost

    def _init_reachable_edges(self):
        """
        for each space, determines which edges can be the arrival of a circulation path
        linking this space
        :return:
        """

        def _parallel_neighbor(e: 'Edge', sp: 'Space', next_edge: bool = True):
            neighbor_edge = sp.next_edge(e) if next_edge else sp.previous_edge(e)
            if parallel(e.vector, neighbor_edge.vector):
                return True
            return False

        def _is_corner_edge(e: 'Edge', sp: 'Space'):

            if (not _parallel_neighbor(e, sp) or
                    not _parallel_neighbor(e, sp, next_edge=False)):
                return True
            sp_pair = self.plan.get_space_of_edge(e.pair)
            if sp_pair:
                if (not _parallel_neighbor(e.pair, sp_pair) or
                        not _parallel_neighbor(e.pair, sp_pair, next_edge=False)):
                    return True
            return False

        def _is_adjacent_to_other_space(e: 'Edge'):
            sp_pair = self.plan.get_space_of_edge(e.pair)
            if not sp_pair or sp_pair.category.external:
                return False
            return True

        def _get_reachable_edges(sp: 'Space'):
            reachable_edges = list(edge for edge in sp.edges if
                                   _is_corner_edge(edge, sp) and _is_adjacent_to_other_space(edge))
            return reachable_edges

        for space in self.plan.mutable_spaces():
            self._reachable_edges[space] = _get_reachable_edges(space)

    def _add_circulation_spaces(self):
        """
        Creates a connectivity graph of the plan, each circulation space is a node
        :return:
        """

        # add all the circulation spaces of the plan in the graph
        for space in self.plan.circulation_spaces():
            self._space_graph.add_node(space)

        # builds connectivity graph for circulation spaces
        for space in self.plan.circulation_spaces():
            for other in self.plan.circulation_spaces():
                if other is not space and other.adjacent_to(space):
                    # if spaces are adjacent, they are connected in the graph
                    # TODO : shouldn't we require a minimum adjacency length (eg: 90 cm)
                    self._space_graph.add_edge(space, other)

        # Create path to connect all circulation space to the root node of each level
        root_nodes = {}

        # find the frontDoor
        front_doors = list(self.plan.get_linears("frontDoor"))

        assert len(front_doors) == 1, "Circulation: A plan should have one and only one front door"

        front_door = front_doors[0]
        entrance = self.plan.get_space_of_edge(front_door.edge)
        root_nodes[entrance.floor.level] = entrance

        # for the other levels find the starting step
        stair_landings = []
        if self.plan.floor_count > 1:
            for starting_step in self.plan.get_linears("startingStep"):
                stair_landing = self.plan.get_space_of_edge(starting_step.edge)
                stair_landings.append(stair_landing)
                if starting_step.floor is entrance.floor:
                    continue
                root_nodes[starting_step.floor.level] = stair_landing

            assert len(root_nodes) == self.plan.floor_count, ("Circulation: "
                                                              "A plan is missing a starting step")
            # Add the connection between stair landings
            stair_landings.sort(key=lambda s: s.floor.level)
            for i, stair_landing in enumerate(stair_landings[1:]):
                self._space_graph.add_edge(stair_landings[i - 1], stair_landings[i])

        # add all the root nodes
        for root_node in root_nodes.values():
            self._space_graph.add_node(root_node)

        # Add the connection to each circulation space to the level root node
        # TODO : an improvement would be to use connected_components to find the optimal path
        for node in self._space_graph.nodes():
            if not self._space_graph.has_path(node, root_nodes[node.floor.level]):
                path, cost = self._draw_path(self._reachable_edges[root_nodes[node.floor.level]],
                                             self._reachable_edges[node], node.floor.level)
                self.cost += cost
                self._add_path(path, root_nodes[node.floor.level], node)
                self._space_graph.add_edge(node, root_nodes[node.floor.level])

    def _add_all_other_spaces(self):
        """
        connects each non circulation space of the plan to a circulation space
        :return:
        """
        # We add each space that is not yet in the graph
        for space in self.plan.mutable_spaces():
            if space not in self._space_graph.nodes():
                self._space_graph.add_node(space)
                for other in self.plan.circulation_spaces():
                    if other is not space and other.adjacent_to(space):
                        self._space_graph.add_edge(space, other)

        for node in list(self._space_graph.nodes()):
            if not self._space_graph.node_connected(node):
                connected_rooms = self._connect_space_to_circulation_graph(node)
                # connected_rooms contains the list of rooms connected by the creation of the path
                # connecting node to a circulation space, in order to prevent
                # the creation of unnecessary paths
                for connected_room in connected_rooms:
                    if not self._space_graph.node_connected(connected_room):
                        # connected_room is no longer isolated
                        self._space_graph.add_edge(connected_room, node)

    def _add_path(self, path: List['Vertex'], departure_space: 'Space',
                  arrival_space: 'Space', link_to_existing_path: bool = False) -> List['Space']:
        """
        update based on computed circulation path
        :return: the list of spaces connected by path
        """

        level = departure_space.floor.level

        connected_rooms = []  # will contain the list of rooms connected by the path

        edge_path = self._get_edge_path(path)
        if not path or not edge_path:
            return connected_rooms

        # TODO : this attribute should no longer be usefull, to be suppressed
        self.paths['edge'][level].append(edge_path)

        # update list of rooms
        for e in edge_path:
            connected = self.plan.get_space_of_edge(e)
            if connected and connected not in connected_rooms and connected.mutable:
                connected_rooms.append(connected)
            connected = self.plan.get_space_of_edge(e.pair)
            if connected and connected not in connected_rooms and connected.mutable:
                connected_rooms.append(connected)

        path_end = path[-1]
        terminal_room = None
        for edge in path_end.edges:
            terminal_room = self.plan.get_space_of_edge(edge)

            if (terminal_room and terminal_room in self._space_graph.nodes()
                    and terminal_room not in connected_rooms and terminal_room.mutable):
                connected_rooms.append(terminal_room)
                break
            terminal_room = self.plan.get_space_of_edge(edge.pair)
            if (terminal_room and terminal_room in self._space_graph.nodes()
                    and terminal_room not in connected_rooms and terminal_room.mutable):
                connected_rooms.append(terminal_room)
                break

        arrival_spaces = [arrival_space] if arrival_space else []
        if terminal_room and not self._space_graph.node_connected(terminal_room):
            arrival_spaces.append(terminal_room)

        if link_to_existing_path:
            # case when the path links an isolated path to an already existing circulation path
            connection_vert = path[-1]
            for path_info in self.paths_info:
                if path_info.edge_path[0][0].start is connection_vert:
                    complementary_edge_path = [(e, 0) for e in edge_path]
                    path_info.edge_path = complementary_edge_path + path_info.edge_path
                    path_info.departure_space = [departure_space]
                    return connected_rooms

        path_info = PathInfo(edge_path=[(e, 0) for e in edge_path],
                             departure_space=[departure_space],
                             arrival_spaces=arrival_spaces)

        self.paths_info.append(path_info)

        return connected_rooms

    def _connect_space_to_circulation_graph(self, space) -> List['Space']:
        """
        connects the given space with a circulation space of the plan
        :return: the list of spaces connected by the circulation drawn to connect space
        """
        path_min = None
        connected_room = None
        cost_min = None
        # compute path between space and every circulation spaces
        for other in self.plan.circulation_spaces():
            if other is not space and space.floor.level is other.floor.level:
                path, cost = self._draw_path(self._reachable_edges[space],
                                             self._reachable_edges[other], space.floor.level)
                if cost_min is None or cost < cost_min:
                    cost_min = cost
                    path_min = path
                    connected_room = other
        # compute path between space and every existing circulation path
        link_to_existing_path = False
        for edge_path in self.paths['edge'][space.floor.level]:
            if edge_path:
                path, cost = self._draw_path(self._reachable_edges[space], edge_path,
                                             space.floor.level)
                if cost_min is None or cost < cost_min:
                    cost_min = cost
                    path_min = path
                    link_to_existing_path = True

        connected_rooms = []
        if path_min is not None:
            arrival_space = connected_room if not link_to_existing_path else None
            connected_rooms = self._add_path(path_min, space, arrival_space, link_to_existing_path)
            self.cost += cost_min

        if connected_room not in connected_rooms:
            connected_rooms.append(connected_room)

        return connected_rooms

    @staticmethod
    def _get_edge_path(path: List['Vertex']) -> List['Edge']:
        """
        from a list of vertices, gets the list of edges connecting those vertices
        :param path:
        :return:
        """
        edge_path = []
        for v, vert in enumerate(path[:-1]):
            for edge in vert.edges:
                if edge.end is path[v + 1]:
                    edge_path.append(edge)
                    break
        return edge_path

    @staticmethod
    def _get_best_item(sp: 'Space', spec: 'Specification') -> Optional['Item']:
        """
        Returns the specification item corresponding to the space of the plan
        :param sp:
        :param spec:
        :return:
        """
        spec_items = spec.items[:]
        corresponding_items = list(filter(lambda i: i.category.name == sp.category.name,
                                          spec_items))

        best_item = min(corresponding_items,
                        key=lambda i: math.fabs(i.required_area - sp.cached_area()),
                        default=None)

        return best_item

    def _set_growing_direction(self, path_info: PathInfo,
                               space_items_dict: Optional[Dict[int, Optional['Item']]] = None,
                               score_function: ScoreArea = score_space_area):
        """
        for each edge of a circulation path, gets the direction in which the corridor has to grow
        so as to bite preferentially on rooms which area is still higher than the minimum spec
        when amputated by the corridor.
        process:
        - decompose the path into its straight portions
        - for each edge of a considered straight portion, computes the most adapted growth direction
        - deduce the most adapted growth direction for the considered portion
        :param path_info:
        :param space_items_dict:
        :param score_function
        :return:
        """

        def _get_lines_of_path(edges: List['Edge']) -> List[List['Edge']]:
            """
            gets the straight portions from a circulation path
            :param edges:
            :return:
            """
            if not edges:
                return [[]]

            lines = [[edges[0]]]
            if len(edges) == 1:
                return lines

            current_vector = edges[0].vector
            current_index = 0
            for _edge in edges[1:]:
                if parallel(_edge.vector, current_vector):
                    lines[current_index].append(_edge)
                else:
                    current_vector = _edge.vector
                    lines.append([_edge])
                    current_index += 1

            return lines

        def _get_score(area_space: 'Dict',
                       path_line: List['Edge'],
                       pair: bool = False,
                       space_item_dict: Optional[Dict[int, Optional['Item']]] = None,
                       _score_function: ScoreArea = score_space_area
                       ) -> float:
            """
            computes the area of each space amputated when a corridor of the path grows on it
            TODO : we should also check the depth of the space to ensure we will not split
                   the space in half by creating the corridor
            :param area_space:
            :param path_line:
            :param pair:
            :param space_item_dict:
            :param _score_function
            :return:
            """
            score = 0

            path_line_selected = [e.pair for e in path_line] if pair else path_line

            for e in path_line_selected:
                sp = self.plan.get_space_of_edge(e)
                area_space[sp] -= e.length * CORRIDOR_WIDTH

            for sp in area_space:
                item = (space_item_dict[sp.id] if space_item_dict
                        else self._get_best_item(sp, self.spec))
                space_area = area_space[sp]
                score += _score_function(space_area, item.min_size.area, item.max_size.area)

            return score

        def _get_growing_direction(path_line: List['Edge'],
                                   space_item_dict: Optional[Dict['Space', 'Item']],
                                   _score_function: ScoreArea,
                                   ) -> float:
            """
            for a given line of edges, gets the direction the corridor has to grow so as to bite
            preferentially on rooms which area is still higher than the minimum spec
            when amputated by the corridor.
            :param path_line:
            :param space_item_dict:
            :param _score_function:
            :return:
            """
            area_space_cw = {}
            area_space_ccw = {}
            for e in path_line:
                # ccw side
                space_ccw = self.plan.get_space_of_edge(e)
                # TODO : this could induce a potential pb if a path_line has two different sides
                #        adjacent to an immutable edge
                if not space_ccw or not space_ccw.mutable:
                    return -1.0
                else:
                    # area_space_ccw[space_ccw] = space_ccw.cached_area()
                    area_space_ccw[space_ccw] = self.updated_areas[space_ccw]

                # cw side
                space_cw = self.plan.get_space_of_edge(e.pair)
                if not space_cw or not space_cw.mutable:
                    return 1.0
                else:
                    # area_space_cw[space_cw] = space_cw.cached_area()
                    area_space_cw[space_cw] = self.updated_areas[space_cw]

            score_cw = _get_score(area_space_cw, path_line, pair=True,
                                  space_item_dict=space_item_dict, _score_function=_score_function)
            score_ccw = _get_score(area_space_ccw, path_line, space_item_dict=space_item_dict,
                                   _score_function=_score_function)

            return 1.0 if score_ccw < score_cw else -1.0

        #####

        path_lines = _get_lines_of_path([t[0] for t in path_info.edge_path])
        level = path_info.departure_space[0].floor.level
        edge_path = []
        for line in path_lines:
            growing_direction = _get_growing_direction(line, space_items_dict, score_function)
            for edge in line:
                self.directions[level][edge] = growing_direction
                edge_path.append((edge, growing_direction))
                # update space area when overlapped by a corridor
                support_edge = edge if growing_direction > 0 else edge.pair
                space_overlapped = self.plan.get_space_of_edge(support_edge)
                self.updated_areas[space_overlapped] -= support_edge.length * CORRIDOR_WIDTH
        path_info.edge_path = edge_path
        return path_info

    def plot(self, show: bool = False, save: bool = True):
        """
        plots plan with circulation paths
        :return:
        """

        ax = self.plan.plot(show=show, save=False)

        number_of_floors = self.plan.floor_count

        for path_info in self.paths_info:
            level = path_info.departure_space[0].floor.level
            _ax = ax[level] if number_of_floors > 1 else ax
            for tup in path_info.edge_path:
                edge = tup[0]
                edge.plot(ax=_ax, color='blue')
                # representing the growing direction
                pt_ini = move_point((edge.start.x, edge.start.y), edge.vector, 0.5)
                vector = (edge.normal if tup[1] > 0 else opposite_vector(edge.normal))
                pt_end = move_point(pt_ini, vector, 90)
                _ax.arrow(pt_ini[0], pt_ini[1], pt_end[0] - pt_ini[0],
                          pt_end[1] - pt_ini[1])

        plot_save(save, show)


class PathCalculator:
    """
    PathCalculator class
    builds and manages a graph that can be used by a circulator so as to compute shortest path
    between two unconnected spaces.
    The library used to build the graph can be specified
    """
    window_cat = [cat for cat in LINEAR_CATEGORIES if LINEAR_CATEGORIES[cat].window_type]

    def __init__(self, plan: 'Plan', cost_rules: Type[Enum], graph_lib: str = 'networkx'):
        self.plan = plan
        self.graph_lib = graph_lib
        self.levels_graphs = None
        self.rules_cost = cost_rules

        self.component_edges = {'duct_edges': self.plan.category_edges('duct'),
                                'window_edges': self.plan.category_edges(*self.window_cat)}

    def __repr__(self):
        return 'Grapher:\n graph library :' + self.graph_lib + '\n'

    def build(self):
        """
        runs through space edges and adds branches to the graph, for each branch computes a weight
        :return:
        """
        self.levels_graphs = {level: EdgeGraph(self.graph_lib) for level in self.plan.levels}

        for space in self.plan.mutable_spaces():
            self._add_to_graph(space)

    def _add_to_graph(self, space: 'Space'):
        """
        add edge to the graph and computes its cost
        return:
        """
        graph = self.levels_graphs[space.floor.level]

        def _get_space_info(_space: 'Space') -> Dict[str, Any]:
            # info needed on the space to attribute a cost to each edge of this space
            num_ducts = _space.count_ducts()
            num_windows = _space.count_windows()
            needed_ducts = list(
                needed_space for needed_space in _space.category.needed_spaces if
                needed_space.name is 'duct')
            needed_windows = list(
                needed_linear for needed_linear in _space.category.needed_linears if
                needed_linear.window_type)
            info = {
                "num_ducts": num_ducts,
                "num_windows": num_windows,
                "needed_ducts": needed_ducts,
                "needed_windows": needed_windows,
            }
            return info

        space_info = _get_space_info(space)
        for edge in space.edges:
            cost = self._cost(edge, space_info)
            graph.add_edge(edge, cost)

    def _get_cost(self, edge: 'Edge', space_info: Dict) -> CostRules:
        """
        gets the rule for edge cost computation
        :return: float
        """
        cost = CostRules.default

        if edge.pair in self.component_edges['duct_edges'] and space_info["needed_ducts"]:
            if space_info["num_ducts"] <= 2:
                cost = CostRules.water_room_less_than_two_ducts
            else:
                cost = CostRules.water_room_default

        elif edge in self.component_edges['window_edges'] and space_info["needed_windows"]:
            if space_info["num_windows"] <= 2:
                cost = CostRules.window_room_less_than_two_windows
            else:
                cost = CostRules.window_room_default

        elif edge in self.component_edges['window_edges']:
            cost = CostRules.circulation_along_window

        return cost

    def _cost(self, edge: 'Edge', space_info: Dict) -> float:
        """
        computes the cost of an edge
        :return: float
        """
        cost = edge.length + self._get_cost(edge, space_info).value
        return cost


if __name__ == '__main__':
    import libs.io.reader as reader
    from libs.plan.plan import Plan
    from libs.modelers.seed import SEEDERS
    from libs.modelers.grid import GRIDS
    from libs.space_planner.space_planner import SPACE_PLANNERS
    from libs.plan.category import SPACE_CATEGORIES
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plan_index", help="choose plan index",
                        default=0)

    args = parser.parse_args()
    plan_index = int(args.plan_index)

    logging.getLogger().setLevel(logging.DEBUG)


    def duplex():
        """
        Test
        :return:
        """
        boundaries = [(0, 500), (400, 500), (400, 0), (1500, 0), (1500, 700), (1000, 700),
                      (1000, 800),
                      (0, 800)]
        boundaries_2 = [(0, 500), (400, 500), (400, 400), (1000, 400), (1000, 800), (0, 800)]

        plan = Plan("Solution_Tests_Multiple_floors")
        floor_1 = plan.add_floor_from_boundary(boundaries)
        floor_2 = plan.add_floor_from_boundary(boundaries_2, floor_level=1)

        terrace_coords = [(400, 400), (400, 200), (1300, 200), (1300, 700), (1000, 700),
                          (1000, 400)]
        plan.insert_space_from_boundary(terrace_coords, SPACE_CATEGORIES["terrace"], floor_1)
        garden_coords = [(400, 200), (400, 0), (1500, 0), (1500, 700), (1300, 700), (1300, 200)]
        plan.insert_space_from_boundary(garden_coords, SPACE_CATEGORIES["garden"], floor_1)
        duct_coords = [(350, 500), (400, 500), (400, 520), (350, 520)]
        plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
        duct_coords = [(350, 780), (400, 780), (400, 800), (350, 800)]
        plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_1)
        hole_coords = [(400, 700), (650, 700), (650, 800), (400, 800)]
        plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_1)
        plan.insert_linear((650, 800), (650, 700), LINEAR_CATEGORIES["startingStep"], floor_1)
        plan.insert_linear((275, 500), (340, 500), LINEAR_CATEGORIES["frontDoor"], floor_1)
        plan.insert_linear((550, 400), (750, 400), LINEAR_CATEGORIES["doorWindow"], floor_1)
        plan.insert_linear((1000, 450), (1000, 650), LINEAR_CATEGORIES["doorWindow"], floor_1)
        plan.insert_linear((0, 700), (0, 600), LINEAR_CATEGORIES["window"], floor_1)

        duct_coords = [(350, 500), (400, 500), (400, 520), (350, 520)]
        plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
        duct_coords = [(350, 780), (400, 780), (400, 800), (350, 800)]
        plan.insert_space_from_boundary(duct_coords, SPACE_CATEGORIES["duct"], floor_2)
        hole_coords = [(400, 700), (650, 700), (650, 800), (400, 800)]
        plan.insert_space_from_boundary(hole_coords, SPACE_CATEGORIES["hole"], floor_2)
        plan.insert_linear((400, 700), (400, 780), LINEAR_CATEGORIES["startingStep"], floor_2)
        plan.insert_linear((500, 400), (600, 400), LINEAR_CATEGORIES["window"], floor_2)
        plan.insert_linear((1000, 550), (1000, 650), LINEAR_CATEGORIES["window"], floor_2)
        plan.insert_linear((0, 700), (0, 600), LINEAR_CATEGORIES["window"], floor_2)

        GRIDS["001"].apply_to(plan)

        plan.plot()

        SEEDERS["directional_seeder"].apply_to(plan)

        plan.plot()

        spec = reader.create_specification_from_file("test_solution_duplex_setup.json")
        spec.plan = plan
        plan.plot()
        space_planner = SPACE_PLANNERS["standard_space_planner"]

        return space_planner.apply_to(spec), space_planner


    def connect_plan():
        """
        Test
        :return:
        """

        best_solutions, space_planner = duplex()

        for solution in best_solutions:
            circulator = Circulator(plan=solution.plan, spec=space_planner.spec)
            circulator.connect()
            circulator.plot()


    connect_plan()
