from network import network
from opponent import *
import traci
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import random
import csv
from qpsolvers import solve_qp
from cvxopt import matrix, solvers
import os
from utils import solve_and_store,solve_with_relaxation,minimize_with_relaxation
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams["axes.unicode_minus"]=False
from scipy.optimize import linprog

from typing import List, Dict, Callable, Any

class control(network):
    def __init__(self,mfd_type='',load_mfd=True):
        self.condition = 'dynamic'  # 动态分区
        self.test_begin = 800
        self.test_end = 900
        #self.range_M = 0.2
        self.range_M_strict = 0.4
        self.range_M_relaxed = 0.4
        self.end_time_threshold = 20000

        self.stop_headway = 7
        self.saturated_flow = 0.5  # 0.5veh/s
        self.sumocfg_file = 'data/yangzhou0.8.sumocfg'
        self.warm_time = 200
        #self.time_length = 3600  # 调试 正常3600
        self.mac_length = 100
        self.mic_length = 10  # signal & route
        self.update_edge_tt = 20 #20s更新一次速度
        self.threshold_veh_num=500
        self.mfd_type= mfd_type
        self.beta = 0.1

        self.oversaturation_ratio=0.3
        '''
        self.warm_time = 0
        self.time_length = 100  # 调试 正常3600
        self.mac_length = 20  # 控制步长=10s joint model
        self.mic_length = 10  # signal & route
        '''

        self.save_path = 'control'
        self.read_path = 'partition'  # revise1

        # 读取region数据
        self._read_region(load_mfd=load_mfd)

    def check_saturated_condition(self):
        #exist oversaturated region
        for region in self.regions.values():
            if region.n >= region.critical*self.oversaturation_ratio:
                # 如果有一个区域的 n 值大于等于 critical，则返回 True
                return True
        # 所有区域的 n 值都小于 critical，返回 False
        return False
    '''
    def check_terminate_condition(self):
        for region in self.regions.values():
            if region.n >= region.critical*self.oversaturation_ratio:
                # 如果有一个区域的 n 值大于等于 critical，则返回 False
                return False
        # 所有区域的 n 值都小于 critical，返回 True
        return True
    '''
    def _draw_density(self, time, compare=False):
        x = []
        y = []
        z = []
        for id in self.edges_id:
            edge = self.edges[id]
            reverse_id = edge.reverse
            vehicle_number = traci.edge.getLastStepVehicleNumber(id)
            if reverse_id == 0:
                edge.twodir_density = vehicle_number / (edge.numlane * edge.length) * 1000
            else:
                reverse_edge = self.edges[reverse_id]
                reverse_vehicle_number = traci.edge.getLastStepVehicleNumber(reverse_id)
                edge.twodir_density = (vehicle_number + reverse_vehicle_number) / (
                        (edge.numlane + reverse_edge.numlane) * edge.length) * 1000

            X = [list(edge.frompos)[0], list(edge.topos)[0]]
            Y = [list(edge.frompos)[1], list(edge.topos)[1]]

            x_temp = list(np.linspace(X[0], X[1], 100))
            y_temp = []
            for i in range(len(x_temp)):
                y_temp.append(
                    (Y[1] - Y[0]) / (X[1] - X[0]) * x_temp[i] - (Y[1] - Y[0]) / (X[1] - X[0]) * X[0] + Y[0])
            x += x_temp
            y += y_temp
            z += [edge.twodir_density] * 100

        plt.scatter(x, y, c=z, s=2, vmin=0, vmax=100, cmap='RdYlGn_r')

        plt.axis('off')
        plt.colorbar()
        if compare == False:
            plt.savefig(self.save_path + '/' + 'density_%s.jpg' % str(time), bbox_inches='tight', dpi=1000)
        else:
            plt.savefig(self.save_path + '/' + 'density_%s(compare).jpg' % str(time), bbox_inches='tight', dpi=1000)

        plt.close()

    def _read_region(self,load_mfd=True):
        self.regions = {}
        with open(self.read_path + '/' + 'region_edge_list.csv') as f:
            f_csv = csv.reader(f)
            k = 0
            for row in f_csv:
                if k == 0:
                    k = 1
                    continue

                id = int(row[2])
                self.regions[id] = Region(id)
                for i in range(len(row) - 4):
                    if row[4 + i] != '':
                        self.regions[id].edge_list.add(row[4 + i])
        if not load_mfd:
            return
        path=os.path.join(self.read_path,self.mfd_type, 'MFD_parameter.csv')
        with open(path) as f:
            f_csv = csv.reader(f)
            k = 0
            for row in f_csv:
                if k == 0:
                    k = 1
                    continue
                # 一条region数据
                temp = row[2][1:-1]  # str表示的mfd_param的list
                temp = temp.split(' ')
                temp = [i for i in temp if i != '']
                id=int(float(row[1]))
                for i in range(3):
                    if 'e' in temp[i]:
                        temp2 = temp[i].split('e')
                        self.regions[id].mfd_param.append(
                            float(temp2[0]) * (10 ** int(temp2[1])))
                    else:
                        self.regions[id].mfd_param.append(temp[i])

                self.regions[id].critical = int(row[3])

    def _region_demands(self, regions=None, begin=0, end=0):
        def reset_Q():
            Q = {}
            for i in regions:
                q = {j: 0 for j in regions}
                Q[i] = q
            return Q

        # 计算总时间段数目
        total_intervals = int((end - begin) / self.mac_length)

        # 初始化所有时段，设置默认需求为0
        for i in regions:
            regions[i].demands = [reset_Q()[i] for _ in range(total_intervals)]

        tree = ET.parse("data/yangzhou.rou.xml")
        root = tree.getroot()

        for child in root:
            if child.tag != 'vehicle':
                continue
            data = child.attrib
            depart = float(data['depart'])

            # 如果该车辆的出发时间不在指定范围内，跳过
            if depart < begin or depart >= end:
                continue

            # 确定车辆属于哪个时间段
            interval_index = int((depart - begin) / self.mac_length)
            # 遍历车辆的route信息并分配需求
            for subchild in child:
                if subchild.tag == 'route':
                    edges = subchild.attrib['edges'].split()
                    o = edges[0]  # 起点
                    d = edges[-1]  # 终点
                    for i in regions:
                        # print(234,regions[i].node_list)
                        if o in regions[i].edge_list:
                            for j in regions:
                                if d in regions[j].edge_list:
                                    # 将需求加到对应时间段的Q中
                                    regions[i].demands[interval_index][j] += 1
                                    break
        return regions

    def _update_items(self,regions=None,edges=None,tls=None):
        self.regions=regions
        self.edges=edges
        self.trafficlights=tls

    def _filter_phase(self):
        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                for tl in self.regions[i].toregion_trafficlight[h]:
                    self.trafficlights[tl]._filter_phases()

    def _initialize_vars(self, joint_control=False, only_perimeter_control=False):
        self._filter_phase() #111

        self.total_delay = 0
        self.total_vehicles = set()
        self.total_output = 0

        self.network_delay=0
        self.network_n = 0
        self.network_output = 0

        self.step = 0

        self.N_data = []
        self.network_data = []

        self.control_active = False  # 标记是否正在进行联合控制
        self.log_file = f'{self.save_path}/control_log.txt'  # 记录日志的文件名

        # 根据不同运行模式初始化
        if joint_control:
            self.b_data = []
            self.c_data = []
        if joint_control or only_perimeter_control:
            self.real_M_set = {f'{i}_{h}': [] for i in self.regions for h in self.regions[i].neighbor}
        # add for PI-based perimeter control
        if only_perimeter_control:
            self.M_last_mac_step={f'{i}_{h}': 0 for i in self.regions for h in self.regions[i].neighbor}

    def _reset_data(self):
        for i in self.regions:
            self.regions[i].n = 0
            self.regions[i].nd = {}
            self.regions[i].q = {}
            for j in self.regions:
                self.regions[i].nd[j] = 0
                self.regions[i].q[j] = 0
        for i in self.regions:
            if self.step < len(self.regions[i].demands):
                self.regions[i].q = self.regions[i].demands[self.step]
            else:
                self.regions[i].q = 0

    def update_edge_travel_time(self):
        for id in self.edges_id:
            edge = self.edges[id]

            vehicle=traci.edge.getLastStepVehicleNumber(id)

            if vehicle==0:
                speed=edge.max_speed
            else:
                speed = traci.edge.getLastStepMeanSpeed(id)

            edge.update_speed(speed)

            edge.travel_time=edge.length/max(np.mean(edge.speed_list),0.1)

            traci.edge.adaptTraveltime(id,edge.travel_time)


    def _get_control_vehicle_data(self):

        for i in self.regions:
            control_vehicles_edge = self.regions[i].edge_list
            # 确定control_vehicles
            control_vehicles = []
            for j in control_vehicles_edge:
                control_vehicles += traci.edge.getLastStepVehicleIDs(j)

            for j in self.regions[i].nd:
                self.regions[i].control_vehicles[j] = []

            for j in control_vehicles:
                d_edge = traci.vehicle.getRoute(j)[-1]
                destination = self.edges[d_edge].eto

                for k in self.regions[i].nd:
                    if d_edge in self.regions[k].edge_list:
                        d_region = k
                        break

                routes = self._get_possible_route(j)

                next_regions = []
                for route in routes:
                    if d_region == i:
                        next_region = i
                        next_regions.append(next_region)
                        continue

                    next_region = 0
                    for k in range(len(route) - 1):
                        if route[k] in self.regions[i].internal_edge_entry:
                            # route[k]为边界edge
                            for m in self.regions[i].neighbor:
                                if route[k + 1] in self.regions[m].edge_list:
                                    next_region = m
                                    break
                    if next_region == 0:
                        # 一些特殊的路线（比如回头路）
                        for k in route:
                            if k not in self.regions[i].edge_list:
                                for m in self.regions[i].neighbor:
                                    if k in self.regions[m].edge_list:
                                        next_region = m
                                        break
                                break
                        # next_region=i
                    next_regions.append(next_region)
                self.regions[i].control_vehicles[d_region].append(Vehicle(id=j, destination=destination, routes=routes,
                                                                              next_regions=next_regions))  # ,boundary=boundary))


    def _update_network_data(self):
        total_vehicles = set()

        for i in self.regions:
            region_n = 0
            region_veh = set()
            for j in self.regions[i].edge_list_IIE:
                veh_id = traci.edge.getLastStepVehicleIDs(j)
                region_n += len(veh_id)
                region_veh.update(veh_id)

            self.regions[i].n = region_n
            total_vehicles.update(region_veh)

            output = 0
            for j in self.regions[i].vehicles:
                if j not in region_veh:
                    output += 1
            self.regions[i].output = output
            self.regions[i].vehicles = region_veh

        self.network_n = len(total_vehicles)
        self.network_delay = self.network_n * self.mic_length
        self.total_delay += self.network_delay

        self.network_output = traci.simulation.getArrivedNumber()

        self.total_output += self.network_output
        self.total_vehicles = total_vehicles



    def _get_data(self, time):
        if time%self.update_edge_tt==0:
            self.update_edge_travel_time()

        temp_M = 0 if time % self.mac_length == 0 else 1

        # ------------joint model-----------
        for i in self.regions:
            for j in self.regions[i].vehicles:
                d_edge = traci.vehicle.getRoute(j)[-1]
                # d = self.edges[d_edge].eto
                # 车辆终点
                for k in self.regions:
                    if d_edge in self.regions[k].edge_list:
                        # if d in self.regions[k].node_list:
                        self.regions[i].nd[k] += 1
                        break

            # ---------------route choice-------------
        self._get_control_vehicle_data()

        # ------------------signal control---------------------
        # 计算每个connection的throughput
        for i in self.regions:
            for h in self.regions[i].neighbor:
                connection_set = self.regions[i].toregion_connection[h]
                for cid in connection_set:
                    fromlane = self.connections[cid].cfrom + '_' + self.connections[cid].fromlane
                    v_set = list(traci.lane.getLastStepVehicleIDs(fromlane))
                    num = 0
                    for v in v_set:
                        dis = traci.lane.getLength(fromlane) - traci.vehicle.getLanePosition(v)
                        if dis < self.mic_length * traci.vehicle.getSpeed(v) or (
                                traci.vehicle.getSpeed(v) < 0.1 and dis < self.mic_length * traci.lane.getMaxSpeed(
                                fromlane)):
                            # 按原速行驶mic_length内可以离开该路段，或者停止的车辆
                            num += 1
                    fromlane_queue = num * self.connections[cid].lanenum
                    # 偏高
                    # tolane = self.connections[cid].cto+'_'+self.connections[cid].tolane
                    toedge = self.connections[cid].cto
                    tolane_length = self.edges[toedge].length

                    # 按照排队长度最长的lane为准看capacity
                    max_queue_downstream = 0
                    for lid in range(self.edges[toedge].numlane):
                        lane = toedge + '_' + str(lid)
                        queue_length = traci.lane.getLastStepHaltingNumber(lane)
                        if queue_length > max_queue_downstream:
                            max_queue_downstream = queue_length
                    tolane_capacity = (tolane_length / self.stop_headway - max_queue_downstream) * 0.8

                    max_throughput = (self.mic_length - 4) * self.saturated_flow * self.connections[cid].lanenum
                    self.connections[cid].throughput = min(fromlane_queue, tolane_capacity,
                                                           max_throughput) / self.mic_length
        # for capacity constraints
        tl_M_max = {}
        tl_M_min = {}
        # superphase集合及throughput
        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                # 计算每个tl每个phase的throughput
                throughput = {}
                reversed_throughput = {}
                for tl in self.regions[i].toregion_trafficlight[h]:
                    connection_set = self.regions[i].toregion_trafficlight[h][tl]
                    if tl not in self.regions[h].toregion_trafficlight[i]:
                        reversed_connection_set = []
                    else:
                        reversed_connection_set = self.regions[h].toregion_trafficlight[i][tl]
                    for p in range(len(self.trafficlights[tl].phase)):
                        phase = self.trafficlights[tl].phase[p]
                        throughput['%s_%s' % (tl, phase)] = 0
                        reversed_throughput['%s_%s' % (tl, phase)] = 0
                        for cid in connection_set:
                            linkindex = self.connections[cid].linkindex
                            if phase[linkindex] == 'g' or phase[linkindex] == 'G':
                                throughput['%s_%s' % (tl, phase)] += self.connections[cid].throughput
                        for cid in reversed_connection_set:
                            linkindex = self.connections[cid].linkindex
                            if phase[linkindex] == 'g' or phase[linkindex] == 'G':
                                reversed_throughput['%s_%s' % (tl, phase)] += self.connections[cid].throughput

                # superphase集合及throughput
                super_phase_set = []
                super_phase_dic = {}
                temp = {}
                phase_num = {}
                for tl in self.regions[i].toregion_trafficlight[h]:
                    temp[tl] = 0
                    phase_num[tl] = len(self.trafficlights[tl].phase) - 1  # 从0算起
                id = 0

                tl_M_max['%s_%s' % (i, h)] = 0
                tl_M_min['%s_%s' % (i, h)] = 100000
                tl_M_max['%s_%s' % (h, i)] = 0
                tl_M_min['%s_%s' % (h, i)] = 100000
                while True:
                    acn = 0
                    arcn = 0
                    super_phase = {}
                    for tl in phase_num:
                        phase = self.trafficlights[tl].phase[temp[tl]]
                        super_phase[tl] = phase
                        acn += throughput['%s_%s' % (tl, phase)]
                        arcn += reversed_throughput['%s_%s' % (tl, phase)]
                    if super_phase != {}:
                        super_phase_set.append(super_phase)

                        if acn >= tl_M_max['%s_%s' % (i, h)]:
                            tl_M_max['%s_%s' % (i, h)] = acn
                        if acn <= tl_M_min['%s_%s' % (i, h)]:
                            tl_M_min['%s_%s' % (i, h)] = acn
                        if arcn >= tl_M_max['%s_%s' % (h, i)]:
                            tl_M_max['%s_%s' % (h, i)] = arcn
                        if arcn <= tl_M_min['%s_%s' % (h, i)]:
                            tl_M_min['%s_%s' % (h, i)] = arcn

                        super_phase_dic[id] = [acn, arcn]

                    id += 1

                    index = 0
                    for tl in phase_num:
                        if temp[tl] != phase_num[tl]:
                            index = 1
                    if index == 0:
                        break
                    for tl in phase_num:
                        if temp[tl] < phase_num[tl]:
                            temp[tl] += 1
                            break
                        else:
                            temp[tl] = 0
                            continue
                self.regions[i].toregion_superphaseset[h] = super_phase_set

                self.regions[i].toregion_superphasedic[h] = super_phase_dic
                if tl_M_max['%s_%s' % (i, h)] == 0:
                    tl_M_max['%s_%s' % (i, h)] = 0.15
                    tl_M_min['%s_%s' % (i, h)] = 0
                if tl_M_max['%s_%s' % (h, i)] == 0:
                    tl_M_max['%s_%s' % (h, i)] = 0.15
                    tl_M_min['%s_%s' % (h, i)] = 0

        # 单位时间内的Mmax和Mmin
        self.M_second_max = {}
        self.M_second_min = {}
        # mic时间内的M
        if temp_M == 0:
            self.previous_M = {}
            for i in self.regions:
                for h in self.regions[i].neighbor:
                    self.previous_M['%d_%d' % (i, h)] = []

        self.previous_notl_M = {}
        for i in self.regions:
            for h in self.regions[i].neighbor:
                connection_set = self.regions[i].toregion_connection[h]
                passnum = 0
                notlpassnum = 0
                vehicles = []
                notlvehicles = []
                for cid in connection_set:
                    connection = self.connections[cid]
                    lane = connection.cfrom + '_' + connection.fromlane
                    vehicles += traci.lane.getLastStepVehicleIDs(lane)
                    if connection.type == 'notl':
                        notlvehicles += traci.lane.getLastStepVehicleIDs(lane)
                for v in self.regions[i].toregion_vehicles[h]:
                    if v not in vehicles:
                        passnum += 1
                for v in self.regions[i].toregion_notl_vehicles[h]:
                    if v not in notlvehicles:
                        notlpassnum += 1
                self.regions[i].toregion_notl_vehicles[h] = notlvehicles
                self.regions[i].toregion_vehicles[h] = vehicles
                self.previous_notl_M['%d_%d' % (i, h)] = notlpassnum / self.mic_length
                self.real_M_set['%d_%d' % (i, h)].append(passnum / self.mic_length)
                if temp_M != 0:
                    self.previous_M['%d_%d' % (i, h)].append(passnum / self.mic_length)
                if temp_M == 0:
                    self.M_second_max['%d_%d' % (i, h)] = self.previous_notl_M['%d_%d' % (i, h)] + tl_M_max[
                        '%d_%d' % (i, h)]
                    self.M_second_min['%d_%d' % (i, h)] = self.previous_notl_M['%d_%d' % (i, h)] + tl_M_min[
                        '%d_%d' % (i, h)]

    def restore_to_fixed_signal(self):
        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                for tl in self.regions[i].toregion_trafficlight[h]:
                    traci.trafficlight.setProgram(tl, '0')



    def _save_network_data(self, time):
        # revise 3: add the whole network performance represented as the final region
        data = {}
        data['time'] = time
        data['real_n_t'] = self.network_n
        data['output_t-1'] = self.network_output
        data['delay']=self.network_delay
        self.network_data.append(data)
    def _save_output(self, prefix, random_generator=False, num=-1):

        output_suffix = f'_RG{num}' if random_generator else ''

        # 保存主要数据
        pd.DataFrame(self.N_data).to_csv(f'{self.save_path}/{prefix}_N{output_suffix}.csv')
        pd.DataFrame(self.network_data).to_csv(f'{self.save_path}/{prefix}_network{output_suffix}.csv')

        # 特定模式保存额外数据
        if prefix.startswith('joint_control'):
            pd.DataFrame(self.b_data).to_csv(f'{self.save_path}/{prefix}_M{output_suffix}.csv')
            pd.DataFrame(self.c_data).to_csv(f'{self.save_path}/{prefix}_c{output_suffix}.csv')

        print(f'{prefix}_total_delay{output_suffix}: {self.total_delay}')
        print(f'{prefix}_total_output{output_suffix}: {self.total_output}')

    def _action_randomroute(self):
        # second best

        for i in self.regions:
            for j in self.regions[i].nd:
                for v in self.regions[i].control_vehicles[j]:
                    routes = v.routes
                    choice = random.randint(0, len(routes) - 1)
                    route = routes[choice]
                    traci.vehicle.setRoute(v.id, route)
    def _action_logit_based_route(self):

        for i in self.regions:
            for j in self.regions[i].nd:
                for v in self.regions[i].control_vehicles[j]:
                    routes = v.routes
                    travel_times=[]
                    for route in routes:
                        travel_time=0
                        for edge in route:
                            travel_time+=self.edges[edge].travel_time
                        travel_times.append(travel_time)
                    utilities=[-self.beta*t for t in travel_times]

                    # 使用数值稳定化来避免 np.exp() 溢出
                    max_utility = np.max(utilities)  # 找到 utilities 中的最大值
                    utilities_stable = utilities - max_utility  # 每个效用减去最大值

                    # 使用 logit 模型计算选择概率
                    exp_utilities = np.exp(utilities_stable)  # 现在效用值稳定
                    total_exp_utilities = np.sum(exp_utilities)
                    probabilities = exp_utilities / total_exp_utilities  # 归一化为概率

                    # 随机抽样选择路径（根据 logit 模型的概率选择）
                    chosen_route_idx = np.random.choice(len(routes), p=probabilities)
                    chosen_route = routes[chosen_route_idx]

                    # 更新车辆的选择路径
                    traci.vehicle.setRoute(v.id, chosen_route)

    def has_loop(self,route):
        return len(route) != len(set(route))

    def remove_loop(self,route):
        seen_edges = {}
        for i, edge in enumerate(route):
            if edge in seen_edges:
                # 删除重复边从第一次出现到该重复边的部分
                return route[:seen_edges[edge] + 1] + route[i + 1:]
            seen_edges[edge] = i
        return route

    def _get_possible_route(self,v):
        present_route = traci.vehicle.getRoute(v)
        present_edge = traci.vehicle.getRoadID(v)

        if present_edge == present_route[-1] or present_edge == present_route[-2] or present_edge==present_route[-3]:
            #present next desitination固定
            for k in range(len(present_route)):
                if present_route[k]==present_edge:
                    present_route=present_route[k:]
                    break
            routes = [present_route]
        else:
            for k in range(len(present_route) - 1):
                if present_route[k] == present_edge:
                    present_next_edge=present_route[k + 1]
                    present_route = present_route[k:]
                    break
            optimal_route = (present_edge,) + traci.simulation.findRoute(present_next_edge, present_route[-1],routingMode=traci.constants.ROUTING_MODE_DEFAULT).edges
            #optimal_route = (present_edge,) + traci.simulation.findRoute(present_next_edge, present_route[-1],routingMode=traci.constants.ROUTING_MODE_AGGREGATED).edges

            if self.has_loop(optimal_route) :
                optimal_route = self.remove_loop(optimal_route)

            max_route_length = 80  # 假设设置一个合理的最大路径长度
            if len(optimal_route) > max_route_length:
                print(f"Route too long for vehicle {v}: present_route or optimal_route",v,present_route,optimal_route)

            if present_route == optimal_route:
                routes = [present_route]
            else:
                routes = [present_route, optimal_route]
                '''
                optimal_travel_time=0
                present_travel_time=0
                for id in optimal_route:
                    optimal_travel_time+=self.edges[id].travel_time
                for id in present_route:
                    present_travel_time+=self.edges[id].travel_time

                if present_travel_time>optimal_travel_time*1.2:
                    routes=[optimal_route]
                elif optimal_travel_time>present_travel_time*1.2:
                    routes=[present_route]
                else:
                    routes = [present_route, optimal_route]
                '''

        return routes

    def _calculate_weight(self, node, phase):
        # self.saturated_flow=1 #饱和流率，由于所有相位设置同一流率，取1
        controlledlinks = traci.trafficlight.getControlledLinks(node)

        weight = {}
        for i in range(len(phase)):
            if phase[i] != 'G' and phase[i] != 'g':
                continue
            if controlledlinks[i] == []:
                continue

            inputlane = controlledlinks[i][0][0]
            outputlane = controlledlinks[i][0][1]
            upstream_queue = traci.lane.getLastStepHaltingNumber(inputlane)
            outputedge = traci.lane.getEdgeID(outputlane)
            downstream_queue = int(
                traci.edge.getLastStepHaltingNumber(outputedge) / traci.edge.getLaneNumber(outputedge))
            # 进口车道的排队数-平均出口车道的排队数
            if inputlane not in weight:
                weight[inputlane] = []
            weight[inputlane].append((upstream_queue - downstream_queue) * self.saturated_flow)

        sumweight = 0
        for inputlane in weight:
            weight[inputlane] = np.mean(weight[inputlane])
            sumweight += weight[inputlane]

        return sumweight


class joint_control(control):
    def __init__(self,mfd_type='',load_mfd=True):
        # 调用父类 control 的构造函数
        super().__init__(mfd_type=mfd_type,load_mfd=load_mfd)
        self.save_path='control/'+'joint_control'

    def run(self):
        self._init()

        # 获取分区的inputlane,outputlane,bound等
        self.regions = self._region_nodelist(label=False, regions=self.regions)
        self.regions = self._region_demands(regions=self.regions,
                                                   begin=self.warm_time, end=100000)

        self._sim_joint_control()
        self._terminate()


    def RG(self):
        self.save_path='control/'+'joint_control_random_generate'
        self.RG_num=10
        for num in range(self.RG_num):
            seed = random.randint(1, 1000)
            self._init(seed)

            self.regions = self._region_nodelist(label=False, regions=self.regions)
            self.regions = self._region_demands(regions=self.regions,
                                                   begin=self.warm_time, end=100000)
            self._sim_joint_control(random_generator = True, num = num)
            self._terminate()
    def _sim_joint_control(self,random_generator=False, num=-1):
        time = self.warm_time
        self._initialize_vars(joint_control=True)


        print('start')
        while traci.simulation.getMinExpectedNumber()>0:

            traci.simulation.step(time=float(time))

            if time==self.test_begin or time==self.test_end:
                #画network流量热力图
                self._draw_density(time)

            self._reset_data()
            self._update_network_data()

            if self.check_saturated_condition(): #exist oversaturated region
                # 进行联合控制
                if not self.control_active:  # 只记录开始时间一次
                    self.control_active = True
                    with open(self.log_file, "a") as file:
                        file.write(f"Begin control: {time}\n")
                    self.begin_control_time=time
                self._joint_control(time-self.begin_control_time)
            else:
                if self.control_active:  # 只记录结束时间一次
                    self.control_active = False
                    with open(self.log_file, "a") as file:
                        file.write(f"End control: {time}\n")
                    self.restore_to_fixed_signal()

            self._save_data(time, termination=not self.control_active)

            if not self.control_active:
                if len(self.total_vehicles)<self.threshold_veh_num:
                    break

            time += self.mic_length
            self.step = int((time-self.warm_time)/self.mac_length)


        self._save_output('joint_control',random_generator, num)

    def _get_oversaturated_regions(self):
        oversaturated_regions = []
        for i in self.regions:
            if self.regions[i].n>=self.regions[i].critical*self.oversaturation_ratio:
                oversaturated_regions.append(i)
        return oversaturated_regions


    def _joint_control(self,time):
        self._get_data(time)
        self._get_completion_flow()

        if time % self.mac_length == 0:
            self.oversaturated_regions = self._get_oversaturated_regions()
            if self.oversaturated_regions:
                log_entry = f"time: {time+self.begin_control_time}, regions: {self.oversaturated_regions}\n"
                with open(self.log_file, "a") as file:
                    file.write(log_entry)


            # joint model
            self.cmax, self.cmin = self._get_c_range()
            print('model1')
            '''
            solution_p = self._perimeter_model1()
            if solution_p==-1:
                self.control_active=False
                return

            print('model2')
            self.b, self.c = self._perimeter_model2(solution_p)
            '''
            self.b, self.c = self._perimeter_model_total()

            self._get_expected_n(self.b, self.c)
            self._get_expected_m()

        # signal control

        mic_step=(time % self.mac_length) / self.mic_length
        print('signal')
        optimized_superphase = self._signal_control(mic_step)
        # optimized_superphase=self._signal_control(self.b)
        self._action_signal(optimized_superphase)
        # route choice
        print('route')
        self._route_choice(self.c)
        self._action_route()
        self._get_real_c()


    def _get_real_c(self):
        self.real_c = {}
        for i in self.regions:
            for j in self.regions[i].nd:
                if i==j:
                    continue
                total_num=len(self.regions[i].control_vehicles[j])
                for h in self.regions[i].neighbor:
                    v_num=0
                    for v in self.regions[i].control_vehicles[j]:
                        for r in range(len(v.routes)):
                            if v.route_choice==v.routes[r]:
                                next_region=v.next_regions[r]
                                break
                        if next_region==h:
                            v_num+=1
                    if total_num==0:
                        self.real_c['%d_%d_%d' % (i, h, j)]=0
                    else:
                        self.real_c['%d_%d_%d' % (i, h, j)] =v_num/total_num


    def _get_expected_m(self):
        self.M = {}
        # mac_length时段内的M
        for i in self.regions:
            for h in self.regions[i].neighbor:
                m = 0
                for j in self.regions[i].nd:
                    if i != j:
                        m += self.c['%d_%d_%d' % (i, h, j)] * self.b['u_%d_%d' % (i, h)] * self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow
                    print(55555,self.b['u_%d_%d' % (i, h)])
                self.M['%d_%d' % (i, h)] = m

    def _get_expected_n(self,b,c):

        for i in self.regions:
            N_bc=0
            for j in self.regions[i].nd:
                N_bc += self.regions[i].nd[j]+self.regions[i].q[j]
                for h in self.regions[i].neighbor:
                    # hij
                    if h!=j:
                        N_bc += self.regions[h].nd[j] / self.regions[h].n * self.regions[h].completion_flow*self.mac_length * b['u_%d_%d' % (h, i)]*c['%d_%d_%d' %(h, i, j)]
                    # ihj
                    if i!=j:
                        N_bc+=-self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow*self.mac_length* b['u_%d_%d' % (i, h)]*c['%d_%d_%d' % (i, h, j)]
            N_bc+= -self.regions[i].nd[i] / self.regions[i].n * self.regions[i].completion_flow*self.mac_length
            self.regions[i].n_p =N_bc
            self.regions[i].n_bc =N_bc

    def _get_completion_flow(self):
        for i in self.regions:
            self.regions[i].completion_flow=self._MFD_function(self.regions[i].n,self.regions[i].mfd_param)

    def _get_c_range(self):
        cmax={}
        cmin={}
        for i in self.regions:
            for j in self.regions[i].nd:
                if i==j:
                    continue
                for h in self.regions[i].neighbor:
                    max=0
                    min=0
                    total=len(self.regions[i].control_vehicles[j])

                    #total=self.regions[i].edge_vehicle_num[j]
                    if total==0:
                        cmax['%d_%d_%d' % (i, h, j)] =1.0
                        cmin['%d_%d_%d' % (i, h, j)]=0.0
                        continue

                    for v in self.regions[i].control_vehicles[j]:
                        #if v.boundary!=1:
                        #    continue
                        if h in v.next_regions:
                            max+=1
                            if len(v.next_regions)==1 or (v.next_regions[0]==h and v.next_regions[1]==h):
                                #只有一条路线 或者 两天路线的next_region 都是h
                                min+=1
                    cmax['%d_%d_%d' % (i, h, j)]=float(max/total)
                    cmin['%d_%d_%d' % (i, h, j)]=float(min/total)

        return cmax,cmin


    def _save_data(self,time,termination=False):

        self._save_network_data(time)
        for i in self.regions:
            # 当前的n
            data = {}
            data['region'] = i
            data['time'] = time
            data['real_n_t'] = self.regions[i].n  #x
            data['output_t-1'] = self.regions[i].output #y=output/step
            if termination:
                self.N_data.append(data)
                continue

            data['completionflow'] = self.regions[i].completion_flow
            data['critical_n'] = self.regions[i].critical
            if (time-self.begin_control_time) % self.mac_length == 0:
                # p带入后的 期望的n
                data['expected_n(p)_t+1'] = self.regions[i].n_p
                # b,c带入后的 期望的n
                data['expected_n(b,c)_t+1'] = self.regions[i].n_bc
            else:
                data['expected_n(p)_t+1'] = -1
                data['expected_n(b,c)_t+1'] = -1

            self.N_data.append(data)
        if termination:
            return


        for i in self.regions:
            for h in self.regions[i].neighbor:
                # b_ih
                data = {}
                data['time'] = time
                data['region_i'] = i
                data['region_h'] = h
                data['solution_M_t+1'] = self.M['%d_%d' % (i, h)]
                data['real_M_t'] = self.real_M_set['%d_%d' % (i, h)][-1]
                if (time-self.begin_control_time) % self.mac_length == 0:
                    data['M_max_t+1']=self.M_second_max['%d_%d' % (i, h)]
                    data['M_min_t+1']=self.M_second_min['%d_%d' % (i, h)]
                else:
                    data['M_max_t+1'] =-1
                    data['M_min_t+1']=-1
                data['b_t+1']=self.b['u_%d_%d' % (i, h)]

                self.b_data.append(data)
                # route
                for j in self.regions[i].nd:
                    if i == j:
                        # ihi不存在
                        continue
                    data = {}
                    data['time'] = time
                    data['region_i'] = i
                    data['region_h'] = h
                    data['region_j'] = j
                    # c_ihj
                    data['solution_c_t+1'] = self.c['%d_%d_%d' % (i, h, j)]
                    data['solution_cmax_t+1'] = self.cmax['%d_%d_%d' % (i, h, j)]
                    data['solution_cmin_t+1'] = self.cmin['%d_%d_%d' % (i, h, j)]
                    data['real_c_t+1'] = self.real_c['%d_%d_%d' % (i, h, j)]
                    self.c_data.append(data)

    '''
    def _perimeter_model1(self):
        index = {}
        temp = 0
        for i in self.regions:
            for h in self.regions[i].neighbor:
                for j in self.regions[i].nd:
                    if i == j:
                        # ihi不存在
                        continue
                    index['p_%d_%d_%d' % (i, h, j)] = temp
                    temp += 1
        var_num = temp

        p=np.zeros(var_num)

        for i in self.oversaturated_regions:
            N = []
            index_i = {}
            temp = 0
            for j in self.regions[i].nd:
                for h in self.regions[i].neighbor:
                    # hij
                    if h != j:
                        N.append(self.regions[h].nd[j] / self.regions[h].n * self.regions[
                            h].completion_flow * self.mac_length)
                        index_i[temp] = 'p_%d_%d_%d' % (h, i, j)
                        temp += 1

                    # ihj
                    if i != j:
                        N.append(-self.regions[i].nd[j] / self.regions[i].n * self.regions[
                            i].completion_flow * self.mac_length)
                        index_i[temp] = 'p_%d_%d_%d' % (i, h, j)
                        temp += 1

            N = np.array(N)

            # 目标系数 p 直接赋值
            for k in range(len(N)):
                p[index[index_i[k]]] += N[k]


        G = []
        H = []
        # capacity

        for i in self.regions:
            for h in self.regions[i].neighbor:
                inequation1 = np.zeros(var_num)
                inequation2 =  np.zeros(var_num)
                for j in self.regions[i].nd:
                    if i == j:
                        continue
                    inequation1[index['p_%d_%d_%d' % (i, h, j)]] += self.regions[i].nd[j] / self.regions[i].n * \
                                                                    self.regions[i].completion_flow
                    inequation2[index['p_%d_%d_%d' % (i, h, j)]] -= self.regions[i].nd[j] / self.regions[i].n * \
                                                                    self.regions[i].completion_flow

                G.append(inequation1)
                H.append(self.M_second_max['%d_%d' % (i, h)])
                G.append(inequation2)
                H.append(-self.M_second_min['%d_%d' % (i, h)])

        G = np.array(G)
        H = np.array(H)

        solution = linprog(p, A_ub=G, b_ub=H, method='highs')

        if not solution.success:  # 约束无解时输出 CSV
            c=[]
            n=[]
            for i in self.regions:
                c.append(self.regions[i].completion_flow)
                n.append(self.regions[i].n)
            print(33333,c)
            print(55555,n)

            # 生成列名
            column_names = ['Constraint_Index'] + list(index.keys()) + ['H']

            # 构造 DataFrame
            df_constraints = pd.DataFrame(G, columns=list(index.keys()))
            df_constraints.insert(0, 'Constraint_Index', range(1, len(G) + 1))  # 添加约束编号
            df_constraints['H'] = H  # 添加右端项 H

            # 保存到 CSV 文件
            csv_filename = "constraints.csv"
            df_constraints.to_csv(csv_filename, index=False)
            print(f"Constraints saved to {csv_filename}")

            # 正确抛出异常
            raise RuntimeError(f"Linear programming failed: {solution.message}")
            #return -1  # 返回 -1 代表无解


        index_reverse = {k: v for v, k in index.items()}
        solution_p = {}
        for i in range(len(solution.x)):  # 使用 solution.x 正确获取解
            if i in index_reverse:  # 确保索引存在
                solution_p[index_reverse[i]] = solution.x[i]
            else:
                print(f"Warning: index {i} not found in index_reverse")  # 调试信息

        return solution_p
    '''
    '''
    def _perimeter_model_total2(self):
        def build_vars():
            index = {}
            counter = 0
            for i in self.regions:
                for h in self.regions[i].neighbor:
                    index[f'b_{i}_{h}'] = counter
                    counter += 1
                    for j in self.regions[i].nd:
                        if i == j:
                            continue
                        index[f'{i}_{h}_{j}'] = counter
                        counter += 1
            return index

        def init_values(index):
            var_num = len(index)
            return np.zeros(var_num)

        def build_objective(index):
            def objective(x):
                total_squared_diff = 0
                for i in self.regions:
                    inflow = 0
                    outflow = 0
                    for j in self.regions[i].nd:
                        inflow += self.regions[i].nd[j] + self.regions[i].q[j]
                        for h in self.regions[i].neighbor:
                            if h != j:
                                c_idx = index[f'{h}_{i}_{j}']
                                b_idx = index[f'b_{h}_{i}']
                                coeff = self.regions[h].nd[j] / self.regions[h].n * self.regions[
                                    h].completion_flow * self.mac_length
                                inflow += x[b_idx] * x[c_idx] * coeff
                            if i != j:
                                c_idx = index[f'{i}_{h}_{j}']
                                b_idx = index[f'b_{i}_{h}']
                                coeff = self.regions[i].nd[j] / self.regions[i].n * self.regions[
                                    i].completion_flow * self.mac_length
                                outflow += x[b_idx] * x[c_idx] * coeff
                    outflow += self.regions[i].nd[i] / self.regions[i].n * self.regions[
                        i].completion_flow * self.mac_length

                    N_diff = inflow - outflow - self.regions[i].critical
                    total_squared_diff += N_diff ** 2

                return total_squared_diff

            return objective

        def build_bounds(index):
            var_num = len(index)
            bounds = [(0, None)] * var_num  # 默认所有变量非负

            for i in self.regions:
                for h in self.regions[i].neighbor:
                    for j in self.regions[i].nd:
                        if i == j:
                            continue
                        var_name = f'{i}_{h}_{j}'
                        if var_name in index:
                            cmin = self.cmin[var_name]
                            cmax = self.cmax[var_name]
                            bounds[index[var_name]] = (cmin, cmax)

            return bounds

        def build_constraints(index) -> List[Dict[str, Any]]:
            constraints = []

            # 约束 1：每个 region 的 total flow - outflow - critical + z >= 0
            for i in self.regions:
                def region_constraint(x, i=i):
                    z = x[index['z']]
                    inflow = 0
                    outflow = 0
                    for j in self.regions[i].nd:
                        inflow += self.regions[i].nd[j] + self.regions[i].q[j]
                        for h in self.regions[i].neighbor:
                            if h != j:
                                c_idx = index[f'{h}_{i}_{j}']
                                b_idx = index[f'b_{h}_{i}']
                                coeff = self.regions[h].nd[j] / self.regions[h].n * self.regions[
                                    h].completion_flow * self.mac_length
                                inflow += x[b_idx] * x[c_idx] * coeff
                            if i != j:
                                c_idx = index[f'{i}_{h}_{j}']
                                b_idx = index[f'b_{i}_{h}']
                                coeff = self.regions[i].nd[j] / self.regions[i].n * self.regions[
                                    i].completion_flow * self.mac_length
                                outflow += x[b_idx] * x[c_idx] * coeff
                    outflow += self.regions[i].nd[i] / self.regions[i].n * self.regions[
                        i].completion_flow * self.mac_length
                    return z - (inflow - outflow - self.regions[i].critical)

                constraints.append({'type': 'ineq', 'fun': region_constraint})

            # 约束 2：每对(i,h)的最大最小 M_second 约束
            for i in self.regions:
                for h in self.regions[i].neighbor:
                    def upper_flow_constraint(x, i=i, h=h):
                        total = 0
                        for j in self.regions[i].nd:
                            if i == j:
                                continue
                            c_idx = index[f'{i}_{h}_{j}']
                            b_idx = index[f'b_{i}_{h}']
                            coeff = self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow
                            total += x[c_idx] * x[b_idx] * coeff
                        return self.M_second_max[f'{i}_{h}'] - total

                    def lower_flow_constraint(x, i=i, h=h):
                        total = 0
                        for j in self.regions[i].nd:
                            if i == j:
                                continue
                            c_idx = index[f'{i}_{h}_{j}']
                            b_idx = index[f'b_{i}_{h}']
                            coeff = self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow
                            total += x[c_idx] * x[b_idx] * coeff
                        return total - self.M_second_min[f'{i}_{h}']

                    constraints.append({'type': 'ineq', 'fun': upper_flow_constraint})
                    constraints.append({'type': 'ineq', 'fun': lower_flow_constraint})

            # 约束 3：路径选择比例之和为 1
            for i in self.regions:
                for j in self.regions[i].nd:
                    if i == j:
                        continue

                    def route_choice_sum(x, i=i, j=j):
                        total = 0
                        for h in self.regions[i].neighbor:
                            c_idx = index[f'{i}_{h}_{j}']
                            total += x[c_idx]
                        return total - 1.0

                    constraints.append({'type': 'eq', 'fun': route_choice_sum})

            return constraints

        def decode_solution(index, solution):
            print('s', solution)
            print('i', index)
            index_reverse = {v: k for k, v in index.items()}
            b = {}
            c = {}

            for i, name in index_reverse.items():
                if name.startswith('b_'):
                    b['u' + name[1:]] = solution[i]
                else:
                    c[name] = solution[i]
            return b, c

        from scipy.optimize import minimize

        index = build_vars()
        x0 = init_values(index)
        objective = build_objective(index)
        bounds = build_bounds(index)
        cons = build_constraints(index)

        result = minimize(
            fun=objective,
            x0=x0,
            bounds=bounds,
            method='SLSQP',
            constraints=cons
        )

        b, c = decode_solution(index, result.x)
        return b, c
    '''

    def _perimeter_model_total(self):
        def build_vars():
            index = {}
            counter = 0
            for i in self.regions:
                for h in self.regions[i].neighbor:
                    index[f'b_{i}_{h}'] = counter
                    counter += 1
                    for j in self.regions[i].nd:
                        if i == j:
                            continue
                        index[f'{i}_{h}_{j}'] = counter
                        counter += 1
            index['z'] = counter
            return index

        def init_values(index):
            var_num=len(index)
            init_value= [0] * var_num  # 默认所有变量非负
            for i in self.regions:
                for h in self.regions[i].neighbor:
                    var_name = f'b_{i}_{h}'
                    if var_name in index:
                        init_value[index[var_name]] = 1
            return init_value



        def build_objective(index):
            def objective(x):
                return x[index['z']]
            return objective

        def build_bounds(index):
            var_num = len(index)
            bounds = [(0, None)] * var_num  # 默认所有变量非负

            for i in self.regions:
                for h in self.regions[i].neighbor:
                    for j in self.regions[i].nd:
                        if i == j:
                            continue
                        var_name = f'{i}_{h}_{j}'
                        if var_name in index:
                            cmin = self.cmin[var_name]
                            cmax = self.cmax[var_name]
                            bounds[index[var_name]] = (cmin, cmax)

            return bounds

        def build_constraints(index) -> List[Dict[str, Any]]:
            constraints = []

            # 约束 1：每个 region 的 total flow - outflow - critical + z >= 0
            for i in self.regions:
                def region_constraint(x, i=i):
                    z = x[index['z']]
                    inflow = 0
                    outflow = 0
                    for j in self.regions[i].nd:
                        inflow += self.regions[i].nd[j] + self.regions[i].q[j]
                        for h in self.regions[i].neighbor:
                            if h != j:
                                c_idx = index[f'{h}_{i}_{j}']
                                b_idx = index[f'b_{h}_{i}']
                                coeff = self.regions[h].nd[j] / self.regions[h].n * self.regions[
                                    h].completion_flow * self.mac_length
                                inflow += x[b_idx] * x[c_idx] * coeff
                            if i != j:
                                c_idx = index[f'{i}_{h}_{j}']
                                b_idx = index[f'b_{i}_{h}']
                                coeff = self.regions[i].nd[j] / self.regions[i].n * self.regions[
                                    i].completion_flow * self.mac_length
                                outflow += x[b_idx] * x[c_idx] * coeff
                    outflow += self.regions[i].nd[i] / self.regions[i].n * self.regions[
                        i].completion_flow * self.mac_length
                    return z -(inflow - outflow - self.regions[i].critical)

                constraints.append({'type': 'ineq', 'fun': region_constraint})

            # 约束 2：每对(i,h)的最大最小 M_second 约束
            for i in self.regions:
                for h in self.regions[i].neighbor:
                    def upper_flow_constraint(x, i=i, h=h):
                        total = 0
                        for j in self.regions[i].nd:
                            if i == j:
                                continue
                            c_idx = index[f'{i}_{h}_{j}']
                            b_idx = index[f'b_{i}_{h}']
                            coeff = self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow
                            total += x[c_idx] * x[b_idx] * coeff
                        return self.M_second_max[f'{i}_{h}'] - total

                    def lower_flow_constraint(x, i=i, h=h):
                        total = 0
                        for j in self.regions[i].nd:
                            if i == j:
                                continue
                            c_idx = index[f'{i}_{h}_{j}']
                            b_idx = index[f'b_{i}_{h}']
                            coeff = self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow
                            total += x[c_idx] * x[b_idx] * coeff
                        return total - self.M_second_min[f'{i}_{h}']

                    constraints.append({'type': 'ineq', 'fun': upper_flow_constraint})
                    constraints.append({'type': 'ineq', 'fun': lower_flow_constraint})

            # 约束 3：路径选择比例之和为 1
            for i in self.regions:
                for j in self.regions[i].nd:
                    if i == j:
                        continue

                    def route_choice_sum(x, i=i, j=j):
                        total = 0
                        for h in self.regions[i].neighbor:
                            c_idx = index[f'{i}_{h}_{j}']
                            total += x[c_idx]
                        return total - 1.0

                    constraints.append({'type': 'eq', 'fun': route_choice_sum})

            return constraints

        def decode_solution(index, solution):
            print('s',solution)
            print('i',index)
            index_reverse = {v: k for k, v in index.items()}
            b = {}
            c = {}

            for i, name in index_reverse.items():
                if name.startswith('b_'):
                    b['u' + name[1:]] = solution[i]
                elif name == 'z':
                    continue  # z 是目标函数，不属于 b 或 c
                else:
                    c[name] = solution[i]
            return b, c


        from scipy.optimize import minimize

        index = build_vars()
        x0 = init_values(index)
        objective = build_objective(index)
        bounds = build_bounds(index)
        cons = build_constraints(index)

        result=minimize_with_relaxation(objective, x0, cons, bounds, slack_weight=100.0, method='SLSQP',
                             verbose=True)
        b,c=decode_solution(index, result)

        '''
        result = minimize(
            fun=objective,
            x0=x0,
            bounds=bounds,
            method="SLSQP",
            constraints=cons
        )

        print(33333,result.success)
        b,c=decode_solution(index, result.x)
        '''
        return b, c

    def _perimeter_model1(self):

        index = {}
        temp = 0
        for i in self.regions:
            for h in self.regions[i].neighbor:
                for j in self.regions[i].nd:
                    if i==j:
                        #ihi不存在
                        continue
                    index['p_%d_%d_%d' % (i, h, j)] = temp
                    temp += 1
        var_num = temp

        p = [0 for i in range(var_num)]
        Q = [[0 for i in range(var_num)] for j in range(var_num)]
        for i in self.oversaturated_regions:
            N=[]
            index_i = {}
            temp = 0
            constant = 0
            for j in self.regions[i].nd:
                constant += self.regions[i].nd[j]
                constant += self.regions[i].q[j]
                for h in self.regions[i].neighbor:
                    # hij
                    if h!=j:
                        N.append(self.regions[h].nd[j] / self.regions[h].n * self.regions[h].completion_flow*self.mac_length)
                        index_i[temp] = 'p_%d_%d_%d' % (h, i, j)
                        temp += 1

                    # ihj
                    if i!=j:
                        N.append(-self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow*self.mac_length)
                        index_i[temp] = 'p_%d_%d_%d' % (i, h, j)
                        temp += 1
            constant += -self.regions[i].nd[i] / self.regions[i].n * self.regions[i].completion_flow*self.mac_length
            constant += -self.regions[i].critical

            N = np.asmatrix(N)
            p0 = 2 * N * constant  # 平方的一次
            p0 = p0.tolist()[0]
            num=len(p0)
            for k in range(num):
                p[index[index_i[k]]] += p0[k]

            Q0 = 2 * N.T * N  # 平方的二次
            Q0 = Q0.tolist()
            for k in range(num):
                for m in range(num):
                    Q[index[index_i[k]]][index[index_i[m]]] += Q0[k][m]

        G = []
        H = []
        # capacity

        for i in self.regions:
            for h in self.regions[i].neighbor:
                inequation1 = [0 for i in range(var_num)]
                inequation2 = [0 for i in range(var_num)]
                for j in self.regions[i].nd:
                    if i == j:
                        continue
                    inequation1[index['p_%d_%d_%d' % (i, h, j)]] +=self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow
                    inequation2[index['p_%d_%d_%d' % (i, h, j)]] -=self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow

                G.append(inequation1)
                H.append(self.M_second_max['%d_%d' % (i, h)])
                G.append(inequation2)
                H.append(-self.M_second_min['%d_%d' % (i, h)])

        Q=np.array(Q)
        p=np.array(p)
        G=np.array(G)
        H=np.array(H)
        solution=solve_qp(Q,p,G,H,solver='osqp')

        if solution is None or len(solution) == 0:
            c = []
            n = []
            for i in self.regions:
                c.append(self.regions[i].completion_flow)
                n.append(self.regions[i].n)

            # 生成列名
            column_names = ['Constraint_Index'] + list(index.keys()) + ['H']

            # 构造 DataFrame
            df_constraints = pd.DataFrame(G, columns=list(index.keys()))
            df_constraints.insert(0, 'Constraint_Index', range(1, len(G) + 1))  # 添加约束编号
            df_constraints['H'] = H  # 添加右端项 H

            # 保存到 CSV 文件
            csv_filename = "constraints.csv"
            df_constraints.to_csv(csv_filename, index=False)
            print(f"Constraints saved to {csv_filename}")

            # 正确抛出异常
            raise RuntimeError(f"Linear programming failed: {solution.message}")
            # return -1  # 返回 -1 代表无解


        index_reverse={k:v for v,k in index.items()}
        solution_p={}
        for i in range(len(index_reverse)):
            solution_p[index_reverse[i]]=solution[i]

        return solution_p

    def _perimeter_model1_1(self):
        #for results 1.0 2.0
        index = {}
        temp = 0
        bounds = []  # 用于存储变量范围

        for i in self.regions:
            for h in self.regions[i].neighbor:
                for j in self.regions[i].nd:
                    if i==j:
                        #ihi不存在
                        continue
                    index['p_%d_%d_%d' % (i, h, j)] = temp
                    bounds.append((0,None))  #666 0，1 4.0
                    temp += 1
        index['z']=temp
        bounds.append((None,None))
        temp += 1

        var_num = temp
        self.coefficient=0.05 #666 1.0 0.001
        p = [0 for i in range(var_num)]
        G = []
        H = []
        for i in self.regions:
            p[index['z']] = 1
            inequation0 =  [0 for i in range(var_num)]
            inequation0 [index['z']] =-1
            constant=-self.regions[i].critical
            for j in self.regions[i].nd:
                constant += self.regions[i].nd[j]
                constant += self.regions[i].q[j]
                for h in self.regions[i].neighbor:
                    # hij
                    if h != j:
                        temp=self.regions[h].nd[j] / self.regions[h].n * self.regions[
                            h].completion_flow * self.mac_length
                        inequation0[index['p_%d_%d_%d' % (h, i, j)]]+=temp
                        p[index['p_%d_%d_%d' % (h, i, j)]]+=self.coefficient*temp

                    # ihj
                    if i != j:
                        temp=self.regions[i].nd[j] / self.regions[i].n * self.regions[
                            i].completion_flow * self.mac_length
                        inequation0[index['p_%d_%d_%d' % (i, h, j)]]-=temp
                        p[index['p_%d_%d_%d' % (i, h, j)]]-=self.coefficient*temp

            constant -= self.regions[i].nd[i] / self.regions[i].n * self.regions[i].completion_flow * self.mac_length
            G.append(inequation0)
            H.append(-constant)

        for i in self.regions:
            for h in self.regions[i].neighbor:
                inequation1 = [0 for i in range(var_num)]
                inequation2 = [0 for i in range(var_num)]
                for j in self.regions[i].nd:
                    if i == j:
                        continue
                    inequation1[index['p_%d_%d_%d' % (i, h, j)]] += self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow
                    inequation2[index['p_%d_%d_%d' % (i, h, j)]] -= self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow

                G.append(inequation1)
                H.append(self.M_second_max['%d_%d' % (i, h)])
                G.append(inequation2)
                H.append(-min(0.5,self.M_second_min['%d_%d' % (i, h)]))

        p=np.array(p)
        G=np.array(G)
        H=np.array(H)
        n=len(self.regions)
        # 调用修改后的求解函数
        solution = solve_with_relaxation(n, p, G, H, bounds)

        # 处理无解情况
        if solution is None:
            c = []
            n = []
            for i in self.regions:
                c.append(self.regions[i].completion_flow)
                n.append(self.regions[i].n)
            print(33333, c)
            print(55555, n)

            # 生成列名
            column_names = ['Constraint_Index'] + list(index.keys()) + ['H']

            # 构造 DataFrame
            df_constraints = pd.DataFrame(G, columns=list(index.keys()))
            df_constraints.insert(0, 'Constraint_Index', range(1, len(G) + 1))  # 添加约束编号
            df_constraints['H'] = H  # 添加右端项 H

            # 保存到 CSV 文件
            csv_filename = "constraints.csv"
            df_constraints.to_csv(csv_filename, index=False)
            print(f"Constraints saved to {csv_filename}")

            # 正确抛出异常
            raise RuntimeError("Linear programming still infeasible after relaxation.")
            # return -1  # 返回 -1 代表无解

        index_reverse = {k: v for v, k in index.items()}
        solution_p = {}
        for i in range(len(solution.x)):  # 使用 solution.x 正确获取解
            if i in index_reverse:  # 确保索引存在
                solution_p[index_reverse[i]] = solution.x[i]
            else:
                print(f"Warning: index {i} not found in index_reverse")  # 调试信息

        return solution_p


    def _perimeter_model2(self,solution_p):
        index = {}
        temp = 0
        for i in self.regions:
            for h in self.regions[i].neighbor:
                #F_ih
                index['f_%d_%d' % (i, h)] = temp
                temp+=1
                for j in self.regions[i].nd:
                    if i == j:
                        # ihi不存在
                        continue
                    #c_ihj
                    index['%d_%d_%d' % (i, h, j)] = temp
                    temp += 1
        var_num = temp

        #F*P=c
        p = [0.0 for i in range(var_num)]
        Q = [[0 for i in range(var_num)] for j in range(var_num)]
        for i in self.regions:
            for j in self.regions[i].nd:
                if i==j:
                    continue
                for h in self.regions[i].neighbor:
                    Q[index['%d_%d_%d' % (i, h, j)]][index['%d_%d_%d' % (i, h, j)]]+=2.0
                    Q[index['f_%d_%d' % (i, h)]][index['f_%d_%d' % (i, h)]] += 2.0*solution_p['p_%d_%d_%d' % (i, h, j)]**2
                    Q[index['%d_%d_%d' % (i, h, j)]][index['f_%d_%d' % (i, h)]]+=-2.0*solution_p['p_%d_%d_%d' % (i, h, j)]
                    Q[index['f_%d_%d' % (i, h)]][index['%d_%d_%d' % (i, h, j)]]+=-2.0*solution_p['p_%d_%d_%d' % (i, h, j)]

        # routechoice_proportion=1
        A = []
        b = []
        for i in self.regions:
            for j in self.regions[i].nd:
                if i==j:
                    continue
                equation = [0 for i in range(var_num)]
                for h in self.regions[i].neighbor:
                    equation[index['%d_%d_%d' % (i, h, j)]] = 1.0
                A.append(equation)
                b.append(1.0)

        G = []
        H = []
        for i in self.regions:
            for h in self.regions[i].neighbor:
                # F>1 -F<-1
                #b<bmax F>1/bmax -F<-1/bmax
                for j in self.regions[i].nd:
                    if i == j:
                        continue
                    # cmin<c<cmax
                    inequation = [0 for i in range(var_num)]
                    inequation[index['%d_%d_%d' % (i, h, j)]] = 1.0
                    G.append(inequation)
                    H.append(self.cmax['%d_%d_%d' % (i, h, j)])

                    inequation = [0 for i in range(var_num)]
                    inequation[index['%d_%d_%d' % (i, h, j)]] = -1.0
                    G.append(inequation)
                    H.append(-self.cmin['%d_%d_%d' % (i, h, j)])

                inequation1 = [0 for i in range(var_num)]
                inequation2 = [0 for i in range(var_num)]
                for j in self.regions[i].nd:
                    if i == j:
                        continue
                    inequation1[index['%d_%d_%d' % (i, h, j)]] +=self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow
                    inequation2[index['%d_%d_%d' % (i, h, j)]] -=self.regions[i].nd[j] / self.regions[i].n * self.regions[i].completion_flow

                inequation1[index['f_%d_%d' % (i, h)]] -=self.M_second_max['%d_%d' % (i, h)]
                inequation2[index['f_%d_%d' % (i, h)]] +=self.M_second_min['%d_%d' % (i, h)]

                G.append(inequation1)
                H.append(0)
                G.append(inequation2)
                H.append(0)

        Q = np.array(Q)
        p = np.array(p)
        G = np.array(G)
        H = np.array(H)
        A = np.array(A)
        b = np.array(b)

        solution = solve_and_store(Q, p, G, H, A, b)
        if solution is None:
            self.save_matrices(Q, p, G, H, A, b)

        index_reverse = {k: v for v, k in index.items()}
        b = {}
        c = {}
        for i in range(len(index_reverse)):
            if index_reverse[i][0] == 'f':
                b['u'+index_reverse[i][1:]] = float(1/solution[i])
            else:
                c[index_reverse[i]] = solution[i]

        return b, c


    def save_matrices(self,Q, p, G, H, A, b):
        # 创建一个目录存储矩阵
        if not os.path.exists('qp_matrices'):
            os.makedirs('qp_matrices')

        # 将矩阵保存为文本文件，便于后续分析
        np.savetxt(f'qp_matrices/Q.txt', Q, delimiter=',')
        np.savetxt(f'qp_matrices/p.txt', p, delimiter=',')
        np.savetxt(f'qp_matrices/G.txt', G, delimiter=',')
        np.savetxt(f'qp_matrices/H.txt', H, delimiter=',')
        if A is not None and b is not None:
            np.savetxt(f'qp_matrices/A.txt', A, delimiter=',')
            np.savetxt(f'qp_matrices/b.txt', b, delimiter=',')

    def _signal_control_version1(self):
        self.tl_M = {}
        # mic_length时段内的tl_M
        for i in self.regions:
            for h in self.regions[i].neighbor:
                n = len(self.previous_M['%d_%d' % (i, h)])
                M = (self.M['%d_%d' % (i, h)] * self.mac_length - sum(
                    self.previous_M['%d_%d' % (i, h)] * self.mic_length)) / (self.mac_length - self.mic_length * n)
                self.tl_M['%d_%d' % (i, h)] = M - self.previous_notl_M['%d_%d' % (i, h)]

        # 确定符合signal_g的superphase集合
        region_temp = []
        phaseset = {}
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                if len(self.regions[i].toregion_superphaseset[h]) == 0:
                    # ih边界没有tl的交叉口
                    phaseset['%d_%d' % (i, h)] = -1
                    continue

                # phaseset应该是双向的 i,h边界的交叉口
                phaseset['%d_%d' % (i, h)] = []
                sumgapset = []
                temp = 0
                for id in range(len(self.regions[i].toregion_superphaseset[h])):
                    num = self.regions[i].toregion_superphasedic[h][id][0]
                    reversed_num = self.regions[i].toregion_superphasedic[h][id][1]
                    gap = (self.tl_M['%d_%d' % (i, h)] - num) / self.tl_M['%d_%d' % (i, h)]
                    reversed_gap = (self.tl_M['%d_%d' % (h, i)] - reversed_num) / self.tl_M['%d_%d' % (h, i)]
                    if gap < 0:
                        gap = -gap
                    if reversed_gap < 0:
                        reversed_gap = -reversed_gap
                    if gap <= self.range_M and reversed_gap <= self.range_M:
                        phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])
                        temp += 1
                    sumgapset.append(gap + reversed_gap)

                if temp == 0:
                    # 没有满足约束的phaseset
                    temp1 = sorted(sumgapset)
                    # 取gap最小的10个
                    value = temp1[:10]
                    for v in value:
                        id = sumgapset.index(v)
                        phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])
                if temp > 100:
                    # 存在超过一百个multi-phase
                    phaseset['%d_%d' % (i, h)] = []
                    temp1 = sorted(sumgapset)
                    # 取gap最小的100个
                    value = temp1[:100]
                    for v in value:
                        id = sumgapset.index(v)
                        phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])

        # calculate_weight
        weight_set = {}
        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                for tl in self.regions[i].toregion_trafficlight[h]:
                    weight_set[tl] = {}
                    for p in range(len(self.trafficlights[tl].phase)):
                        phase = self.trafficlights[tl].phase[p]
                        weight_set[tl][phase] = self._calculate_weight(tl, phase)

        optimized_superphase = {}
        # 在phaseset中选择最大压的
        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                if phaseset['%d_%d' % (i, h)] == -1:
                    optimized_superphase['%d_%d' % (i, h)] = -1
                    continue

                weight = {}
                superphaseset = {}
                index = 0
                for superphase in phaseset['%d_%d' % (i, h)]:
                    # super_phase[tl] = phase
                    index += 1
                    weight[index] = 0
                    superphaseset[index] = superphase

                    for tl in superphase:
                        weight[index] += weight_set[tl][superphase[tl]]

                for key, value in weight.items():
                    if value == max(weight.values()):
                        optimized_superphase['%d_%d' % (i, h)] = superphaseset[key]
                        break
        return optimized_superphase

    def _signal_control(self,mic_step):
        print(444,mic_step)
        total_mic_step = self.mac_length/self.mic_length
        range_M_strict = self.range_M_strict* (total_mic_step-mic_step)
        range_M_relaxed = self.range_M_relaxed* (total_mic_step-mic_step)

        self.tl_M = {}
        # mic_length时段内的tl_M
        for i in self.regions:
            for h in self.regions[i].neighbor:
                n = len(self.previous_M['%d_%d' % (i, h)])
                M = (self.M['%d_%d' % (i, h)] * self.mac_length - sum(
                    self.previous_M['%d_%d' % (i, h)] * self.mic_length)) / (self.mac_length - self.mic_length * n)
                self.tl_M['%d_%d' % (i, h)] = M - self.previous_notl_M['%d_%d' % (i, h)]

        # 确定符合signal_g的superphase集合
        region_temp = []
        phaseset = {}
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                if len(self.regions[i].toregion_superphaseset[h]) == 0:
                    # ih边界没有tl的交叉口
                    phaseset['%d_%d' % (i, h)] = -1
                    continue

                # phaseset应该是双向的 i,h边界的交叉口
                phaseset['%d_%d' % (i, h)] = []
                sumgapset = []
                temp = 0
                for id in range(len(self.regions[i].toregion_superphaseset[h])):
                    num = self.regions[i].toregion_superphasedic[h][id][0]
                    reversed_num = self.regions[i].toregion_superphasedic[h][id][1]
                    gap = (self.tl_M['%d_%d' % (i, h)] - num) / self.tl_M['%d_%d' % (i, h)]
                    reversed_gap = (self.tl_M['%d_%d' % (h, i)] - reversed_num) / self.tl_M['%d_%d' % (h, i)]
                    if gap < 0:
                        gap = -gap
                    if reversed_gap < 0:
                        reversed_gap = -reversed_gap

                    if i in self.oversaturated_regions:
                        if h in self.oversaturated_regions:
                            if gap <= range_M_strict and reversed_gap <= range_M_strict:
                                phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])
                                temp += 1
                        else:
                            if gap <= range_M_relaxed and reversed_gap <= range_M_strict:
                                # h->i should be strict
                                phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])
                                temp += 1
                    else:
                        if h in self.oversaturated_regions:
                            if gap <= range_M_strict and reversed_gap <= range_M_relaxed:
                                # i->h should be strict
                                phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])
                                temp += 1
                        else:
                            if gap <= range_M_relaxed and reversed_gap <= range_M_relaxed:
                                phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])
                                temp += 1

                    sumgapset.append(gap + reversed_gap)

                if temp == 0:
                    # 没有满足约束的phaseset
                    print('non satisfied')
                    temp1 = sorted(sumgapset)
                    # 取gap最小的10个
                    value = temp1[:10]
                    for v in value:
                        id = sumgapset.index(v)
                        phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])
                if temp > 0 and temp <= 100:
                    print('satisfied within 100',temp)
                if temp > 100:
                    print('satisfied exceed 100',temp)
                    '''
                    # 存在超过一百个multi-phase
                    phaseset['%d_%d' % (i, h)] = []
                    temp1 = sorted(sumgapset)
                    # 取gap最小的100个
                    value = temp1[:100]
                    for v in value:
                        id = sumgapset.index(v)
                        phaseset['%d_%d' % (i, h)].append(self.regions[i].toregion_superphaseset[h][id])
                    '''
        # calculate_weight
        weight_set = {}
        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                for tl in self.regions[i].toregion_trafficlight[h]:
                    weight_set[tl] = {}
                    for p in range(len(self.trafficlights[tl].phase)):
                        phase = self.trafficlights[tl].phase[p]
                        weight_set[tl][phase] = self._calculate_weight(tl, phase)

        optimized_superphase = {}
        # 在phaseset中选择最大压的
        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                if phaseset['%d_%d' % (i, h)] == -1:
                    optimized_superphase['%d_%d' % (i, h)] = -1
                    continue

                weight = {}
                superphaseset = {}
                index = 0
                for superphase in phaseset['%d_%d' % (i, h)]:
                    # super_phase[tl] = phase
                    index += 1
                    weight[index] = 0
                    superphaseset[index] = superphase

                    for tl in superphase:
                        weight[index] += weight_set[tl][superphase[tl]]

                for key, value in weight.items():
                    if value == max(weight.values()):
                        optimized_superphase['%d_%d' % (i, h)] = superphaseset[key]
                        break
        return optimized_superphase

    def _action_signal(self, optimized_superphase):
        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                superphase = optimized_superphase['%d_%d' % (i, h)]
                if superphase == -1:
                    continue
                for tl in superphase:
                    traci.trafficlight.setRedYellowGreenState(tl, superphase[tl])
                    traci.trafficlight.setPhaseDuration(tl, self.mic_length)

    def _route_choice(self,c):

        self.big_num=1000
        for i in self.regions:
            sum=0
            for j in self.regions[i].edge_list:
                edge=self.edges[j]
                sum+=edge.length*edge.numlane
            self.regions[i].ave_density=self.regions[i].n/sum

        for i in self.regions:
            index = {}
            temp = 0
            for j in self.regions[i].nd:
                for v in self.regions[i].control_vehicles[j]:
                    for r in range(len(v.routes)):
                        index['%s_%s_%s' % (j,v.id,r)] = temp
                        temp += 1
            var_num = temp
            p = [0.0 for i in range(var_num)]
            Q = [[0.0 for i in range(var_num)] for j in range(var_num)]

            for e in self.regions[i].edge_list:
                edge=self.edges[e]
                N=[]
                index2={}
                temp=0
                constant=-self.regions[i].ave_density
                for j in self.regions[i].nd:
                    for v in self.regions[i].control_vehicles[j]:
                        for r in range(len(v.routes)):
                            route=v.routes[r]
                            if len(route)>=3 and route[2]==edge:
                                #下一个link为edge
                                N.append(float(1/(edge.length*edge.numlane)))
                                index2[temp]='%s_%s_%s' % (j,v.id,r)
                                temp+=1

                N = np.asmatrix(N)
                p0 = 2 * N * constant  # 平方的一次
                p0 = p0.tolist()[0]
                num = len(p0)

                for k in range(num):
                    p[index[index2[k]]] += p0[k]

                Q0 = 2 * N.T * N  # 平方的二次
                Q0 = Q0.tolist()
                for k in range(num):
                    for m in range(num):
                        Q[index[index2[k]]][index[index2[m]]] += Q0[k][m]

            # r/num-c
            for j in self.regions[i].nd:
                if i == j:
                    continue
                #if self.regions[i].edge_vehicle_num[j] != 0:
                 #   continue
                for h in self.regions[i].neighbor:
                    N = []
                    index3 = {}
                    temp = 0
                    constant = -c['%d_%d_%d' % (i, h, j)]
                    for v in self.regions[i].control_vehicles[j]:
                        #if v.boundary != 1:
                        #    # 不是边界车辆
                        #    continue
                        total=len(self.regions[i].control_vehicles[j])
                        for r in range(len(v.routes)):
                            next_region = v.next_regions[r]
                            if next_region == h:
                                N.append(float(
                                    1 / total))
                                index3[temp] = '%s_%s_%s' % (j, v.id, r)
                                temp += 1
                    N = np.asmatrix(N)
                    p0 = 2 * N * constant  # 平方的一次
                    p0 = p0.tolist()[0]
                    num = len(p0)

                    for k in range(num):
                        p[index[index3[k]]] += self.big_num*p0[k]

                    Q0 = 2 * N.T * N  # 平方的二次
                    Q0 = Q0.tolist()
                    for k in range(num):
                        for m in range(num):
                            Q[index[index3[k]]][index[index3[m]]] += self.big_num*Q0[k][m]


            # proportion=1
            A = []
            b = []
            for j in self.regions[i].nd:
                for v in self.regions[i].control_vehicles[j]:
                    equation = [0 for i in range(var_num)]
                    for r in range(len(v.routes)):
                        equation[index['%s_%s_%s' % (j,v.id,r)]] = 1.0
                    A.append(equation)
                    b.append(1.0)
            G = []
            H = []

            # 0<R<1
            for j in self.regions[i].nd:
                for v in self.regions[i].control_vehicles[j]:
                    if len(v.routes) == 1:
                        continue
                    for r in range(len(v.routes)):
                        inequation = [0 for i in range(var_num)]
                        inequation[index['%s_%s_%s' % (j,v.id,r)]] = -1.0
                        G.append(inequation)
                        H.append(0.0)

            Q = np.array(Q)
            p = np.array(p)
            G = np.array(G)
            H = np.array(H)
            A = np.array(A)
            b = np.array(b)

            solution = solve_qp(Q, p, G, H,A,b ,solver='osqp')

            for j in self.regions[i].nd:
                for v in self.regions[i].control_vehicles[j]:
                    temp=0
                    a=0
                    for r in range(len(v.routes)):
                        #print(solution[index['%s_%s_%s' % (j,v.id,r)]])
                        if solution[index['%s_%s_%s' % (j,v.id,r)]]>temp:
                            temp=solution[index['%s_%s_%s' % (j,v.id,r)]]
                            v.route_choice=v.routes[r]
                        a+=solution[index['%s_%s_%s' % (j,v.id,r)]]

    def _action_route(self):
        for i in self.regions:
            for j in self.regions[i].nd:
                for v in self.regions[i].control_vehicles[j]:
                    if len(v.routes)>1 and traci.vehicle.getRoute(v.id)!=v.route_choice:
                        #print(1234567890)
                        #print(traci.vehicle.getRoute(v.id))
                        #print(v.route_choice)
                        traci.vehicle.setRoute(v.id,v.route_choice)

class only_perimeter_control(joint_control):
    def __init__(self,_with_logit_route_choice=False,load_mfd=True):
        # 调用父类 control 的构造函数
        super().__init__(load_mfd=load_mfd)
        if _with_logit_route_choice:
            self.save_path= 'control/only_perimeter_control_with_logit_route_choice'
        else:
            self.save_path= 'control/only_perimeter_control'

        self._with_logit_route_choice=_with_logit_route_choice
        self.M_last_mac_step={}
        self.K_P=0.05
        self.K_I=0.005
    def run(self):
        self._init()
        # 获取分区的inputlane,outputlane,bound等
        self.regions = self._region_nodelist(label=False, regions=self.regions)
        self.regions = self._region_demands(regions=self.regions,
                                                   begin=self.warm_time, end=100000)
        self._sim_onlyperimeter_control()
        self._terminate()
    def _sim_onlyperimeter_control(self):

        time = self.warm_time
        self._initialize_vars(only_perimeter_control=True)

        print('start')
        while traci.simulation.getMinExpectedNumber()>0:
            traci.simulation.step(time=float(time))
            self._reset_data()
            self._update_network_data()

            if self.check_saturated_condition():
                #exist oversaturated region
                if not self.control_active:
                    self.control_active = True
                    with open(self.log_file, "a") as file:
                        file.write(f"Begin control: {time}\n")
                    self.begin_control_time = time
                self._only_perimeter_control(time - self.begin_control_time)
                if self._with_logit_route_choice:
                    self._action_logit_based_route()
            else:
                if self.control_active:  # 只记录结束时间一次
                    self.control_active = False
                    with open(self.log_file, "a") as file:
                        file.write(f"End control: {time}\n")
                    self.restore_to_fixed_signal()

            self._save_data(time)

            if not self.control_active:
                if len(self.total_vehicles) < self.threshold_veh_num:
                    break
            if time > self.end_time_threshold:
                break
            time += self.mic_length
            self.step = int((time-self.warm_time)/self.mac_length)


        if self._with_logit_route_choice:
            self._save_output('only_perimeter_control_with_logit_route_choice')
        else:
            self._save_output('only_perimeter_control')


    def _get_expected_m(self):
        self.M = {}
        # mac_length时段内的M
        for i in self.regions:
            for h in self.regions[i].neighbor:
                self.M['%d_%d' % (i, h)] = self.M_last_mac_step['%d_%d' % (i, h)]-self.K_P*(self.regions[i].n-self.regions[i].n_last_mac_step)-self.K_I*(self.regions[i].n-self.regions[i].critical)
                self.M_last_mac_step['%d_%d' % (i, h)]=self.M['%d_%d' % (i, h)]
            self.regions[i].n_last_mac_step=self.regions[i].n
    def _only_perimeter_control(self,time):
        self._get_data(time)
        self._get_completion_flow()
        if time % self.mac_length == 0:
            self.oversaturated_regions = self._get_oversaturated_regions()
            self._get_expected_m()
        mic_step=(time % self.mac_length) / self.mic_length

        optimized_superphase = self._signal_control(mic_step)
        self._action_signal(optimized_superphase)


    def _save_data(self,time):
        for i in self.regions:
            #当前的n
            data = {}
            data['region'] = i
            data['time']=time
            data['real_n_t'] = self.regions[i].n
            data['output_t-1'] = self.regions[i].output
            self.N_data.append(data)
        self._save_network_data(time)

class backpressure_control(only_perimeter_control):
    def __init__(self,_with_logit_route_choice=False,partitioning=False):
        # 调用父类 control 的构造函数
        if partitioning:
            load_mfd=False
            self.partitioning=True
            self._with_logit_route_choice=True
            self.sumocfg_file = 'data/yangzhou.sumocfg'
        else:
            load_mfd=True
            self.partitioning=False
        super().__init__(load_mfd=load_mfd)


        if _with_logit_route_choice:
            self.save_path= 'control/backpressure_control_with_logit_route_choice'
        else:
            self.save_path= 'control/backpressure_control'

        self._with_logit_route_choice=_with_logit_route_choice

    def run(self):
        self._init()
        # 获取分区的inputlane,outputlane,bound等
        self.regions = self._region_nodelist(label=False, regions=self.regions)
        self.regions = self._region_demands(regions=self.regions,
                                                   begin=self.warm_time,end=100000)

        self._sim_backpressure_control()
        self._terminate()

    def _sim_backpressure_control(self):
        time = self.warm_time
        self._initialize_vars()

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulation.step(time=float(time))
            self._update_network_data()

            if self.check_saturated_condition():
                # exist oversaturated region
                if not self.control_active:
                    self.control_active = True
                    with open(self.log_file, "a") as file:
                        file.write(f"Begin control: {time}\n")
                    self.begin_control_time = time
                self._backpressure_signal_control(time - self.begin_control_time)
                if self._with_logit_route_choice:
                    self._action_logit_based_route()
            else:
                if self.control_active:  # 只记录结束时间一次
                    self.control_active = False
                    with open(self.log_file, "a") as file:
                        file.write(f"End control: {time}\n")
                    self.restore_to_fixed_signal()

            self._save_data(time)

            if not self.control_active:
                if len(self.total_vehicles) < self.threshold_veh_num:
                    break
            if time > self.end_time_threshold:
                break
            time += self.mic_length


        if self._with_logit_route_choice:
            self._save_output('backpressure_control_with_logit_route_choice')
        else:
            self._save_output('backpressure_control')


    def _backpressure_signal_control(self, time):
        self._get_data(time)

        region_temp = []
        for i in self.regions:
            region_temp.append(i)
            for h in self.regions[i].neighbor:
                if h in region_temp:
                    continue
                for tl in self.regions[i].toregion_trafficlight[h]:
                    max_weight = -100000
                    max_weight_phase = 0
                    # print(self.trafficlights[tl].phase)
                    for p in range(len(self.trafficlights[tl].phase)):
                        phase = self.trafficlights[tl].phase[p]
                        weight = self._calculate_weight(tl, phase)
                        if weight >= max_weight:
                            max_weight_phase = phase
                            max_weight = weight
                    traci.trafficlight.setRedYellowGreenState(tl, max_weight_phase)
                    traci.trafficlight.setPhaseDuration(tl, self.mic_length)

    def _get_data(self,time):
        if self._with_logit_route_choice:
            if time%self.update_edge_tt==0:
                self.update_edge_travel_time()

        for i in self.regions:
            self.regions[i].nd={}
            for j in self.regions:
                self.regions[i].nd[j]=0

        for i in self.regions:
            '''
            vehicle_number = 0
            vehicles = set()
            for j in self.regions[i].edge_list_IIE:
                vehicle_number += traci.edge.getLastStepVehicleNumber(j)
                vehicles.update(traci.edge.getLastStepVehicleIDs(j))
            self.regions[i].n=vehicle_number
            
            if time != self.warm_time:
                output = 0
                for j in self.regions[i].vehicles:
                    if j not in vehicles:
                        output += 1
                self.regions[i].output = output
            else:
                self.regions[i].output=-1
            self.regions[i].vehicles = vehicles
            '''
            for j in self.regions[i].vehicles:
                d_edge = traci.vehicle.getRoute(j)[-1]
                #车辆终点
                for k in self.regions:
                    if d_edge in self.regions[k].edge_list:
                        self.regions[i].nd[k] += 1
                        break
        if self.partitioning:
            self._get_control_vehicle_data()
            return
        self._get_completion_flow()
        self._get_control_vehicle_data()