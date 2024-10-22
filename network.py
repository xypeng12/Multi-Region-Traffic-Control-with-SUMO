import traci
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import scipy.signal as sg

from opponent import *

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
class network:
    def _init(self,seed=None):
        self._init_sim(seed)
        self._init_nodes()
        self._init_edges()
        self._init_connections()
        self._init_trafficlight()
    def _MFD_function(self, x, a):
        #revise2
        G = a[0]* (x ** 3)+a[1] * (x ** 2)+a[2] *x
        return G

    def _get_critical(self, x, mfd_param):
        x2 = range(int(min(x)), int(max(x)), 2)
        y2 = []
        for j in range(len(x2)):
            y2.append(self._MFD_function(x2[j], mfd_param))
        index = sg.argrelmax(np.array(y2))[0]
        top=0
        if len(index) == 0:
            print('没到极值')
            top=-1
            critical = x2[len(x2) - 1]
        else:
            critical = x2[int(index[0])]

        return critical,top

    def _fit(self,x, y):
        time=0
        while True:
            y_x = []
            for j in range(len(x)):
                if x[j] == 0:
                    y_x.append(0.0)
                else:
                    y_x.append(float(y[j] / x[j]))

            mfd_param = np.polyfit(x, y_x, 2)  # 拟合2次方程的参数
            critical,top=self._get_critical(x,mfd_param)

            if time>=2:
                break
            x_update = []
            y_update = []
            for j in range(len(x)):
                y_temp = self._MFD_function(x[j], mfd_param)
                if y_temp > y[j] and x[j]>critical:
                    continue
                if top==-1 and y_temp > y[j] and x[j]>critical*0.3:
                    continue
                x_update.append(x[j])
                y_update.append(y[j])
            x = x_update
            y = y_update
            time+=1

        return x, y,mfd_param,critical

    def _update_internal_edge(self,regions):
        for i in regions:
            regions[i].edge_list_IIE = regions[i].edge_list.copy()

        for j in self.connections:
            if self.connections[j].cfrom[0] != ":":
                # cto not internal edge
                continue
            for i in regions:
                if self.connections[j].cto in set(regions[i].edge_list):
                    regions[i].edge_list_IIE.add(self.connections[j].cfrom)

    def _region_nodelist(self,label=True,regions=None):
        if label==True:
            regions=self.regions

        self._update_internal_edge(regions)

        for i in regions:
            for edge_id in regions[i].edge_list:
                edge=self.edges[edge_id]
                if edge.efrom not in regions[i].node_list:
                    regions[i].node_list.append(edge.efrom)
                if edge.eto not in regions[i].node_list:
                    regions[i].node_list.append(edge.eto)
                if edge.eto in self.endnodes:
                    regions[i].output_edge_list.append(edge_id)

        for i in regions:
            for node in regions[i].node_list:
                temp=0
                for nnode in self.nodes[node].neighbor:
                    if nnode not in regions[i].node_list:
                        temp=1
                        for j in self.edges:
                            edge=self.edges[j]
                            if node==edge.eto and nnode==edge.efrom:
                                regions[i].external_edge_entry.append(j)
                            if node==edge.efrom and nnode==edge.eto:
                                regions[i].external_edge_exit.append(j)
                        for j in regions:
                            if nnode in regions[j].node_list:
                                if j not in regions[i].neighbor:
                                    regions[i].neighbor.append(j)
                if temp==1:
                    regions[i].bound_node.append(node)

            for node in regions[i].bound_node:
                for nnode in self.nodes[node].neighbor:
                    if nnode in regions[i].node_list:
                        for j in self.edges:
                            edge=self.edges[j]
                            if node==edge.eto and nnode==edge.efrom:
                                regions[i].internal_edge_entry.append(j)
                            if node==edge.efrom and nnode==edge.eto:
                                regions[i].internal_edge_exit.append(j)


            for k in regions[i].neighbor:
                regions[i].toregion_connection[k]=[]
                regions[i].toregion_vehicles[k] = []
                regions[i].toregion_notl_vehicles[k]=[]
            for edge_id in regions[i].internal_edge_entry:
                for lid in range(self.edges[edge_id].numlane):
                    lane=edge_id+'_'+str(lid)
                    link_lane_group=traci.lane.getLinks(lane)
                    for j in range(len(link_lane_group)):
                        link_lane=list(link_lane_group[j])[0]
                        link_edge=traci.lane.getEdgeID(link_lane)
                        if link_edge not in regions[i].external_edge_exit:
                            continue
                        for cid in self.connections:
                            if self.connections[cid].name == lane + '*' + link_lane:
                                connection_id=cid
                                break
                        for k in regions[i].neighbor:
                            if link_edge in regions[k].edge_list:
                                regions[i].toregion_connection[k].append(connection_id)

        for i in regions:
            outputlaneset_bound = set()
            for h in regions[i].neighbor:
                connection_set =regions[i].toregion_connection[h]
                for cid in connection_set:
                    connection = self.connections[cid]
                    outputlane = connection.cfrom + '_' + connection.fromlane
                    if outputlane not in outputlaneset_bound:
                        outputlaneset_bound.add(outputlane)
            regions[i].outputlaneset_bound = outputlaneset_bound
            outputlaneset=outputlaneset_bound
            for edge_id in regions[i].output_edge_list:
                for lid in range(self.edges[edge_id].numlane):
                    lane=edge_id+'_'+str(lid)
                    outputlaneset.add(lane)
            regions[i].outputlaneset = outputlaneset

        self.endnodes = set()

        for i in regions:
            for k in regions[i].neighbor:
                regions[i].toregion_trafficlight[k]={}
                for cid in regions[i].toregion_connection[k]:
                    if self.connections[cid].type!='tl':
                        continue
                    tl=self.connections[cid].tl
                    if tl not in regions[i].toregion_trafficlight[k]:
                        regions[i].toregion_trafficlight[k][tl]=[]
                    regions[i].toregion_trafficlight[k][tl].append(cid)

        for i in regions:
            for h in regions[i].neighbor:
                for tl in regions[i].toregion_trafficlight[h]:
                    connection_set = regions[i].toregion_trafficlight[h][tl]
                    if tl not in regions[h].toregion_trafficlight[i]:
                        reversed_connection_set = []
                    else:
                        reversed_connection_set =regions[h].toregion_trafficlight[i][tl]
                    green_phase=[]
                    for p in range(len(self.trafficlights[tl].phase)):
                        phase = self.trafficlights[tl].phase[p]
                        for cid in connection_set:
                            linkindex = self.connections[cid].linkindex
                            if phase[linkindex] == 'g' or phase[linkindex] == 'G':
                                green_phase.append(phase)
                                break
                    new_phase=[]
                    for phase in green_phase:
                        for cid in reversed_connection_set:
                            linkindex=self.connections[cid].linkindex
                            phase = phase[:linkindex] + 'r' + phase[linkindex + 1:]
                        new_phase.append(phase)

                    for phase in new_phase:
                       if phase not in self.trafficlights[tl].phase:
                           self.trafficlights[tl].phase.append(phase)

        if label==True:
            self.regions=regions
        else:
            return regions
    def _init_sim(self,seed=None):
        # 访问sumo
        #sumoBinary = 'sumo-gui'
        sumoBinary='sumo'
        sumoCmd = [sumoBinary, "-c",self.sumocfg_file]
        if seed is not None:
            sumoCmd += ["--seed", str(seed)]
            print(f"SUMO started with seed: {seed}")
        else:
            print("SUMO started with default random seed")

        traci.start(sumoCmd)


    def _init_nodes(self):
        self.nodes = {}
        for id in traci.junction.getIDList():
            position=traci.junction.getPosition(id)
            self.nodes[id] = Node(id,position=position)

        tree = ET.parse("data\\yangzhou.net.xml")
        root = tree.getroot()
        self.endnodes = set()
        for child in root:
            if child.tag != 'junction':
                continue
            data = child.attrib
            if data['type']=="dead_end":
                self.endnodes.add(data['id'])

    def _init_trafficlight(self):
        tree = ET.parse("data\\yangzhou.net.xml")
        root = tree.getroot()
        self.trafficlights = {}
        for child in root:
            if child.tag != 'tlLogic':
                continue
            data = child.attrib
            id=data['id']
            program=data['programID']
            phase=[]
            duration=[]
            for i in child:
                if i.tag != 'phase':
                    continue
                state=i.attrib['state']
                temp=0
                for p in state:
                    if p=='G' or p=='g':
                        temp=1
                if temp==1:
                    phase.append(i.attrib['state'])
                    duration.append(i.attrib['duration'])
            self.trafficlights[id]=trafficlight(id,program=program,phase=phase,duration=duration)


    def _init_connections(self):
        tree = ET.parse("data\\yangzhou.net.xml")
        root = tree.getroot()
        self.connections={}
        index=0
        for child in root:
            if child.tag != 'connection':
                continue
            data = child.attrib
            name=data['from']+'_'+data['fromLane']+'*'+data['to']+'_'+data['toLane']
            id=index
            cfrom=data['from']
            cto=data['to']
            fromlane =data['fromLane']
            tolane=data['toLane']
            dir=data['dir']

            if 'tl' in data:
                type='tl'
                tl=data['tl']
                linkindex=int(data['linkIndex'])
                self.connections[id]=Connection(id,name=name,cfrom=cfrom,cto=cto,fromlane=fromlane,tolane=tolane,dir=dir,type=type,tl=tl,linkindex=linkindex)
                index+=1
            else:
                type='notl'
                self.connections[id]=Connection(id,name=name,cfrom=cfrom,cto=cto,fromlane=fromlane,tolane=tolane,dir=dir,type=type)
                index+=1

        for i in self.connections:
            num=0
            for j in self.connections:
                if self.connections[i].cfrom==self.connections[j].cfrom and self.connections[i].fromlane==self.connections[j].fromlane:
                    num+=1
            self.connections[i].lanenum=float(1./num)


    def _init_edges(self):
        tree=ET.parse("data\\yangzhou.net.xml")
        root=tree.getroot()
        self.edges = {}
        self.edges_id=[]
        self.edges_IIE_id=[]
        for child in root:
            if child.tag != 'edge':
                continue
            data=child.attrib
            id=data['id']
            self.edges_IIE_id.append(id)
            if not (('function' in child.attrib) and (child.attrib['function']=="internal")):
                self.edges_id.append(id)
                efrom=data['from']
                eto=data['to']

                length=[]
                speed=[]
                for i in child:
                    if i.tag == 'lane':
                        length.append(float(i.attrib['length']))
                        speed.append(float(i.attrib['speed']))
                length=np.mean(np.array(length))
                speed=np.mean(np.array(speed))

                frompos = self.nodes[efrom].position
                topos = self.nodes[eto].position

                if eto not in self.nodes[efrom].neighbor:
                    self.nodes[efrom].neighbor.append(eto)
                if efrom not in self.nodes[eto].neighbor:
                    self.nodes[eto].neighbor.append(efrom)
                numlane = traci.edge.getLaneNumber(id)


                self.edges[id] = Edge(id, efrom=efrom, eto=eto, frompos=frompos, topos=topos, length=length,numlane=numlane,max_speed=speed)

        for i in self.edges:
            edge = self.edges[i]
            for j in self.edges:
                edge_r = self.edges[j]
                if edge_r.eto == edge.efrom and edge_r.efrom == edge.eto:
                    edge.reverse = j


    def _terminate(self):
        traci.close()
