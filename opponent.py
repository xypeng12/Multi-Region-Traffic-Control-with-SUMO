import traci
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import math
import scipy.signal as sg
import random

from xml.etree.ElementTree import Element
import csv
#import PulP as pl
from qpsolvers import solve_qp
from cvxopt import matrix, solvers
from sympy import *
from scipy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

class trafficlight:
    def __init__(self,id,program=None,phase=None,duration=None):
        self.id = id
        self.program=program
        self.phase=phase
        self.duration=duration

class Connection:
    def __init__(self,id,name=None,cfrom=None, cto = None, fromlane = None, tolane = None, dir = None, type = None, tl = None, linkindex =0):
        self.id=id
        self.name=name
        self.cfrom=cfrom
        self.cto=cto
        self.fromlane=fromlane
        self.tolane=tolane
        self.dir=dir
        self.type=type
        self.tl=tl
        self.linkindex=linkindex
        self.lanenum=1 #1 1/2 1/3...取决于有几个connection共用该lane
        self.throughput=0

class Edge:
    def __init__(self,id,efrom=0,eto=0,frompos=0,topos=0,length=0,numlane=0,max_speed=0):
        self.id=id
        self.reverse=0
        self.efrom=efrom
        self.eto=eto
        self.frompos=frompos
        self.topos=topos
        self.length =length
        self.vehicle_number=[]
        self.density=0
        self.twodir_density=0
        self.density_interval=[]
        self.twodir_density_interval=[]
        self.numlane=numlane
        self.max_speed=max_speed

        self.speed_list=[]
        self.travel_time=0
        self.num_speed=5 #5 to get the mean speed

    def update_speed(self,speed):
        if len(self.speed_list)<self.num_speed:
            self.speed_list.append(speed)
        else:
            self.speed_list=self.speed_list[1:]
            self.speed_list.append(speed)



class Node:
    def __init__(self, id,position=0):
        self.position=position
        self.id=id
        self.neighbor=[]


class Vehicle:
    def __init__(self, id=0,destination=0,routes = None, next_regions =None):
        self.id=id
        self.destination =destination
        self.routes=routes
        self.next_regions=next_regions #next_region和route对应
        self.route_choice=[] #route_choice=选择的next_region

class Region:
    def __init__(self, id):
        self.id=id
        self.node_list=[]
        self.edge_list=set()
        self.edge_list_IIE=set() #edge_list_including_internal_edge

        self.bound_node=[]
        self.external_edge_entry=[]
        self.internal_edge_entry = []
        self.external_edge_exit = []
        self.internal_edge_exit = []
        self.output_edge_list = []
        self.toregion_connection={}
        self.toregion_vehicles={}
        self.toregion_notl_vehicles={}
        self.toregion_trafficlight={}
        self.toregion_superphaseset={}
        self.toregion_superphasedic={}

        self.neighbor=[]#相邻子区
        self.ave_density=0
        self.twodir_edge_num=0#双向单向edge都算一个 edge总数

        #perimeter control
        # a control step k
        #for region i
        #Ni(k)
        self.n=0
        #Nii(k) Nij(k)
        self.n_p=0
        self.n_bc=0
        self.output=0
        self.n_last_mac_step=0 #for PI-based PC


        self.nd={}
        #Qii(k) Qij(k)
        self.q={}
        #列表 存所有步长的q{}
        self.demands=[]

        self.mfd_param=[]
        self.completion_flow=0
        self.critical=0

        #route choice
        self.control_vehicles={}
        self.ave_density=0.0

        self.input_lane=[]
        self.output_lane=[]
        #0-1200step的累计车辆数，输入车辆数，输出车辆数
        self.accumulation=[]

        self.G=[]
        self.outputlaneset_bound={}
        self.outputlaneset={}
        self.output_vehicles={}
        self.wait_vehicle_num=0

        self.travel_time=[]
        self.delay=[]
        #本step的 vehicle set
        self.vehicles=[]
