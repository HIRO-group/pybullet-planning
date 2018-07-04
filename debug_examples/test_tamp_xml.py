#!/usr/bin/env python

from __future__ import print_function

import json
import os
import time

from pybullet_tools.utils import STATIC_MASS, CLIENT, user_input, connect, \
    disconnect, set_point, set_quat, set_pose, wait_for_interrupt, load_model, set_joint_position, \
    joint_from_name, has_joint, get_bodies, HideOutput, base_values_from_pose, set_camera, \
    get_image, create_shape, create_shape_array, get_box_geometry, get_cylinder_geometry, \
    Point, Pose, NULL_ID, euler_from_quat, quat_from_pose, point_from_pose, set_camera_pose, \
    get_sphere_geometry, reset_simulation, create_mesh, get_mesh_geometry, create_sphere, create_mesh, \
    load_pybullet, add_data_path
from pybullet_tools.pr2_utils import DRAKE_PR2_URDF, set_group_conf, REST_LEFT_ARM, rightarm_from_leftarm
from pybullet_tools.pr2_problems import create_floor

import pybullet as p
import numpy as np
import glob
from lxml import etree
from pybullet_tools.utils import quaternion_from_matrix

# https://docs.python.org/3.5/library/xml.etree.elementtree.html

def parse_array(element):
    return np.array(element.text.split(), dtype=np.float)

def parse_pose(element):
    homogenous = [0, 0, 0, 1]
    matrix = np.reshape(np.concatenate([parse_array(element), homogenous]), [4, 4])
    point = matrix[:3, 3]
    quat = quaternion_from_matrix(matrix)
    return (point, quat)

def parse_boolean(element):
    text = element.text
    if text == 'true':
        return True
    if text == 'false':
        return False
    raise ValueError(text)

def parse_object(obj, mesh_directory):
    name = obj.find('name').text
    mesh_filename = obj.find('geom').text
    pose = parse_pose(obj.find('pose'))

    mesh_path = os.path.join(mesh_directory, mesh_filename)
    print(mesh_path)
    geom = get_mesh_geometry(mesh_path)
    movable = parse_boolean(obj.find('moveable'))

    color = (.75, .75, .75, 1)
    if 'red' in name:
        color = (1, 0, 0, 1)
    if 'green' in name:
        color = (0, 1, 0, 1)
    if 'blue' in name:
        color = (0, 0, 1, 1)

    collision_id, visual_id = create_shape(geom, color=color)
    body_id = p.createMultiBody(baseMass=STATIC_MASS, baseCollisionShapeIndex=collision_id,
                                baseVisualShapeIndex=visual_id, physicsClientId=CLIENT)
    set_pose(body_id, pose)

    print(name, mesh_filename, movable)
    return body_id

def parse_robot(robot):
    name = robot.find('name').text
    # urdf = robot.find('name').text
    # fixed_base = not parse_boolean(robot.find('movebase'))
    pose = parse_pose(robot.find('basepose'))
    torso = parse_array(robot.find('torso'))
    left_arm = parse_array(robot.find('left_arm'))
    right_arm = parse_array(robot.find('right_arm'))
    assert (name == 'pr2')

    with HideOutput(True):
        robot_id = load_model(DRAKE_PR2_URDF, fixed_base=True)
    set_group_conf(robot_id, 'base', base_values_from_pose(pose))
    set_group_conf(robot_id, 'torso', torso)
    set_group_conf(robot_id, 'left_arm', left_arm)
    set_group_conf(robot_id, 'right_arm', right_arm)
    # print(robot.tag)
    # print(robot.attrib)
    # print(list(robot.iter('basepose')))
    return robot_id

def main():
    benchmark = 'tmp-benchmark-data'
    problem = 'problem2'
    #problem = 'problem3' # Clutter
    #problem = 'problem4' # Nonmono

    root_directory = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(root_directory, '..', 'problems', benchmark, problem)
    [mesh_directory] = list(filter(os.path.isdir, (os.path.join(directory, o)
                                                 for o in os.listdir(directory) if o.endswith('meshes'))))
    print(mesh_directory)
    [path] = glob.glob(os.path.join(directory, '*.xml'))
    connect(use_gui=True)
    add_data_path()
    load_pybullet("plane.urdf")

    xmlData = etree.parse(path)
    #root = xmlData.getroot()
    #print(root.items())
    for obj in xmlData.findall('/objects/obj'):
        parse_object(obj, mesh_directory)
    for robot in xmlData.findall('/robots/robot'):
        parse_robot(robot)
    wait_for_interrupt()
    disconnect()

if __name__ == '__main__':
    main()