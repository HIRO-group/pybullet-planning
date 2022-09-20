from __future__ import print_function

import copy
from this import d
from xml.dom import NotFoundErr
import pybullet as p
import random
import numpy as np
import time
from itertools import islice, count
import panda_dynamics_model as pdm
import numpy as np
import scipy
from .ikfast.franka_panda.ik import is_ik_compiled, ikfast_inverse_kinematics, PANDA_LEFT_INFO, bi_panda_inverse_kinematics, PANDA_RIGHT_INFO
from .ikfast.utils import USE_CURRENT, USE_ALL
from .pr2_problems import get_fixed_bodies
from .panda_utils import TOP_HOLDING_LEFT_ARM_CENTERED, TOP_HOLDING_LEFT_ARM, SIDE_HOLDING_LEFT_ARM, GET_GRASPS, get_gripper_joints, \
    get_carry_conf, get_top_grasps, get_side_grasps, open_arm, arm_conf, get_gripper_link, get_arm_joints, \
    learned_pose_generator, PANDA_TOOL_FRAMES, get_x_presses, BI_PANDA_GROUPS, joints_from_names, arm_from_arm,\
    is_drake_pr2, get_group_joints, get_group_conf, compute_grasp_width, PANDA_GRIPPER_ROOTS, get_group_links, \
    are_forces_balanced, get_other_arm, TARGET, PLATE_GRASP_LEFT_ARM
from .utils import invert, multiply, get_name, set_pose, get_link_pose, is_placement, \
    pairwise_collision, set_joint_positions, get_joint_positions, sample_placement, get_pose, waypoints_from_path, \
    unit_quat, plan_base_motion, plan_joint_motion, base_values_from_pose, pose_from_base_values, \
    uniform_pose_generator, sub_inverse_kinematics, add_fixed_constraint, remove_debug, remove_fixed_constraint, \
    disable_real_time, enable_gravity, joint_controller_hold, get_distance, \
    get_min_limit, user_input, step_simulation, get_body_name, get_bodies, BASE_LINK, \
    add_segments, get_max_limit, link_from_name, BodySaver, get_aabb, Attachment, interpolate_poses, \
    plan_direct_joint_motion, has_gui, create_attachment, wait_for_duration, get_extend_fn, set_renderer, \
    get_custom_limits, all_between, get_unit_vector, wait_if_gui, joint_from_name, get_joint_info,\
    set_base_values, euler_from_quat, INF, elapsed_time, get_moving_links, flatten_links, get_relative_pose, get_link_name, \
    get_max_force, compute_jacobian, get_COM, matrix_from_quat, check_overlap, set_point, set_joint_position, get_joint_position, \
    is_pose_on_r, quat_from_euler, set_joint_positions_torque, body_from_name, get_same_relative_pose, plan_waypoints_joint_motion, \
    is_pose_close, get_configuration, get_max_limits, get_min_limits, create_sub_robot, refine_path, plan_cartesian_motion, is_b1_on_b2, \
    get_mass, LockRenderer, plan_direct_joint_motion_force_aware, plan_joint_motion_force_aware
import math
import csv
# GRASP_INFO = {
#     'top': GraspInfo(lambda body: get_top_grasps(body, under=True, tool_pose=Pose(), max_width=INF,  grasp_length=0),
#                      approach_pose=Pose(0.1*Point(z=1))),
# }

BASE_EXTENT = 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = -0.02
APPROACH_DISTANCE = 0.02 + GRASP_LENGTH
SELF_COLLISIONS = False
TABLE = 3

##################################################

def get_base_limits(robot):
    if is_drake_pr2(robot):
        joints = get_group_joints(robot, 'base')[:2]
        lower = [get_min_limit(robot, j) for j in joints]
        upper = [get_max_limit(robot, j) for j in joints]
        return lower, upper
    return BASE_LIMITS

##################################################

class Pose(object):
    num = count()
    #def __init__(self, position, orientation):
    #    self.position = position
    #    self.orientation = orientation
    def __init__(self, body, value=None, support=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.init = init
        self.index = next(self.num)
    @property
    def bodies(self):
        return flatten_links(self.body)
    def assign(self):
        set_pose(self.body, self.value)
    def iterate(self):
        yield self
    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'p{}'.format(index)

class Grasp(object):
    def __init__(self, grasp_type, body, value, approach, carry):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value) # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)
    def get_attachment(self, robot, arm):
        tool_link = link_from_name(robot, PANDA_TOOL_FRAMES[arm])
        return Attachment(robot, tool_link, self.value, self.body)
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class Conf(object):
    def __init__(self, body, joints, values=None, init=False, velocities=None):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init
        self.torque_joints = {}
        self.velocities = velocities[:len(joints)] if velocities is not None else velocities

    @property
    def bodies(self): # TODO: misnomer
        return flatten_links(self.body, get_moving_links(self.body, self.joints))
    def assign(self, bodies=[]):
        print(self.velocities)
        set_joint_positions_torque(self.body, self.joints, self.values, self.velocities)
    def iterate(self):
        yield self
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)

#####################################

def compute_joint_velocities(target_ee_velocity, Ja, Jl):
    Jl_inv = np.linalg.pinv(Jl)
    Ja_inv = np.linalg.pinv(Ja)
    linear_vel = np.matmul(target_ee_velocity[:3], Jl_inv)
    angular_vel = np.matmul(target_ee_velocity[3:], Ja_inv)
    return linear_vel + angular_vel

def velocity_move(robot, arm, target_pose):
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    print(target_pose, len(target_pose))
    target_joints = target_pose[1]
    target_pose = target_pose[0]
    gripper_link = get_gripper_link(robot, arm)
    joints = get_arm_joints(robot, arm)

    max_forces = [get_max_force(robot, joint) for joint in joints]
    # cur_pose = get_link_pose(int(robot), int(gripper_link))
    cur_pose = get_link_pose(robot, gripper_link)
    print(cur_pose)
    jointPoses = [target_joints[i] for i in range(len(target_joints))]
    while not is_pose_close(cur_pose, target_pose):
        print(len(target_joints))
        print(len(joints))
        Jl, Ja =  compute_jacobian(robot, gripper_link)
        Jl = np.array(Jl)[:7]
        Ja = np.array(Ja)[:7]

        diff = [target_pose[0][i] - cur_pose[0][i] for i in range(len(target_pose[0]))]
        ta = euler_from_quat(target_pose[1])
        ca = euler_from_quat(cur_pose[1])
        diffA = [ta[i] - ca[i] for i in range(len(ca))]
        div = 10
        vels = [0, 0.1, 0, 0, 0, 0]
        joint_velocities = compute_joint_velocities(vels, Ja, Jl)
        print(joint_velocities)
        p.setJointMotorControlArray(robot, joints, p.VELOCITY_CONTROL, targetVelocities=joint_velocities, forces=max_forces)
        cur_pose = get_link_pose(robot, gripper_link)

class Command(object):
    def control(self, dt=0):
        raise NotImplementedError()
    def apply(self, state, **kwargs):
        raise NotImplementedError()
    def iterate(self):
        raise NotImplementedError()

class Commands(object):
    def __init__(self, state, savers=[], commands=[]):
        self.state = state
        self.savers = tuple(savers)
        self.commands = tuple(commands)
    def assign(self):
        for saver in self.savers:
            saver.restore()
        return copy.copy(self.state)
    def apply(self, state, **kwargs):
        for command in self.commands:
            for result in command.apply(state, **kwargs):
                yield result
    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)

#####################################

def set_joint_force_limits(robot, arm):
    links = get_group_links(robot, arm_from_arm(arm))
    joints = get_group_joints(robot, arm_from_arm(arm))
    for i in range(len(joints)):
        joint = joints[i]
        limits = get_joint_info(robot, joint)
        p.changeDynamics(robot, links[i], jointLimitForce=limits.jointMaxForce)

######################################
def calc_torques(robot, arm, joints, mass = 4):
    ee_link = get_gripper_link(robot, arm)
    max_limits = [get_max_force(robot, joint) for joint in joints]
    Jl, Ja = compute_jacobian(robot, ee_link)
    Jl = np.array(Jl)
    Ja = np.array(Ja)
    J = Jl[:7,:]
    J = np.linalg.pinv(J)
    force = mass * 9.8
    force3d = np.array([0, 0, force])
    torques = np.matmul(force3d, J)
    return torques, any(torques[i] > max_limits[i] for i in range(len(torques)))

torques_exceded = False
class Trajectory(Command):
    _draw = False
    def __init__(self, path, bodies):
        self.path = tuple(path)
        self.bodies = bodies
        self.file = '/home/liam/exp_data/torque_data_fa_2kg_2.csv'
        # TODO: constructor that takes in this info
    def apply(self, state, sample=1):
        global torques_exceded
        handles = add_segments(self.to_points()) if self._draw and has_gui() else []
        print('in traj apply')
        for conf in self.path[::sample]:
            conf.assign(self.bodies)
            torques, hold = calc_torques(conf.body, 'right', conf.joints)
            torques_exceded |= hold
            # with open(self.file, 'a') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(torques)
            # wait_for_duration(.01)
            yield
        print('finished conf assigns')
        end_conf = self.path[-1]
        count = 0
        if isinstance(end_conf, Pose):
            state.poses[end_conf.body] = end_conf
        for handle in handles:
            remove_debug(handle)
    def control(self, dt=0, **kwargs):
        # TODO: just waypoints
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()
            for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
                # step_simulation()
                time.sleep(dt)
    def to_points(self, link=BASE_LINK):
        # TODO: this is computationally expensive
        points = []
        for conf in self.path:
            with BodySaver(conf.body):
                conf.assign()
                #point = np.array(point_from_pose(get_link_pose(conf.body, link)))
                point = np.array(get_group_conf(conf.body, 'base'))
                point[2] = 0
                point += 1e-2*np.array([0, 0, 1])
                if not (points and np.allclose(points[-1], point, atol=1e-3, rtol=0)):
                    points.append(point)
        points = get_target_path(self)
        return waypoints_from_path(points)
    def distance(self, distance_fn=get_distance):
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            total += distance_fn(q1.values, q2.values)
        return total
    def iterate(self):
        for conf in self.path:
            yield conf
    def reverse(self):
        return Trajectory(reversed(self.path), self.bodies)
    #def __repr__(self):
    #    return 't{}'.format(id(self) % 1000)
    def __repr__(self):
        d = 0
        if self.path:
            conf = self.path[0]
            d = 3 if isinstance(conf, Pose) else len(conf.joints)
        return 't({},{})'.format(d, len(self.path))

def get_torques_exceded_global():
    global torques_exceded
    return torques_exceded

def reset_torques_exceded_global():
    global torques_exceded
    torques_exceded = False

def create_trajectory(robot, joints, path, bodies, velocities=None):
    confs = []
    index = 0
    if velocities is not None:
        for i in range(len(velocities)):
            confs.append(Conf(robot, joints, path[i], velocities=velocities[i]))
            index+=1
    for i in range(index, len(path)):
            confs.append(Conf(robot, joints, path[i], velocities=None))
    return Trajectory(confs, bodies=bodies)

##################################################

class GripperCommand(Command):
    def __init__(self, robot, arm, position, teleport=False):
        self.robot = robot
        self.arm = arm
        self.position = position
        self.teleport = teleport
    def apply(self, state, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        start_conf = get_joint_positions(self.robot, joints)
        end_conf = [self.position] * len(joints)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(self.robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions
    def control(self, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        positions = [self.position]*len(joints)
        for _ in joint_controller_hold(self.robot, joints, positions):
            yield

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, self.position)
class Attach(Command):
    vacuum = True
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, PANDA_TOOL_FRAMES.get(self.arm, self.arm))
        #self.attachment = None
    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = multiply(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)
    def apply(self, state, **kwargs):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        del state.poses[self.body]
        yield
    def control(self, dt=0, **kwargs):
        if self.vacuum:
            add_fixed_constraint(self.body, self.robot, self.link)
            #add_fixed_constraint(self.body, self.robot, self.link, max_force=1) # Less force makes it easier to pick
        else:
            # TODO: the gripper doesn't quite work yet
            gripper_name = '{}_gripper'.format(self.arm)
            joints = joints_from_names(self.robot, BI_PANDA_GROUPS[gripper_name])
            values = [get_min_limit(self.robot, joint) for joint in joints] # Closed
            for _ in joint_controller_hold(self.robot, joints, values):
                # step_simulation()
                time.sleep(dt)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

class FixObj(Command):
    def __init__(self, b1, b2):
        self.b1 = b1
        self.b2 = b2
        print("&&&&&&&&&&&&&&&&&&&&&&& ", self.b1)
        print("*********************** ", self.b2)
    def assign(self):
        pass
    def apply(self, state, **kwargs):

        if state.poses.get(self.b1, None):
            wait_for_duration(0.5)
            state.attachments[self.b1] = create_attachment(self.b2, -1, self.b1)
        yield
    def control(self, dt=0, **kwargs):
        p.createConstraint(self.b2, -1, self.b1, -1,  # Both seem to work
                                    p.JOINT_FIXED, jointAxis=unit_point(),
                                    parentFramePosition=point,
                                    childFramePosition=unit_point(),
                                    parentFrameOrientation=quat,
                                    childFrameOrientation=unit_quat(),
                                    physicsClientId=CLIENT)
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, get_body_name(self.b2), get_body_name(self.b1))

class Detach(Command):
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot, PANDA_TOOL_FRAMES.get(self.arm, self.arm))
        # TODO: pose argument to maintain same object
    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        state.poses[self.body] = Pose(self.body, get_pose(self.body))
        del state.grasps[self.body]
        yield
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

##################################################

class Clean(Command):
    def __init__(self, body):
        self.body = body
    def apply(self, state, **kwargs):
        state.cleaned.add(self.body)
        self.control()
        yield
    def control(self, **kwargs):
        p.addUserDebugText('Cleaned', textPosition=(0, 0, .25), textColorRGB=(0,0,1), #textSize=1,
                           lifeTime=0, parentObjectUniqueId=self.body)
        #p.setDebugObjectColor(self.body, 0, objectDebugColorRGB=(0,0,1))
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class Cook(Command):
    # TODO: global state here?
    def __init__(self, body):
        self.body = body
    def apply(self, state, **kwargs):
        state.cleaned.remove(self.body)
        state.cooked.add(self.body)
        self.control()
        yield
    def control(self, **kwargs):
        # changeVisualShape
        # setDebugObjectColor
        #p.removeUserDebugItem # TODO: remove cleaned
        p.addUserDebugText('Cooked', textPosition=(0, 0, .5), textColorRGB=(1,0,0), #textSize=1,
                           lifeTime=0, parentObjectUniqueId=self.body)
        #p.setDebugObjectColor(self.body, 0, objectDebugColorRGB=(1,0,0))
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

##################################################

def get_grasp_gen(problem, collisions=False, randomize=True):
    for grasp_type in problem.grasp_types:
        if grasp_type not in GET_GRASPS:
            raise ValueError('Unexpected grasp type:', grasp_type)
    def fn(body):
        # TODO: max_grasps
        # TODO: return grasps one by one
        grasps = []
        arm = 'right'
        #carry_conf = get_carry_conf(arm, 'top')
        print("$$$$$$$$$$$$$$$$$$$$$$$$ grasp gen")
        if 'top' in problem.grasp_types:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([1, 0, 0])
            grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM_CENTERED)
                          for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        if 'side' in problem.grasp_types:
            approach_vector = APPROACH_DISTANCE*get_unit_vector([2, 0, -1])
            grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
                          for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))
        filtered_grasps = []
        for grasp in grasps:
            grasp_width = compute_grasp_width(problem.robot, arm, body, grasp.value) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        if randomize:
            random.shuffle(filtered_grasps)
        return [(g,) for g in filtered_grasps]
        #for g in filtered_grasps:
        #    yield (g,)
    return fn

##################################################

def accelerate_gen_fn(gen_fn, max_attempts=1):
    def new_gen_fn(*inputs):
        generator = gen_fn(*inputs)
        while True:
            for i in range(max_attempts):
                try:
                    output = next(generator)
                except StopIteration:
                    return
                if output is not None:
                    print(gen_fn.__name__, i)
                    yield output
                    break
    return new_gen_fn

def get_stable_gen(problem, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    robot = problem.robot
    link = get_gripper_link(robot, problem.holding_arm)
    def gen(body, surface):
        print("in stable gen")
        # TODO: surface poses are being sampled in pr2_belief
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        while True:
            surface = random.choice(surfaces) # TODO: weight by area
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            
            p.assign()
            if are_forces_balanced(body, p, surface,robot, link, problem.movable) and not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                if surface == body_from_name(TARGET):
                    p.value = ((p.value[0][0],p.value[0][1],p.value[0][2]),p.value[1])
                    p.assign()
                yield (p,)
    # TODO: apply the acceleration technique here
    return gen

def get_stable_gen_dumb(problem, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    robot = problem.robot
    def gen(body, surface):
        # TODO: surface poses are being sampled in pr2_belief
        print("in stable gen dumb fn")
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        while True:
            surface = random.choice(surfaces) # TODO: weight by area
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                print("no body pose")
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                p.value = ((p.value[0][0],p.value[0][1],p.value[0][2]+0.01),p.value[1])
                p.assign()
                print("pose found")
                yield (p,)
    # TODO: apply the acceleration technique here
    return gen

##################################################

def get_tool_from_root(robot, arm):
    root_link = link_from_name(robot, PANDA_GRIPPER_ROOTS[arm])
    tool_link = link_from_name(robot, PANDA_TOOL_FRAMES[arm])
    return get_relative_pose(robot, root_link, tool_link)

def iterate_approach_path(robot, arm, gripper, pose, grasp, body=None):
    tool_from_root = get_tool_from_root(robot, arm)
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

def get_ir_sampler(problem, custom_limits={}, max_attempts=100, collisions=True, learned=True):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                print("pairwise collision")
                return
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = arm_conf(arm, grasp.carry)
        base_joints = get_group_joints(robot, 'base')
        arm_joints = get_arm_joints(robot, arm)
        base_conf = get_pose(robot)
        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            count = 0
            for _ in range (max_attempts):
                count += 1
                # if not all_between(lower_limits, base_conf, upper_limits):
                #     continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                # bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    print("if robot loop collision",{get_body_name(b) :pairwise_collision(robot, b) for b in obstacles + [obj]})
                    continue
                #print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

#################################################

def get_torque_limits_mock_test(problem):
    def test(a, poses = None, ptotalMass = None, pcomR = None):
        return True

##################################################
def get_torque_limits_not_exceded_test(problem):
    robot = problem.robot
    max_limits = []
    baseLink = 1
    # max_limits[-2] = 10
    # max_limits[-3] = 10
    EPS =  1
    comR = []
    totalMass = 0
    def test(a, poses = None, ptotalMass = None, pcomR = None):
        l = get_gripper_link(robot, a)
        joints = get_arm_joints(robot, a)
        for joint in joints:
            max_limits.append(get_max_force(problem.robot, joint))
        print("in torque test")
        Jl, Ja = compute_jacobian(problem.robot, l)
        Jl = np.array(Jl)
        Ja = np.array(Ja)
        # J = Jl[:7,:]
        J = np.concatenate((Jl[:7,:], Ja[:7,:]), axis = 1)
        J = np.linalg.pinv(J)
        if pcomR is not None and ptotalMass is not None:
            comR = pcomR
            totalMass = ptotalMass
        else:
            comR, totalMass = get_COM(problem.movable, b2=body_from_name(TARGET), poses=poses)
        force = totalMass * 9.8
        # force3d = np.array([0, 0, force, 0, 0, 0])
        force3d = np.array([0, 0, force])
        torques = np.matmul(force3d, J)
        # print(torques)
        testVal = True
        # print(torques)
        # print(max_limits)
        for i in range(len(torques)):
            if (abs(torques[i]) >= max_limits[i]*EPS) and i != 5:
                testVal = False
        print("torque test: ", testVal)
        return testVal
    return test



def get_torque_limits_not_exceded_test_v2(problem, arm, mass=None):
    robot = problem.robot
    max_limits = []
    baseLink = 1
    joints = get_arm_joints(robot, arm)
    for joint in joints:
            max_limits.append(get_max_force(problem.robot, joint))
    ee_link = get_gripper_link(robot, arm)
    EPS =  1
    totalMass = mass
    if totalMass is None:
        totalMass = get_mass(problem.movable[-1])
    comR = []
    totalMass = 0
    def test(poses = None, ptotalMass = None, velocities=None, accelerations=None):
        # return True
        totalMass = ptotalMass
        print("in torque test")
        if totalMass is None:
            totalMass = get_mass(problem.movable[-1])
        if velocities == None or accelerations == None:
            velocities = [0]*len(poses)
            accelerations = [0]*len(poses)
        with LockRenderer(lock=True):
            hold = get_joint_positions(robot, joints)
            set_joint_positions(robot, joints, poses)
            accelerations += [0.0] * 2
            velocities += [0.0] * 2
            Jl, Ja = compute_jacobian(problem.robot, ee_link, velocities=velocities, accelerations=accelerations)
            M = pdm.get_mass_matrix(poses)
            p = np.ndarray(())
            C = pdm.get_coriolis_matrix(np.asarray([poses[:7]], dtype=np.float64).transpose(), np.asarray([velocities[:7]], dtype=np.float64).transpose())
            torquesInert = np.matmul(M, accelerations[:7])
            torquesC = np.matmul(C, velocities[:7])
            torquesG = pdm.get_gravity_vector(poses)

            set_joint_positions(robot, joints, hold)
            Jl = np.array(Jl).transpose()
            Ja = np.array(Ja).transpose()
            
            J = np.concatenate((Jl, Ja))

            print(J.shape)
            # J = np.transpose(J)
            Jt = np.transpose(J)
            print(J.shape)
            force = totalMass * -9.8
            toolV = np.matmul(J, velocities)
            v = np.array(toolV[:3])
            w = np.array(toolV[3:])
            aOR = np.cross(v, w)/ (np.linalg.norm(w)**2)
            fCoriolis = -2 * totalMass * np.cross(v, w)
            fCentrifugal = np.cross(-totalMass * w, np.cross(w, aOR))
            fFictitious = fCoriolis + fCentrifugal
            force3d = np.array([fFictitious[0], fFictitious[1], force + fFictitious[2], 0, 0, 0])
            # print(force3d)
            # print(len(J), len(J[0]))
            torquesExt = np.matmul(Jt, force3d)
            torques = torquesExt[:7] + torquesInert + torquesC + torquesG
            for i in range(len(max_limits)-1):
                if (abs(torques[i]) >= max_limits[i]*EPS):
                    print("torque test: FAILED", i, torques[i])
                    print("Velocities: ", velocities)
                    print("Accelerations: ", accelerations)
                    return False
            print("torque test: PASSED")
            return True
    return test

def get_dynamics_fn(problem):
    t = problem.time_step
    def dynam_fn(q, prev_q, prev_v, prev_a):
        acc = [0]*len(q)
        vel = [0]*len(q)
        for i in range(len(q)):
            dist = q[i] - prev_q[i]
            vel[i]  = dist/t
            acc[i] = (prev_v[i]-vel[i])/t
        return vel, acc
    return dynam_fn

originalJointPoses = None
prevOriginalJointPoses = None
def get_sample_stable_holding_conf_gen_v2(problem, custom_limits={}, max_attempts = 25, stepSize = 0.00001):
    holding_arm = problem.holding_arm
    robot = problem.robot
    max_limits = []
    l = PANDA_TOOL_FRAMES[holding_arm]
    joints = BI_PANDA_GROUPS[arm_from_arm(problem.holding_arm)]
    jointNums = []
    arm_link = get_gripper_link(robot, holding_arm)
    for joint in joints:
        jointNum = joint_from_name(robot, joint)
        jointNums.append(jointNum)
        max_limits.append(get_max_force(problem.robot, jointNum))
    gripper_link = get_gripper_link(robot, holding_arm)
    arm_joints = get_arm_joints(robot, holding_arm)
    test_torque_limits = get_torque_limits_not_exceded_test(problem)
    jointStart = get_joint_position(robot, joint_from_name(robot, BI_PANDA_GROUPS[arm_from_arm(holding_arm)][-1]))
    link_start = euler_from_quat(get_link_pose(robot, gripper_link)[1])[1]
    hand_pose = get_link_pose(robot, gripper_link)
    pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
    originalTargetPose = Pose(body_from_name(TARGET), pose)
    y_range = np.arange(hand_pose[0][1], hand_pose[0][1] + 0.2, .01) # full extension
    def gen(arm, pcomR, ptotalMass, obj, poses=None):
        obstacles = set()
        global originalJointPoses, prevOriginalJointPoses
        start_target_pose = get_pose(body_from_name(TARGET))
        # originalJointPoses = get_joint_positions(robot, jointNums)
        gripper_ori = problem.gripper_ori
        movable = get_objects_on_target(problem, poses=poses)
        attempts = 0
        
        originalObjPoses = [get_pose(obj) for obj in movable]
        # if poses is not None:
        #     originalObjPoses = [poses[obj] for obj in poses]
        originalPoses = originalJointPoses
        set_joint_positions(robot, jointNums, originalPoses)
    
        # original gripper pose in world frame
        originalGripperPose = get_link_pose(robot, gripper_link)
        newGripperPose = originalGripperPose
        hand_pose = newGripperPose #hold
        # hold = newTargetPose.value
        pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
        originalTargetPose = Pose(body_from_name(TARGET), pose)
        originalTargetPose.assign()
        newObjectPoses = [pose for pose in originalObjPoses]
        holding_grasp = problem.holding_grasp
        attachment = holding_grasp.get_attachment(robot, holding_arm)
        attachments = {attachment.child: attachment}
        count = 0

        # print(mi, ma)
        # print(jointNums)
        custom_limits = {}
        # custom_limits[jointNums[0]] =  (PLATE_GRASP_LEFT_ARM[0]+1.3, PLATE_GRASP_LEFT_ARM[0]+1.3)
        custom_limits[jointNums[0]] = (PLATE_GRASP_LEFT_ARM[0]-(math.pi/2), PLATE_GRASP_LEFT_ARM[0] + (math.pi/2))
        custom_limits[jointNums[-1]] = (PLATE_GRASP_LEFT_ARM[-1] - (math.pi/4), math.pi/2)

        # custom_limits = {jointNums[0]:(PLATE_GRASP_LEFT_ARM[0]-.3, PLATE_GRASP_LEFT_ARM[0]+.3)}
        while True:
            set_joint_positions(robot, jointNums, originalPoses)
            hand_pose = get_link_pose(robot, gripper_link)
            
            pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
            newTargetPose = Pose(body_from_name(TARGET), pose)
            newTargetPose.assign()
            for i in range(len(movable)):
                set_pose(movable[i], originalObjPoses[i])
            if attempts >= max_attempts:
                originalTargetPose.assign()
                set_joint_positions(robot, jointNums, originalPoses)
                for i in range(len(movable)):
                    set_pose(movable[i], originalObjPoses[i])
                print("Stable Config Failure")
                return None, None
            attempts += 1
            
            new_y = np.random.choice(y_range)
            newGripperPose = ((newGripperPose[0][0],
                                    new_y,
                                    newGripperPose[0][2]), gripper_ori)
            # hand_pose = newGripperPose #hold
            # hold = newTargetPose.value
            # pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
            # newTargetPose.value = pose
            # newTargetPose.assign()
            # for i in range(len(newObjectPoses)):
            #     newObjectPoses[i] = get_same_relative_pose(newObjectPoses[i], originalTargetPose.value, newTargetPose.value)
               
            for a in range(max_attempts):
                set_joint_positions(robot, jointNums, originalPoses)
                hand_pose = get_link_pose(robot, gripper_link)
                pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
                newTargetPose = Pose(body_from_name(TARGET), pose)
                newTargetPose.assign()
                for i in range(len(movable)):
                    set_pose(movable[i], originalObjPoses[i])
                newJointAngles = bi_panda_inverse_kinematics(robot, holding_arm, gripper_link, newGripperPose, custom_limits=custom_limits)
                if newJointAngles is None:
                    print("IK failed")
                    set_joint_positions(robot, jointNums, originalPoses)
                    hand_pose = get_link_pose(robot, gripper_link)
                    pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
                    newTargetPose = Pose(body_from_name(TARGET), pose)
                    newTargetPose.assign()
                    for i in range(len(movable)):
                        set_pose(movable[i], originalObjPoses[i])
                    continue
                set_joint_positions(robot, jointNums, newJointAngles)
                hand_pose = get_link_pose(robot, gripper_link)
                pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
                newTargetPose = Pose(body_from_name(TARGET), pose)
                newTargetPose.assign()
                
                if test_torque_limits(holding_arm, ptotalMass=ptotalMass, pcomR=pcomR):
                    resolutions = .03**np.ones(len(arm_joints))
                    
                    grasp = problem.holding_grasp
                    attachment = grasp.get_attachment(problem.robot, arm)
                    attachments = {attachment.child: attachment}
                    current_pose = get_joint_positions(robot, jointNums)
                   
                    jointPoses = get_joint_positions(robot, jointNums)
                    set_joint_positions(robot, jointNums, originalPoses)
                    path = plan_joint_motion(robot, jointNums, newJointAngles, attachments=attachments.values(),
                                            obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                            custom_limits=custom_limits, resolutions=resolutions,
                                            restarts=4, iterations=25, smooth=None)

                    if path is None:
                        print("Failed to find reconfig path")
                        set_joint_positions(robot, jointNums, originalPoses)
                        hand_pose = get_link_pose(robot, gripper_link)
                        pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
                        newTargetPose = Pose(body_from_name(TARGET), pose)
                        newTargetPose.assign()
                        for i in range(len(movable)):
                            set_pose(movable[i], originalObjPoses[i])
                        continue
                    set_joint_positions(robot, jointNums, newJointAngles)
                    print("here?")
                    # path.insert(0, originalPoses)
                    mt = create_trajectory(robot, arm_joints, path, bodies=problem.movable)
                    cmd = Commands(State(attachments=attachments), savers=[], commands=[mt])
                    shift = newTargetPose.value[0][1] - start_target_pose[0][1]
                    hand_pose = get_link_pose(robot, gripper_link)
                    pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
                    newTargetPose = Pose(body_from_name(TARGET), pose)
                    newTargetPose.assign()
                    for i in range(len(newObjectPoses)):
                        newObjectPoses[i] = get_same_relative_pose(newObjectPoses[i], originalTargetPose.value, newTargetPose.value)
                    originalTargetPose.value = newTargetPose.value
                    originalGripperPose = newGripperPose
                    originalTargetPose.assign()
                    
                    for i in range(len(movable)):
                        set_pose(movable[i], newObjectPoses[i])
                    return mt, shift
    return gen

def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=True):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    def fn(arm, obj, pose, grasp, reconfig=None):
        print("in base ik")
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)
        
        default_conf = arm_conf(arm, grasp.carry)
        custom_limits[-2] = (default_conf[-2]-.01, default_conf[-2]+.01)
        pose.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()
        ikfaskt_info = PANDA_LEFT_INFO
        gripper_link = link_from_name(robot, PANDA_GRIPPER_ROOTS[arm])
        grasp_conf = bi_panda_inverse_kinematics(robot, arm, arm_link, gripper_pose, max_attempts=5, max_time=3.5, obstacles=obstacles)
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles): # [obj]
            print('Grasp IK failure', grasp_conf)
            return None
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            print('Approach IK failure', approach_conf)
            #wait_if_gui()
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.01**np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits={}, resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=4, iterations2=5, smooth=None)
            if approach_path is None:
                print('Approach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path, bodies = problem.movable)
        if reconfig is not None:
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[reconfig, mt])
        else:
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])

        return (cmd,)
    return fn

##################################################
def get_ik_fn_force_aware(problem, custom_limits={}, collisions=True, teleport=True, max_attempts = 100):
    robot = problem.robot
    obstacles = problem.fixed + problem.surfaces if collisions else []
    # torque_test_left = get_torque_limits_not_exceded_test_v2(problem, 'left')
    torque_test_right = get_torque_limits_not_exceded_test_v2(problem, 'right')
    dynam_fn = get_dynamics_fn(problem)
    def fn(arm, obj, pose, grasp, reconfig=None):
        torque_test = torque_test_left if arm == 'left' else torque_test_right
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot, arm)
        # arm_link = link_from_name(robot, 'r_panda_link8')
        arm_joints = get_arm_joints(robot, arm)
        objMass = get_mass(obj)
        objPose = get_pose(obj)[0]
        default_conf = arm_conf(arm, grasp.carry)
        pose.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()
        ikfaskt_info = PANDA_RIGHT_INFO
        gripper_link = link_from_name(robot, PANDA_GRIPPER_ROOTS[arm])
        grasp_conf = None
        # custom_limits[-2] = (default_conf[-2]-.1, default_conf[-2]+.1)
        # custom_limits[-1] = (default_conf[-1]-.001, default_conf[-1]+.001)
        grasp_conf = bi_panda_inverse_kinematics(robot, arm, arm_link, gripper_pose, max_attempts=25, max_time=3.5, obstacles=obstacles)
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles): # [obj]
            print('Grasp IK failure', grasp_conf)
            return None
        if not torque_test(grasp_conf):
            print('grasp conf torques exceded')
            return None
        # if grasp_conf is None:
        print("found grasp")
        set_joint_positions(robot, arm_joints, default_conf)
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
        # approach_conf = bi_panda_inverse_kinematics(robot, arm, arm_link, approach_pose, max_attempts=5, max_time=3.5, obstacles=obstacles)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            print('Approach IK failure', approach_conf)
            print('In collision: ', any(pairwise_collision(robot, b) for b in obstacles + [obj]))
            #wait_if_gui()
            return None
        print(approach_conf)
        print(len(approach_conf), len(arm_joints))
        approach_conf = get_joint_positions(robot, arm_joints)
        if not torque_test(approach_conf):
            print('approach conf torques exceded')
            return None
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.01**np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion_force_aware(robot, arm_joints, grasp_conf, torque_test, dynam_fn, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits={}, resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_data = plan_joint_motion_force_aware(robot, arm_joints, approach_conf, torque_test, dynam_fn, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=4, iterations=10, smooth=None)
            if approach_data is None or approach_data[0] is None:
                print('Approach path failure')
                return None
            (approach_path, approach_vels, _) = approach_data
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path, bodies = problem.movable, velocities=approach_vels)
        if reconfig is not None:
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[reconfig, mt])
        else:
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])

        return (cmd,)
    return fn

def test_path_torque_constraint(robot, arm, joints, path, mass, r, test_fn):
    reset = get_joint_positions(robot, joints)
    for conf in path:
        set_joint_positions(robot, joints, conf)
        if not test_fn(arm, ptotalMass=mass, pcomR=r):
            print('conf torques exceded in path')
            set_joint_positions(robot, joints, reset)
            return True
    set_joint_positions(robot, joints, reset)
    return False

##################################################
poses = {}
def get_ik_ir_gen(problem, max_attempts=5, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    holding_arm = problem.holding_arm
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=1, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)
    if holding_arm is not None:
        torque_test = get_torque_limits_not_exceded_test(problem)
        stepSize = 0.01
        safe_conf_fn = get_sample_stable_holding_conf_gen_v2(problem, stepSize=stepSize, max_attempts=5)
        comR = []

        gripper_link = link_from_name(problem.robot, PANDA_GRIPPER_ROOTS[problem.holding_arm])
        plate_width = problem.target_width
        joints = get_arm_joints(problem.robot, problem.holding_arm)
        originalTPose = Pose(body_from_name(TARGET), get_pose(body_from_name(TARGET)))
    def gen(*inputs):
        print("in bi manual ik ir")
        global originalJointPoses, prevOriginalJointPoses
        originalJointPoses = get_joint_positions(problem.robot, joints)
        a, b, p, g = inputs
        attempts = 0
        global poses
        originalObjectPs = poses
        hand_pose = get_link_pose(problem.robot, gripper_link)
        pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
        originalTargetPose = Pose(body_from_name(TARGET), pose)
        originalP = p
        while True:
            reconfig = None
            if max_attempts <= attempts:
                # if poses.get(b, None) is not None:
                #     poses.pop(b)
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            # if holding_arm is not None:
            set_joint_positions(problem.robot, joints, originalJointPoses)
            # originalTargetPose.assign()
            hand_pose = get_link_pose(problem.robot, gripper_link)
            pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
            current_pose = Pose(body_from_name(TARGET), pose)
            current_pose.assign()
            target_overlap = is_pose_on_r(p.value, body_from_name(TARGET))
            if (target_overlap):
                poses[b] = p.value
            comR, totalMass = get_COM(bodies = [], poses=poses, b2=body_from_name(TARGET))
            if (not torque_test(a, poses, pcomR=comR, ptotalMass=totalMass)):
                reconfig, shift = safe_conf_fn(a, obj = b, pcomR=comR, ptotalMass=totalMass, poses = poses)
                if reconfig is None:
                    set_joint_positions(problem.robot, joints, originalJointPoses)
                    originalTargetPose.assign()
                    for obj in originalObjectPs:
                        tempP = Pose(obj, originalObjectPs[obj])
                        tempP.assign()
                        poses[obj] = originalObjectPs[obj]
                    continue
                poseOnR = {}
                for obj in poses:
                    if obj != b and is_pose_on_r(poses[obj], body_from_name(TARGET)):
                        poseOnR[obj] = poses[obj]
                hand_pose = get_link_pose(problem.robot, gripper_link)
                pose=((hand_pose[0][0], hand_pose[0][1] - (problem.target_width / 2) - 0.07, hand_pose[0][2]), quat_from_euler((0,0,math.pi/2)))
                for obj in poseOnR:
                    new_pose = Pose(obj, get_same_relative_pose(p.value, originalTargetPose.value, pose))
                    new_pose.assign()
                newTargetPose = Pose(body_from_name(TARGET), pose)

                if target_overlap:
                    new_pose = get_same_relative_pose(p.value, originalTargetPose.value, pose)
                    p.value = new_pose
                    poses[b] = p.value
                    p.assign()
                newTargetPose.assign()

            ir_generator = ir_sampler(a,b,p,g)

            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                if holding_arm is not None:
                    originalP.assign()
                    poses = {}
                    for obj in originalObjectPs:
                        tempP = Pose(obj, originalObjectPs[obj])
                        tempP.assign()
                        poses[obj] = originalObjectPs[obj]
                    p = originalP
                    p.assign()
                    set_joint_positions(problem.robot, joints, originalJointPoses)
                    originalTargetPose.assign()
                continue
            ik_outputs = ik_fn(a, b, p, g, reconfig)
            if ik_outputs is None:
                if holding_arm is not None:
                    originalP.assign()
                    p = originalP
                    poses = {}
                    for obj in originalObjectPs:
                        tempP = Pose(obj, originalObjectPs[obj])
                        tempP.assign()
                        poses[obj] = originalObjectPs[obj]
                    set_joint_positions(problem.robot, joints, originalJointPoses)
                    originalTargetPose.assign()
                continue
            print('IK attempts:', attempts)

            yield ir_outputs + ik_outputs
            return
            #if not p.init:
            #    return
    return gen


def get_ik_ir_gen_no_reconfig(problem, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    holding_arm = problem.holding_arm
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=1, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)

    def gen(*inputs):
        print("in no reconfig ik ir")
        a, b, p, g = inputs
        ir_generator = ir_sampler(a,b,p,g)

        attempts = 0
        while True:
            print("ik loop no reconfig")
            reconfig = None
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1

            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                print("no ir outputs")
                continue
            ik_outputs = ik_fn(a, b, p, g, reconfig)
            if ik_outputs is None:
                print("no ik outputs")
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            #if not p.init:
            #    return
    return gen

def get_ik_ir_gen_force_aware(problem, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    holding_arm = problem.holding_arm
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=1, **kwargs)
    ik_fn = get_ik_fn_force_aware(problem, teleport=teleport, **kwargs)

    def gen(*inputs):
        print("in force aware ik ir")
        a, b, p, g = inputs
        ir_generator = ir_sampler(a,b,p,g)

        attempts = 0
        while True:
            print("force aware ik ir loop")
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1

            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                print("no ir")
                continue
            ik_outputs = ik_fn(a, b, p, g)
            if ik_outputs is None:
                print("no ik")
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            #if not p.init:
            #    return
    return gen

##################################################

def get_motion_gen(problem, custom_limits={}, collisions=True, teleport=False):
    # TODO: include fluents
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    def fn(bq1, bq2, fluents=[]):
        saver.restore()
        bq1.assign()
        if teleport:
            path = [bq1, bq2]
        elif is_drake_pr2(robot):
            raw_path = plan_joint_motion(robot, bq2.joints, bq2.values, attachments=[],
                                         obstacles=obstacles, custom_limits=custom_limits, self_collisions=SELF_COLLISIONS,
                                         restarts=4, iterations=50, smooth=50)
            if raw_path is None:
                print('Failed motion plan!')
                #set_renderer(True)
                #for bq in [bq1, bq2]:
                #    bq.assign()
                #    wait_if_gui()
                return None
            path = [Conf(robot, bq2.joints, q) for q in raw_path]
        else:
            goal_conf = base_values_from_pose(bq2.value)
            raw_path = plan_base_motion(robot, goal_conf, BASE_LIMITS, obstacles=obstacles)
            if raw_path is None:
                print('Failed motion plan!')
                return None
            path = [Pose(robot, pose_from_base_values(q, bq1.value)) for q in raw_path]
        bt = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
        return (cmd,)
    return fn

##################################################

def get_press_gen(problem, max_attempts=25, learned=True, teleport=False):
    robot = problem.robot
    fixed = get_fixed_bodies(problem)

    def gen(arm, button):
        fixed_wo_button = list(filter(lambda b: b != button, fixed))
        pose = get_pose(button)
        grasp_type = 'side'

        link = get_gripper_link(robot, arm)
        default_conf = get_carry_conf(arm, grasp_type)
        joints = get_arm_joints(robot, arm)

        presses = get_x_presses(button)
        approach = ((APPROACH_DISTANCE, 0, 0), unit_quat())
        while True:
            for _ in range(max_attempts):
                press_pose = random.choice(presses)
                gripper_pose = multiply(pose, invert(press_pose)) # w_f_g = w_f_o * (g_f_o)^-1
                #approach_pose = gripper_pose # w_f_g * g_f_o * o_f_a = w_f_a
                approach_pose = multiply(gripper_pose, invert(multiply(press_pose, approach)))

                if learned:
                    base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp_type)
                else:
                    base_generator = uniform_pose_generator(robot, gripper_pose)
                set_joint_positions(robot, joints, default_conf)
                set_pose(robot, next(base_generator))
                raise NotImplementedError('Need to change this')
                if any(pairwise_collision(robot, b) for b in fixed):
                    continue

                approach_movable_conf = sub_inverse_kinematics(robot, joints[0], link, approach_pose)
                #approach_movable_conf = inverse_kinematics(robot, link, approach_pose)
                if (approach_movable_conf is None) or any(pairwise_collision(robot, b) for b in fixed):
                    continue
                approach_conf = get_joint_positions(robot, joints)

                gripper_movable_conf = sub_inverse_kinematics(robot, joints[0], link, gripper_pose)
                #gripper_movable_conf = inverse_kinematics(robot, link, gripper_pose)
                if (gripper_movable_conf is None) or any(pairwise_collision(robot, b) for b in fixed_wo_button):
                    continue
                grasp_conf = get_joint_positions(robot, joints)
                bp = Pose(robot, get_pose(robot)) # TODO: don't use this

                if teleport:
                    path = [default_conf, approach_conf, grasp_conf]
                else:
                    control_path = plan_direct_joint_motion(robot, joints, approach_conf,
                                                     obstacles=fixed_wo_button, self_collisions=SELF_COLLISIONS)
                    if control_path is None: continue
                    set_joint_positions(robot, joints, approach_conf)
                    retreat_path = plan_joint_motion(robot, joints, default_conf,
                                                     obstacles=fixed, self_collisions=SELF_COLLISIONS)
                    if retreat_path is None: continue
                    path = retreat_path[::-1] + control_path[::-1]
                mt = Trajectory(Conf(robot, joints, q) for q in path)
                yield (bp, mt)
                break
            else:
                yield None
    return gen

#####################################

def control_commands(commands, robot = 1, **kwargs):
    # wait_if_gui('Control?')
    disable_real_time()
    enable_gravity()

    for i, command in enumerate(commands):
        print(i, command)
        print(type(command))

        command.control(*kwargs)


class State(object):
    def __init__(self, attachments={}, cleaned=set(), cooked=set()):
        self.poses = {body: Pose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked
        self.bodies = get_bodies()
        self.body_names = [get_body_name(body) for body in self.bodies]


    def assign(self):
        for attachment in self.attachments.values():
            #attach.attachment.assign()
            attachment.assign()

def apply_commands(state, commands, time_step=None, pause=False, **kwargs):
    #wait_if_gui('Apply?')
    prev_joints = ()
    for i, command in enumerate(commands):
        print(i, command)
        print(type(command))
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
            else:
                wait_for_duration(time_step)
            # p.stepSimulation()
        if pause:
            wait_if_gui()
    return state

#####################################

def get_target_point(conf):
    # TODO: use full body aabb
    robot = conf.body
    link = link_from_name(robot, 'torso_lift_link')
    #link = BASE_LINK
    # TODO: center of mass instead?
    # TODO: look such that cone bottom touches at bottom
    # TODO: the target isn't the center which causes it to drift
    with BodySaver(conf.body):
        conf.assign()
        lower, upper = get_aabb(robot, link)
        center = np.average([lower, upper], axis=0)
        point = np.array(get_group_conf(conf.body, 'base'))
        #point[2] = upper[2]
        point[2] = center[2]
        #center, _ = get_center_extent(conf.body)
        return point


def get_target_path(trajectory):
    # TODO: only do bounding boxes for moving links on the trajectory
    return [get_target_point(conf) for conf in trajectory.path]


def get_objects_on_target(problem, target = TARGET, poses = None):
    obs = []
    if isinstance(target, str):
        target = body_from_name(target)
    if poses is not None:
        for obj in poses:
            if check_overlap(poses[obj][0], target):
                obs.append(obj)
    else:
        for obj in problem.movable:
            if check_overlap(get_pose(obj)[0], target):
                obs.append(obj)
    return obs


def check_joint_deltas(q1, q2, limit = math.pi/100):
    for i in range(len(q1)):
        if abs(q1[i] - q2[i]) > limit:
            return True
    return False

def get_linear_path(robot, arm, target_link, target_pose, start_pose, attachments = None, max_iterations = 5, max_attempts=25, gripper_ori=None, custom_limits={}):
    orientation = gripper_ori if gripper_ori else start_pose[1]
    diff = (target_pose[0][0] - start_pose[0][0],
            target_pose[0][1] - start_pose[0][1])
    target_joints = [joint_from_name(robot, name) for name in BI_PANDA_GROUPS[arm_from_arm(arm)]]
    reset = get_joint_positions(robot, target_joints)
    dy = diff[1] / max_iterations
    waypoints = [((start_pose[0][0], start_pose[0][1] + (dy), start_pose[0][2]), orientation) for i in range(max_iterations)]
    waypoints.append(target_pose)
    path = []
    arm_joints = get_arm_joints(robot, arm)
    old_joints = get_joint_positions(robot, target_joints)

    for waypoint in waypoints:
        old_joints = get_joint_positions(robot, target_joints)
        for j in range(max_attempts):
            if j > max_attempts:
                return None
            new_joints = bi_panda_inverse_kinematics(robot, arm, target_link, waypoint, custom_limits=custom_limits)
            if new_joints is None or not check_x_y_rot(get_link_pose(robot, target_link)[1], waypoint[1]) or check_joint_deltas(new_joints, old_joints):
                print(is_pose_close(get_link_pose(robot, target_link), waypoint))
                continue
            path.append(new_joints)
            set_joint_positions(robot, arm_joints, new_joints)
            break

    current_pose = get_joint_positions(robot, arm_joints)

    resolutions = 0.05**np.ones(len(arm_joints))
    realPath = []
    return path
    return plan_direct_joint_motion(robot, arm_joints, waypoints=path, end_conf=waypoints[-1], attachments=attachments.values(),
                                                  self_collisions=SELF_COLLISIONS, resolutions=resolutions/2., custom_limits=custom_limits)
    # for point in path:
    #     realPath += plan_joint_motion(robot, arm_joints, point, attachments=attachments.values(),
    #                                             self_collisions=SELF_COLLISIONS,
    #                                           resolutions=resolutions,
    #                                           restarts=4, iterations=25, smooth=25)
    #     set_joint_positions(robot, arm_joints, point)
    # return realPath

def check_x_y_rot(quat1, quat2, limit=1e-2*math.pi):
    euler1 = euler_from_quat(quat1)
    euler2 = euler_from_quat(quat2)
    dx = abs(euler1[0] - euler2[0])
    dy = abs(euler1[1] - euler2[1])
    return dy<limit and dx < limit


def hack_table_place(problem, state):
    torque_test = get_torque_limits_not_exceded_test(problem)
    robot = problem.robot
    jointNums = get_arm_joints(robot, problem.holding_arm)
    jointPoses = get_joint_positions(robot, jointNums)
    gripper_link = get_gripper_link(robot, problem.holding_arm)
    gripper_pose = get_link_pose(robot, gripper_link)
    table_pose = get_pose(problem.post_goal)
    above = table_pose[0][2] + 0.01
    diff = abs(gripper_pose[0][2] - above)
    custom_limits = {}
    max_limits = []
    for jointNum in jointNums:
        max_limits.append(get_max_force(problem.robot, jointNum))
    # custom_limits[jointNums[0]] =  (PLATE_GRASP_LEFT_ARM[0]+1.3, PLATE_GRASP_LEFT_ARM[0]+1.3)
    custom_limits[jointNums[0]] = (PLATE_GRASP_LEFT_ARM[0]-(math.pi/4), PLATE_GRASP_LEFT_ARM[0] + (math.pi/2))
    custom_limits[jointNums[-1]] = (PLATE_GRASP_LEFT_ARM[-1] - (math.pi/2), math.pi/2 - .1)
    gripper_pose = ((gripper_pose[0][0], gripper_pose[0][1] + 0.06,
                    gripper_pose[0][2] + 0.06),problem.gripper_ori)
    newJointPoses = None
    repeat_count = 0
    count = 0
    # while (newJointPoses is None ) and count < 200:
    #     count += 1
    #     newJointPoses = bi_panda_inverse_kinematics(robot, problem.holding_arm, gripper_link, gripper_pose, custom_limits=custom_limits)
    if newJointPoses is not None:
        set_joint_positions_torque(robot, jointNums, newJointPoses)
    # print(jointNums, newJointPoses)
    # for body in problem.movable:
    #     if is_b1_on_b2(body, body_from_name(TARGET)):
    #         add_fixed_constraint(body, body_from_name(TARGET))
    wait_for_duration(0.5)
    if newJointPoses is not None:
        set_joint_positions(robot, jointNums, newJointPoses)
    dT = .07
    newJointPoses = get_joint_positions(robot, jointNums)
    end = newJointPoses[0]
    newJointPoses = list(newJointPoses)
    count = 0
    while count < 300 and newJointPoses[0] < end + math.pi:
        prevPose = newJointPoses
        newJointPoses[0] += dT
        count += 1
        print(count, newJointPoses[0])
        set_joint_positions_torque(robot, jointNums, newJointPoses)
        newJointPoses = list(get_joint_positions(robot, jointNums))
        wait_for_duration(0.05)
        gripper_pose = get_link_pose(robot, gripper_link)
    gripper_pose = ((gripper_pose[0][0], gripper_pose[0][1],
                    gripper_pose[0][2]), gripper_pose[1])
    wait_for_duration(1)
    # for body in problem.movable:
    #     if is_b1_on_b2(body, body_from_name(TARGET)):
    #         remove_fixed_constraint(body, body_from_name(TARGET), -1)
    detach = Detach(problem.robot, problem.holding_arm, body_from_name(TARGET))
   
    count = 0
    if newJointPoses is not None and count < 100:
        set_joint_positions(robot, jointNums, newJointPoses)
        count += 1
    remove_fixed_constraint(body_from_name(TARGET), robot, gripper_link)
    print("detatched!")
    new_joint_poses = list(newJointPoses)
    while newJointPoses[0] > end:
        newJointPoses[0] -= dT
        set_joint_positions_torque(robot, jointNums, newJointPoses)
        newJointPoses = list(get_joint_positions(robot, jointNums))
        wait_for_duration(0.01)
        # p.stepSimulation()
        gripper_pose = get_link_pose(robot, gripper_link)

def check_eq_conf(q1, q2, eps=[1e-2]):
    if len(q1) != len(q2):
        raise "Bad conf passed to check eq"
    for i in range(len(q1)):
        if abs(q2[i] - q1[i]) > eps[i]:
            return False
    return True


num_steps = 3
def create_graph(start_conf, end_conf, step_size = [], root = None):
    if check_eq_conf(start_conf, end_conf):
        return root
    if len(step_size) == 0:
        for i in range(len(start_conf)):
            step_size.append((end_conf[i] - start_conf[i])/num_steps)
    if root == None:
        root = Node(start_conf)
    for i in range(len(start_conf)):
        if check_eq_conf([start_conf[i]], [end_conf[i]]):
            continue
        child_conf = tuple([(start_conf[j] + step_size[j]) if i == j else start_conf[j] for j in range(len(start_conf))])
        child_node = Node(child_conf, root)
        child_node.children = create_graph(child_conf, end_conf, step_size, child_node)
    return root


class Node:
    def __init__(self, conf, parent = None):
        self.conf = conf
        self.parent = parent
        self.children = []
        self.score = 0
    def add_child(conf):
        self.children.append(conf)

class PriorityQueue:
    def __init__(self, target_conf, robot, joint_nums, arm):
        self.queue = []
        self.target = euler_from_quat(target_conf)
        self.robot = robot
        self.joint_nums = joint_nums
        self.gripper_link = get_gripper_link(robot, arm)
    def enqueue(self, node):
        hold = get_joint_positions(self.robot, self.joint_nums)
        set_joint_positions(self.robot, self.joint_nums, node.conf)
        ori = get_link_pose(self.robot, self.gripper_link)[1]
        set_joint_positions(self.robot, self.joint_nums, hold)
        ori = euler_from_quat(ori)
        dist = math.sqrt((ori[0] - self.target[0])**2 + (ori[1] - self.target[1])**2)
        node.score = 1/dist if dist > 0 else 0
        for i in range(len(self.queue)):
            if node.score > self.queue[i].score:
                self.queue.insert(i, node)
                return
        self.queue.append(node)
    def dequeue(self):
        return self.queue.pop(0)

    def peek(self):
        return self.queue[0].score


def find_waypoints_stable_gripper(start_conf, end_conf, gripper_ori, joint_nums, robot, arm):
    # generate a graph such that each node is a position with one
    # joint moved one step closer to the goal pose by some stepsize sub angle of total (ie theta/25)
    root = create_graph(start_conf, end_conf)
    # use 1/diff in gripper ori as heuristic
    q = PriorityQueue(gripper_ori, robot, joint_nums, arm)
    q.enqueue(root)
    while len(q.queue) > 0:
        node = q.dequeue()
        if check_eq_conf(node.conf, end_conf):
            path = []
            temp = node
            while temp.parent is not None:
                path.insert(0, temp.conf)
                temp = temp.parent
            return path
            path.insert(end_conf)
        for child in node.children:
            q.enqueue(child)
    return None

    # use A* search to find optimal path moving one joint at at time until goal is reached
    # based on heuristic


def greedy_find_waypoints_stable_gripper(start_conf, end_conf, target_ori, joint_nums, robot, arm, steps):
    path = []
    gripper_link = get_gripper_link(robot, arm)
    step_size = [(start_conf[i] - end_conf[i])/steps for i in range(len(start_conf))]
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",check_eq_conf(start_conf, end_conf))
    while not check_eq_conf(start_conf, end_conf, step_size) and len(path) < steps:
        print(len(path))
        for step in range(num_steps):
            scores = []
            confs = []
            for i in range(len(start_conf)):
                if check_eq_conf([start_conf[i]], [end_conf[i]], [.01]):
                    continue
                new_conf = start_conf
                new_conf = tuple([start_conf[j] + (step_size[j]) if j == i else start_conf[j] for j in range(len(step_size))])
                confs.append(new_conf)
                hold = get_joint_positions(robot, joint_nums)
                set_joint_positions(robot, joint_nums, new_conf)
                gripperOri = get_link_pose(robot, gripper_link)[1]
                score = math.sqrt((gripperOri[0] - target_ori[0])**2 + (gripperOri[1] - target_ori[1])**2)
                scores.append(score)
            index = np.argmax(scores)
            path.append(confs[index])
            start_conf = confs[index]
    print("(((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((",path)
    return path


def get_dynamics_fn_v2(problem):
    from numpy.linalg import inv

    def min_jerk(pos=None, dur=None, vel=None, acc=None, psg=None):

        N = pos.shape[0]					# number of point
        D = pos.shape[1]					# dimensionality

        if not vel:
            vel = np.zeros((2,D))			# default endpoint vel is 0
        if not acc:
            acc = np.zeros((2,D))			# default endpoint acc is 0

        t0 = np.array([[0],[dur]])

        if not psg:					# passage times unknown, optimize
            if N > 2:
                psg = np.arange(dur/(N-1), dur-dur/(N-1)+1, dur/(N-1)).T
                func = lambda psg_: mjCOST(psg_, pos, vel, acc, t0)
                psg = scipy.optimize.fmin(func = func, x0 = psg)
            else:
                psg = []

        print(psg)
        trj = mjTRJ(psg, pos, vel, acc, t0, dur)

        return trj, psg

    ################################################################
    ###### Compute jerk cost
    ################################################################

    def mjCOST(t, x, v0, a0, t0):

        N = max(x.shape)
        D = min(x.shape)

        v, a = mjVelAcc(t, x, v0, a0, t0)
        aa   = np.concatenate(([a0[0][:]], a, [a0[1][:]]), axis = 0)
        aa0  = aa[0:N-1][:]
        aa1  = aa[1:N][:]
        vv   = np.concatenate(([v0[0][:]], v, [v0[1][:]]), axis = 0)
        vv0  = vv[0:N-1][:]
        vv1  = vv[1:N][:]
        tt   = np.concatenate((t0[0]   , t, t0[1]   ), axis = 0)
        T    = np.diff(tt)[np.newaxis].T*np.ones((1,D))
        xx0  = x[0:N-1][:]
        xx1  = x[1:N][:]

        j=3*(3*aa0**2*T**4-2*aa0*aa1*T**4+3*aa1**2*T**4+24*aa0*T**3*vv0- \
            16*aa1*T**3*vv0 + 64*T**2*vv0**2 + 16*aa0*T**3*vv1 - \
            24*aa1*T**3*vv1 + 112*T**2*vv0*vv1 + 64*T**2*vv1**2 + \
            40*aa0*T**2*xx0 - 40*aa1*T**2*xx0 + 240*T*vv0*xx0 + \
            240*T*vv1*xx0 + 240*xx0**2 - 40*aa0*T**2*xx1 + 40*aa1*T**2*xx1- \
            240*T*vv0*xx1 - 240*T*vv1*xx1 - 480*xx0*xx1 + 240*xx1**2)/T**5

        J = sum(sum(abs(j)))

        return J

    ################################################################
    ###### Compute trajectory
    ################################################################

    def mjTRJ(tx, x, v0, a0, t0, P):

        N = max(x.shape)
        D = min(x.shape)
        X_list = []
        aa = a0
        vv = v0
        tt = t0
        if len(tx) > 0:
            v, a = mjVelAcc(tx, x, v0, a0, t0)
            aa   = np.concatenate(([a0[0][:]],  a, [a0[1][:]]), axis = 0)
            vv   = np.concatenate(([v0[0][:]],  v, [v0[1][:]]), axis = 0)
            tt   = np.concatenate((t0[0], tx,t0[1]), axis = 0)

        ii = 0
        for i in range(1,int(P)+1):
            t = (i-1)/(P-1)*(t0[1]-t0[0]) + t0[0]
            if t > tt[ii+1]:
                ii = ii+1
            T = (tt[ii+1]-tt[ii])*np.ones((1,D))
            t = (t-tt[ii])*np.ones((1,D))
            aa0 = aa[ii][:]
            aa1 = aa[ii+1][:]
            vv0 = vv[ii][:]
            vv1 = vv[ii+1][:]
            xx0 = x[ii][:]
            xx1 = x[ii+1][:]

            tmp = aa0*t**2/2 + t*vv0 + xx0 + t**4*(3*aa0*T**2/2 - aa1*T**2 + \
                                                8*T*vv0 + 7*T*vv1 + 15*xx0 - 15*xx1)/T**4 + \
                t**5*(-(aa0*T**2)/2 + aa1*T**2/2 - 3*T*vv0 - 3*T*vv1 - 6*xx0+ \
                        6*xx1)/T**5 + t**3*(-3*aa0*T**2/2 + aa1*T**2/2 - 6*T*vv0 - \
                                            4*T*vv1 - 10*xx0 + 10*xx1)/T**3
            X_list.append(tmp)

        X = np.concatenate(X_list)

        return X, aa, vv

    ################################################################
    ###### Compute intermediate velocities and accelerations
    ################################################################

    def mjVelAcc(t, x, v0, a0, t0):

        N = max(x.shape)
        D = min(x.shape)
        mat = np.zeros((2*N-4,2*N-4))
        vec = np.zeros((2*N-4,D))
        tt = np.concatenate((t0[0], t, t0[1]), axis = 0)

        for i in range(1, 2*N-4+1, 2):

            ii = int(math.ceil(i/2.0))
            T0 = tt[ii]-tt[ii-1]
            T1 = tt[ii+1]-tt[ii]

            tmp = [-6/T0, -48/T0**2, 18*(1/T0+1/T1), \
                72*(1/T1**2-1/T0**2), -6/T1, 48/T1**2]

            if i == 1:
                le = 0
            else:
                le = -2

            if i == 2*N-5:
                ri = 1
            else:
                ri = 3

            mat[i-1][i+le-1:i+ri] = tmp[3+le-1:3+ri]
            vec[i-1][:] = 120*(x[ii-1][:]-x[ii][:])/T0**3 \
                        + 120*(x[ii+1][:]-x[ii][:])/T1**3

        for i in range(2, 2*N-4+1, 2):

            ii = int(math.ceil(i/2.0))
            T0 = tt[ii]-tt[ii-1]
            T1 = tt[ii+1]-tt[ii]

            tmp = [48/T0**2, 336/T0**3, 72*(1/T1**2-1/T0**2), \
                384*(1/T1**3+1/T0**3), -48/T1**2, 336/T1**3]

            if i == 2:
                le = -1
            else:
                le = -3

            if i == 2*N-4:
                ri = 0
            else:
                ri = 2

            mat[i-1][i+le-1:i+ri] = tmp[4+le-1:4+ri]
            vec[i-1][:] = 720*(x[ii][:]-x[ii-1][:])/T0**4 \
                        + 720*(x[ii+1][:]-x[ii][:])/T1**4

        T0 = tt[1] - tt[0]
        T1 = tt[N-1]-tt[N-2]
        vec[0][:] = vec[0][:] +  6/T0*a0[0][:]    +  48/T0**2*v0[0][:]
        vec[1][:] = vec[1][:] - 48/T0**2*a0[0][:] - 336/T0**3*v0[0][:]
        vec[2*N-6][:] = vec[2*N-6][:] +  6/T1*a0[1][:]    -  48/T1**2*v0[1][:]
        vec[2*N-5][:] = vec[2*N-5][:] + 48/T1**2*a0[1][:] - 336/T1**3*v0[1][:]

        avav = inv(mat).dot(vec)
        a = avav[0:2*N-4:2][:]
        v = avav[1:2*N-4:2][:]

        return v, a
    return min_jerk