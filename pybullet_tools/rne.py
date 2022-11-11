import numpy as np
from copy import deepcopy

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def adjoint(a):
    # https://physics.stackexchange.com/a/244364
    R = a[:3, :3]
    t = a[:3, 3]
    return np.block([[R, skew(t) @ R],
                     [np.zeros((3, 3), dtype=a.dtype), R]])

def spatial_inertia(m, c, I):
    C = skew(c)
    return np.block([[m * np.eye(3), m * C.transpose()],
                    [m * C, I + m * C @ C.transpose()]])

def crm(v):
    vv = v.transpose()[0]
    return np.block([[skew(vv[3:]), skew(vv[:3])],
                     [np.zeros((3,3)), skew(vv[3:])]])

def crf(v):
    return -crm(v).transpose()

def parent(link_id):
    return link_id - 1

def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    TF = np.array([[np.cos(q), -np.sin(q), 0, a],
                 [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                 [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                 [0, 0, 0, 1]])

    return TF

def get_parent_to_child_transform(joint_angles, parent_link, child_link):
    dh_params = [[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi/2, joint_angles[1]],
                 [0, 0.316, np.pi/2, joint_angles[2]],
                 [0.0825, 0, np.pi/2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi/2, joint_angles[4]],
                 [0, 0, np.pi/2, joint_angles[5]],
                 [0.088, 0.0, np.pi/2, joint_angles[6]],
                 [0, 0.107, 0, 0]]

    T = np.eye(4)
    for i in range(parent_link, child_link):
        if i < 8:
            T = np.matmul(T, get_tf_mat(i, dh_params))
        else:
            T = np.matmul(T, np.eye(4))

    return np.linalg.inv(T)

link_inertia_I = """
      <inertia ixx="7.0337e-01" ixy="-1.3900e-04" ixz="6.7720e-03" iyy="7.0661e-01" iyz="1.9169e-02" izz="9.1170e-03"/>
      <inertia ixx="7.9620e-03" ixy="-3.9250e-03" ixz="1.0254e-02" iyy="2.8110e-02" iyz="7.0400e-04" izz="2.5995e-02"/>
      <inertia ixx="3.7242e-02" ixy="-4.7610e-03" ixz="-1.1396e-02" iyy="3.6155e-02" iyz="-1.2805e-02" izz="1.0830e-02"/>
      <inertia ixx="2.5853e-02" ixy="7.7960e-03" ixz="-1.3320e-03" iyy="1.9552e-02" iyz="8.6410e-03" izz="2.8323e-02"/>
      <inertia ixx="3.5549e-02" ixy="-2.1170e-03" ixz="-4.0370e-03" iyy="2.9474e-02" iyz="2.2900e-04" izz="8.6270e-03"/>
      <inertia ixx="1.9640e-03" ixy="1.0900e-04" ixz="-1.1580e-03" iyy="4.3540e-03" iyz="3.4100e-04" izz="5.4330e-03"/>
      <inertia ixx="1.2516e-02" ixy="-4.2800e-04" ixz="-1.1960e-03" iyy="1.0027e-02" iyz="-7.4100e-04" izz="4.8150e-03"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
""".split('\n')[1:-1]

def inertia_urdf_to_matrix(line):
    data = line.split('/')[0].split(' ')[7:]
    nums = []
    for dd in data:
        nums.append(float(dd.split('"')[1]))
    I = [[nums[0], nums[1], nums[2]], [nums[1], nums[3], nums[4]], [nums[2], nums[4], nums[5]]]
    return I

def new_inertia(r, m):
    nums = [0.0]*6
    # ixx
    nums[0] = m * (r[1]**2 + r[2]**2)
    # ixy
    nums[1] = -m * (r[0] * r[1])
    # ixz
    nums[2] = -m * (r[0] * r[2])
    # iyy
    nums[3] = m * (r[0]**2 + r[2]**2)
    # iyz
    nums[4] = -m * (r[1]*r[2])
    # izz
    nums[5] = m * (r[0]**2 + r[1]**2)
    I = [[nums[0], nums[1], nums[2]], [nums[1], nums[3], nums[4]], [nums[2], nums[4], nums[5]]]
    return I

inertia_matrices = []
for line in link_inertia_I:
    I = inertia_urdf_to_matrix(line)
    inertia_matrices.append(np.array(I))

link_inertia_c = """
    <origin rpy="0 0 0" xyz="3.875e-03 2.081e-03 -0.1750"/>
    <origin rpy="0 0 0" xyz="-3.141e-03 -2.872e-02 3.495e-03"/>
    <origin rpy="0 0 0" xyz="2.7518e-02 3.9252e-02 -6.6502e-02"/>
    <origin rpy="0 0 0" xyz="-5.317e-02 1.04419e-01 2.7454e-02"/>
    <origin rpy="0 0 0" xyz="-1.1953e-02 4.1065e-02 -3.8437e-02"/>
    <origin rpy="0 0 0" xyz="6.0149e-02 -1.4117e-02 -1.0517e-02"/>
    <origin rpy="0 0 0" xyz="1.0517e-02 -4.252e-03 6.1597e-02"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <origin rpy="0 0 0" xyz="0 0 0"/>
""".split('\n')[1:-1]

cs = []
for line in link_inertia_c:
    data = list(map(float, line.split('/')[0].split('xyz=')[-1].split('"')[1].split(' ')))
    cs.append(np.array(data))

link_inertia_m = """
    <mass value="4.970684"/>
    <mass value="0.646926"/>
    <mass value="3.228604"/>
    <mass value="3.587895"/>
    <mass value="1.225946"/>
    <mass value="1.666555"/>
    <mass value="7.35522e-01"/>
    <mass value="0.0"/>
    <mass value="0.68"/>
    <mass value="0.0"/>
""".split('\n')[1:-1]

ms = []
for line in link_inertia_m:
    data = line.split('/')[0].split('="')[-1].split('"')[0]
    ms.append(float(data))

def get_inertia_matricies():
    global inertia_matrices
    return inertia_matrices

def add_inertia_matrix(I):
    global inertia_matrices
    inertia_matrices.append(I)

def remove_inertia_matrix():
    global inertia_matrices
    inertia_matrices.pop()

def get_ms_global():
    global ms
    return ms

def add_mass_to_ms_global(mass):
    global ms
    ms[-1] = mass

def get_cs_global():
    global cs
    return cs

def add_com_to_cs_global(c):
    global cs
    cs[-1] = c

has_payload = False

def get_has_payload():
    global has_payload
    return has_payload

def set_has_payload(val):
    global has_payload
    has_payload = val

def add_payload(r, m):
    hand_width = 0.14 #m
    remove_payload()
    if m > 0:
        set_has_payload(True)
        add_mass_to_ms_global(m)
        I = new_inertia([0,0,hand_width + 0.025], m)
        add_inertia_matrix(I)

def remove_payload():
    if get_has_payload():
        add_mass_to_ms_global(0.0)
        add_com_to_cs_global([0.0,0.0,0.0])
        remove_inertia_matrix()
        set_has_payload(False)


def rne(q, qd, qdd):
    a_grav = np.array([[0], [0], [-9.81], [0], [0], [0]])
    v = [[], [], [], [], [], [], [], [], [], []]  # 9 links
    a = [[], [], [], [], [], [], [], [], [], []]
    f = [[], [], [], [], [], [], [], [], [], []]
    tau = [np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, 0.0, 0.0, 0.0]
    Xups = []
    originalN = len(q)
    q = tuple(deepcopy(q)) + (0.0, 0.0, 0.0)
    qd = tuple(deepcopy(qd)) + (0.0, 0.0, 0.0)
    qdd = tuple(deepcopy(qdd)) + (0.0, 0.0, 0.0)
#     S = np.array([[0], [0], [0], [0], [0], [1]])

    nb = 9 + int(get_has_payload())  # number of links minus link0 which is static plus hand plus link8/payloadlink
    ms = get_ms_global()
    cs = get_cs_global()

    inertia_matrices = get_inertia_matricies()
    # forward pass
    for i in range(1, nb + 1):
        py_idx = i - 1

    #     Xj = rotz(q[py_idx])
        vJ = np.array([[0], [0], [0], [0], [0], [qd[py_idx]]])

    #     Xup_i = Xj @ adjoint(get_parent_to_child_transform(q, parent(i), i))
        Xup_i = get_parent_to_child_transform(q, parent(i), i)

        if i == 7:
            Xup_i[2, 3] = 0
        if parent(i) == 0:
    #         Xup_i = get_parent_to_child_transform(q, parent(i), i)

            v[py_idx] = vJ
            a[py_idx] = adjoint(Xup_i) @ (-a_grav) + np.array([[0], [0], [0], [0], [0], [qdd[py_idx]]])
        else:
    #         Xup_i = get_parent_to_child_transform(q, parent(i), i) @ Xups[parent(i) - 1]
            v[py_idx] = adjoint(Xup_i) @ v[parent(i) - 1] + vJ
            a[py_idx] = adjoint(Xup_i) @ a[parent(i) - 1] + np.array([[0], [0], [0], [0], [0], [qdd[py_idx]]]) + crm(v[py_idx]) @ vJ

        Xups.append(Xup_i)

        I = spatial_inertia(ms[py_idx], cs[py_idx], inertia_matrices[py_idx])
        f[py_idx] = I @ a[py_idx] + crf(v[py_idx]) @ I @ v[py_idx]



    # backward pass
    payload_link = int(get_has_payload())
    for i in range(nb, 1 - 1, -1):
        py_idx = i - 1
        tau[py_idx] = f[py_idx][-1]
        if parent(i) != 0:
            f[parent(i) - 1] += adjoint(Xups[py_idx]).transpose() @ f[py_idx]  # inv?

    tau_rne = np.array([t[0] for t in tau[:originalN]])
    return tau_rne