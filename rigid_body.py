import numpy as np
from quaternion_computation import normalize_quaternion, quaternion_multiply, quaternion_to_rotation

class RigidBody(object):
    def __init__(self):
        # Rigid Body Parameters
        self.vertices= (
            (-5,-5, 5),
            (-5, 5, 5),
            ( 5, 5, 5),
            ( 5,-5, 5),
            (-5,-5,-5),
            (-5, 5,-5),
            ( 5, 5,-5),
            ( 5,-5,-5)
            )

        # Mass of the object
        self.m = 100.0
        self.m_inverse = 1 / self.m

        # Inertia for the object
        self.iBody = np.zeros((3, 3))
        self.iBody[0, 0] = (1.0 / 12.0) * (self.m * ((10 * 10) + (10 * 10)))
        self.iBody[1, 1] = (1.0 / 12.0) * (self.m * ((10 * 10) + (10 * 10)))
        self.iBody[2, 2] = (1.0 / 12.0) * (self.m * ((10 * 10) + (10 * 10)))
        
        # Initial state values 
        self.velocity = np.array([10, 10, 10])
        self.position = np.array([-100, 80, -100])
        self.quaternion = np.array([ -0.5704403, -0.5704403, -0.5704403, 0.1542514 ])
        self.angular_velocity = np.array([0.1, 0.1, 0.5])

        # Simulation Parameters
        self.gravity_acceleration = np.array([0, -9.8, 0]) # Gravitation acceleration
        self.ks = 1000 # Spring Stiffness
        self.timestep = 1/30.0 # Amount of time by which the simulation advances
        self.nsubsteps = 2 # Number of steps within a timestep for finer calculations
        self.floor_y = 0.0 # Position of the floor


    def simulate_one_substep(self):
        # detect collision and determine vertices that collided
        is_collision, collision_vertices = self.collision_detection()
        # calculate the total forces and torques exerted on the body
        total_force, total_torque = self.calculate_forces_and_torques(is_collision, collision_vertices)
        # calculate the linear and angular accelerations based on the forces and torques
        linear_acceleration, angular_acceleration = self.calculate_accelerations(total_force, total_torque)
        # integrate everything to the get new state variables
        new_v, new_x, new_w, new_q = self.integrate(linear_acceleration, angular_acceleration)
        
        # update the class variables
        self.velocity = new_v
        self.position = new_x
        self.angular_velocity = new_w
        self.quaternion = new_q

    def collision_detection(self):
        is_collision = False
        collision_vertices = []

        # your code begins here
        rotation_m = quaternion_to_rotation(self.quaternion)
      
        for i in range(8):
            v_after_rotate = (rotation_m @ np.array(self.vertices[i]).reshape(-1,1)).flatten()
            abs_vert = v_after_rotate+self.position
            if abs_vert[1] <= self.floor_y:
                is_collision = True 
                collision_vertices.append(v_after_rotate)
        
        return is_collision, collision_vertices
    
    def calculate_forces_and_torques(self, is_collision, collision_vertices):
        total_force = np.array([0.0, 0.0, 0.0])
        total_torque = np.array([0.0, 0.0, 0.0])
        
        # your code begins here 
        rotation_m = quaternion_to_rotation(self.quaternion)
        if not is_collision:
            total_force = self.gravity_acceleration * self.m
            ## torque caused by gravity on the 8 vertices cancel out 
            # for i in range(8):
            #     v_after_rotate = (rotation_m @ np.array(self.vertices[i]).reshape(-1,1)).flatten()
            #     torque = np.cross(v_after_rotate, total_force)
            #     total_torque += torque
        else:
            total_force = self.gravity_acceleration * self.m
            ## torque caused by gravity on the 8 vertices cancel out 
            # for i in range(8):
            #     v_after_rotate = (rotation_m @ np.array(self.vertices[i]).reshape(-1,1)).flatten()
            #     torque = np.cross(v_after_rotate, total_force)
            #     total_torque += torque
            for i in range(len(collision_vertices)):
                ## penalty method for collision handling
                collision_force = abs(collision_vertices[i][1])*self.ks*np.array([0.0,1.0,0.0])
                total_force += collision_force
                torque = np.cross(collision_vertices[i], collision_force)
                total_torque += torque
 
        return total_force, total_torque
    
    def calculate_accelerations(self, force, torque):
        linear_acceleration = np.array([0.0, 0.0, 0.0])
        angular_acceleration = np.array([0.0, 0.0, 0.0])

        # your code begins here 
        linear_acceleration =  force * self.m_inverse
        rotation_m = quaternion_to_rotation(self.quaternion)
        I = rotation_m @ self.iBody @ rotation_m.T
        angular_acceleration = (np.linalg.inv(I) @ (torque-np.cross(self.angular_velocity, (I @ self.angular_velocity.reshape(-1,1)).flatten())).reshape(-1,1)).flatten()

        return linear_acceleration, angular_acceleration
        
    def integrate(self, linear_acceleration, angular_acceleration):
        # your code begins here

        # default values (the following lines can be modified)
        new_v = self.velocity + self.timestep*linear_acceleration
        new_x = self.position + self.timestep*new_v
        new_w = self.angular_velocity + self.timestep*angular_acceleration
        new_q = normalize_quaternion(self.quaternion + np.dot(self.timestep/2*np.array([0.0,new_w[0],new_w[1],new_w[2]]), self.quaternion))
        return new_v, new_x, new_w, new_q

