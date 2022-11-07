import random
import time

import cv2
import ipdb
import numpy as np
import PIL
import pygame
import pyglet
import pymunk
import pymunk.pyglet_util
from pygame.color import *
from pygame.locals import *
from pyglet.gl import *
from pyglet.window import key, mouse
from pymunk import Vec2d
from scipy.spatial import ConvexHull

"""Main Simulation class for carrots.

The attributes and methods are quite straightforward, see carrot_sim.py
for basic usage.
"""


class TwoArmSim(pyglet.window.Window):
    def __init__(self):
        pyglet.window.Window.__init__(self, vsync=False)

        # Sim window parameters. These also define the resolution of the image
        self.width = 500
        self.height = 500
        self.set_caption("CarrotSim")

        # Simulation parameters.
        self.bar_width = 80.0
        self.vel_mag = 50.0
        self.velocity = np.array([0, 0])

        self.global_time = 0.0
        self.onion_pieces = []
        self.onion_num = 120
        self.onion_size = 12

        self.body0: pymunk.Body | None = None
        self.body1: pymunk.Body | None = None
        self.shape0: pymunk.Shape | None = None
        self.shape1: pymunk.Shape | None = None

        self.space = pymunk.Space()

        self.image = None
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES
        self.graphics_batch = pyglet.graphics.Batch()

        self.create_world()

        # User flags.
        # If this flag is enabled, then the rendering will be done at every
        # simulation timestep. This makes the sim much slower, but is better
        # for visualizing the pusher continuously.
        self.RENDER_EVERY_TIMESTEP = True

    """
    1. Methods for Generating and Removing Sim Elements
    """

    """
    1.1 Methods for Initializing World
    """

    def create_world(self):
        self.space.gravity = Vec2d(0, 0)  # planar setting
        self.space.damping = 0.0001  # quasi-static. low value is higher damping.
        self.space.iterations = 5  # TODO(terry-suh): re-check. what does this do?
        self.space.color = pygame.color.THECOLORS["white"]
        self.add_onions(self.onion_num, self.onion_size)
        self.wait(1.0)  # give some time for colliding pieces to stabilize.
        self.render()

    """
    1.2 Methods for Generating and Removing Onion Pieces
    """

    def generate_random_poly(self, center, radius):
        """
        Generates random polygon by random sampling a circle at "center" with
        radius "radius". Its convex hull is taken to be the final shape.
        """
        k = 10
        r = radius * np.random.rand(k, 1)
        theta = 2 * np.pi * np.random.rand(k, 1)
        p = np.hstack((r * np.cos(theta), r * np.sin(theta)))
        return np.take(p, ConvexHull(p).vertices, axis=0)

    def create_onion(self, radius):
        """
        Create a single onion piece by defining its shape, mass, etc.
        """
        mass = 10.

        points = self.generate_random_poly((0, 0), radius)
        inertia = pymunk.moment_for_poly(mass, points.tolist(), (0, 0))
        body = pymunk.Body(mass, inertia)
        # TODO(terry-suh): is this a reasonable distribution?
        body.position = Vec2d(random.randint(100, 400), random.randint(100, 400))
        # body.position = Vec2d(random.randint(0, 500), random.randint(0, 500))

        shape = pymunk.Poly(body, points.tolist())
        shape.friction = 0.6
        shape.color = (255, 255, 255, 255)
        return body, shape

    def add_onion(self, radius):
        """
        Create and add a single onion piece to the sim.
        """
        onion_body, onion_shape = self.create_onion(radius)
        self.space.add(onion_shape.body, onion_shape)
        self.onion_pieces.append([onion_body, onion_shape])

    def add_onions(self, num_pieces, radius):
        """
        Create and add multiple onion pieces to sim.
        """
        for i in range(num_pieces):
            self.add_onion(radius)

    def remove_onions(self):
        """
        Remove all onion pieces in sim.
        """
        for i in range(len(self.onion_pieces)):
            onion_body = self.onion_pieces[i][0]
            onion_shape = self.onion_pieces[i][1]
            self.space.remove(onion_shape, onion_shape.body)
        self.onion_pieces = []

    """
    1.3 Methods for Generating and Removing Pusher
    """

    def create_bar(self, position, theta):
        """
        Create a single bar by defining its shape, mass, etc.
        """
        mass = 100.0
        body = pymunk.Body(mass, float("inf"))
        # body.position = Vec2d(position[0], position[1])

        theta = theta - np.pi / 2  # pusher is perpendicular to push direction.
        v = np.array([self.bar_width / 2.0 * np.cos(theta), self.bar_width / 2.0 * np.sin(theta)])

        start = np.array(position) + v + np.array([self.width * 0.5, self.height * 0.5])
        end = np.array(position) - v + np.array([self.width * 0.5, self.height * 0.5])

        shape = pymunk.Segment(body, start.tolist(), end.tolist(), 5)
        shape.elasticity = 0.1
        shape.friction = 0.6
        shape.color = (0, 255, 0, 255)
        return body, shape

    def add_bar(self, position, theta):
        """
        Create and add a single bar to the sim.
        """
        pusher_body, pusher_shape = self.create_bar(position, theta)
        self.space.add(pusher_body, pusher_shape)

        return pusher_body, pusher_shape

    def remove_bar(self, body, shape):
        """
        Remove bar from simulation.
        """
        self.space.remove(body, shape)

    """
    2. Methods for Updating and Rendering Sim
    """

    """
    2.1 Methods Related to Updating
    """

    # Update the sim by applying action, progressing sim, then stopping.
    def update(self, u):
        """
        Once given a control action, run the simulation forward and return.

        u: (8, ). (-0.5, 0.5)
        """
        u = u.reshape((2, 4))

        # Parse into integer coordinates
        uxi = float(self.width) * u[:, 0]
        uyi = float(self.height) * u[:, 1]
        uxf = float(self.width) * u[:, 2]
        uyf = float(self.height) * u[:, 3]

        # transform into angular coordinates
        thetas = np.arctan2(uyf - uyi, uxf - uxi)
        lengths = np.linalg.norm(np.stack([uxf - uxi, uyf - uyi], axis=1), axis=1)
        print("lengths: {}".format(lengths))

        start0 = np.array([uxi[0], uyi[0]])
        start1 = np.array([uxi[1], uyi[1]])

        # add the bar and set velocity.
        self.body0, self.shape0 = self.add_bar(start0, thetas[0])
        self.body1, self.shape1 = self.add_bar(start1, thetas[1])

        print("body0 position: {} vs {}".format(start0, self.body0.position))
        print("body1 position: {} vs {}".format(start1, self.body1.position))

        vels = [[0, 0], [0, 0]]

        tolerance = 3.0  # stopping criteria. See below.
        step_dt = 1 / 60.0  # Sim bandwidth

        # Step through the simulation
        steps = 0
        while True:
            # 1: Check if we have reached the goal.
            # dx0 = np.array(self.body0.position) - start0
            # dx1 = np.array(self.body1.position) - start1
            dx0 = self.body0.position
            dx1 = self.body1.position
            push_length0 = np.linalg.norm(dx0)
            push_length1 = np.linalg.norm(dx1)

            reach0 = np.abs(push_length0 - lengths[0]) < tolerance
            reach1 = np.abs(push_length1 - lengths[1]) < tolerance

            print("")
            print("push lengths: {} {}".format(push_length0, push_length1))
            print("     lengths: {} {}".format(lengths[0], lengths[1]))

            if reach0 or reach1:
                print("break!")
                break

            # If user wants to render at every simulation timestep, then
            # render here as well.
            # TODO(terry-suh): there should be an option in between where the
            # user specifies dt of rendering.
            # ipdb.set_trace()
            if self.RENDER_EVERY_TIMESTEP:
                self.render()

            # self.body0.velocity = vels[0].tolist()
            # self.body1.velocity = vels[1].tolist()
            v_d = 100
            vels[0].append(np.linalg.norm(self.body0.velocity) - v_d)
            vels[1].append(np.linalg.norm(self.body1.velocity) - v_d)
            
            def pid(vs):
                k_p, k_i, k_d = -100, -100, -100
                f = []
                for v in vs:
                    force = k_p * v[-1] + k_i * sum(v) + k_d * (v[-1] - v[-2])
                    f.append(force)
                return f
            
            f = pid(vels)
            print(f)
            
            if max(abs(f[0]), abs(f[1])) > 200000:
                break

            self.body0.apply_force_at_local_point((f[0] * np.cos(thetas[0]), f[0] * np.sin(thetas[0])))
            self.body1.apply_force_at_local_point((f[1] * np.cos(thetas[1]), f[1] * np.sin(thetas[1])))

            self.space.step(step_dt)
            self.global_time += step_dt

            steps += 1

        # Wait 1 second in sim time to slow down moving pieces, and render.
        self.wait(1.0)
        self.remove_bar(self.body0, self.shape0)
        self.remove_bar(self.body1, self.shape1)
        self.render()

        return None

    def wait(self, time):
        """
        Wait for some time in the simulation. Gives some time to stabilize bodies in collision.
        """
        t = 0
        step_dt = 1 / 60.0
        while t < time:
            self.space.step(step_dt)
            t += step_dt

    """
    2.2 Methods related to rendering
    """

    def on_draw(self):
        self.render()

    def render(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        self.space.debug_draw(self.draw_options)
        self.dispatch_events()  # necessary to refresh somehow....
        self.flip()
        self.update_image()

    """
    2.3 Methods related to image publishing
    """

    def update_image(self):
        pitch = -(self.width * len("RGB"))
        img_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data("RGB", pitch=pitch)
        pil_im = PIL.Image.frombytes("RGB", (self.width, self.height), img_data)
        cv_image = np.array(pil_im)[:, :, ::-1].copy()
        self.image = cv_image

    def get_current_image(self):
        return self.image

    """
    3. Methods for External Commands
    """

    def refresh(self):
        self.remove_onions()
        self.add_onions(self.onion_num, self.onion_size)
        self.wait(1.0)  # Give some time for collision pieces to stabilize.
        self.render()

    def change_onion_num(self, onion_num):
        self.onion_num = onion_num
        self.refresh()

    def change_onion_size(self, onion_size):
        self.onion_size = onion_size
        self.refresh()
