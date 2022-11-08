import numpy as np
import pygame.color
import pyglet.window
import pymunk
import pymunk.pyglet_util
from attrs import define
from PIL import Image
from pyglet import gl
from pymunk import Vec2d
from scipy.spatial import ConvexHull


@define
class Pusher:
    body: pymunk.Body
    shape: pymunk.Shape


@define
class Particle:
    body: pymunk.Body
    shape: pymunk.Shape


@define
class ParticleCfg:
    density: float
    friction: float


@define
class PusherCfg:
    density: float
    friction: float
    bar_width: float


def gen_random_poly(radius: float, n_verts: int, *, rng: np.random.Generator | None = None):
    """Generate random polygon by sampling in a circle and taking the convex hull.
    (n, 2).
    """
    if rng is None:
        rng = np.random.default_rng(seed=12323)

    min_r = 1e-3
    r = rng.uniform(min_r, radius, size=n_verts)
    theta = rng.uniform(0, 2 * np.pi, size=n_verts)
    # (num, 2)
    points = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    v_idxs = ConvexHull(points).vertices

    return np.take(points, v_idxs, axis=0)


def create_particle(pos: np.ndarray, radius: float, cfg: ParticleCfg, *, rng: np.random.Generator | None = None):
    n_verts = 10
    bevel_radius = 1e-5

    poly_verts = gen_random_poly(radius, n_verts, rng=rng)
    body = pymunk.Body()
    body.position = Vec2d(pos[0], pos[1])

    shape = pymunk.Poly(body, poly_verts.tolist(), radius=bevel_radius)
    shape.density = cfg.density
    shape.friction = cfg.friction
    shape.color = (255, 255, 255, 255)

    return Particle(body, shape)


# def create_pusher(pos: np.ndarray, theta: float, cfg: PusherCfg):
#     body = pymunk.Body()
#     body.position = pos
#
#     # Pusher is perpendicular to push direction.
#     theta = theta - np.pi / 2
#     v = np.array([cfg.bar_width / 2.0 * np.cos(theta), cfg.bar_width / 2.0 * np.sin(theta)])
#
#     start = pos + v + np.array([cfg.width * 0.5, self.height * 0.5])
#     end = pos - v + np.array([self.width * 0.5, self.height * 0.5])


class BiArmSim(pyglet.window.Window):
    def __init__(self):
        super().__init__(vsync=False)

        # Sim window parameters. These also define the resolution of the image
        self.width = 500
        self.height = 500
        self.set_caption("BiArmSim")

        # Simulation parameters.
        self.bar_width = 80.0
        self.vel_mag = 50.0

        self.global_time = 0.0
        self.particles: list[Particle] = []
        self.particle_num = 120
        self.particle_size = 12

        self.pushers: list[Pusher] = []
        self.space = pymunk.Space()

        self.image = None
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES
        self.graphics_batch = pyglet.graphics.Batch()

        self.rng = np.random.default_rng(seed=84123)

        self.create_world()

        self.render_every = 2

    def create_world(self):
        self.space.gravity = Vec2d(0.0, 0.0)
        self.space.damping = 0.0001  # quasi-static. low value is higher damping.
        # Inner iterations for contact solver. Default is 10.
        self.space.iterations = 10
        self.space.color = pygame.color.THECOLORS["white"]

        self.add_particles(self.particle_num, self.particle_size)
        self.advance_s(1.0)
        self.render()

    def add_particle(self, radius: float):
        particle_cfg = ParticleCfg(density=1.0, friction=0.6)

        pos = self.rng.uniform(100, 400, size=(2,))
        particle = create_particle(pos, radius, particle_cfg, rng=self.rng)

        self.space.add(particle.body, particle.shape)
        self.particles.append(particle)

    def add_particles(self, n_particles: int, radius: float):
        for i in range(n_particles):
            self.add_particle(radius)

    def remove_particles(self):
        for particle in self.particles:
            self.space.remove(particle.shape, particle.body)
        self.particles = []

    def advance_s(self, duration_s: float) -> None:
        t = 0
        step_dt = 1 / 60.0
        while t < duration_s:
            self.space.step(step_dt)
            t += step_dt

    def on_draw(self):
        self.render()

    def render(self):
        self.clear()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()
        self.space.debug_draw(self.draw_options)
        self.dispatch_events()  # necessary to refresh somehow....
        self.flip()
        self.update_image()

    def update_image(self):
        pitch = -(self.width * len("RGB"))
        img_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data("RGB", pitch=pitch)
        pil_im = Image.frombytes("RGB", (self.width, self.height), img_data)
        cv_image = np.array(pil_im)[:, :, ::-1].copy()
        self.image = cv_image

    def get_current_image(self):
        return self.image

    def refresh(self):
        # pass
        self.remove_particles()
        self.add_particles(self.particle_num, self.particle_size)
        self.wait(1.0)  # Give some time for collision pieces to stabilize.
        self.render()
