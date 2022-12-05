import time

import ipdb
import numpy as np
import pygame.color
import pyglet.window
import pymunk
import pymunk.pyglet_util
from attrs import define
from jaxtyping import Float, Shaped
from loguru import logger as log
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
    mass: float
    inertia: float
    elasticity: float
    friction: float
    bar_width: float
    # Radius of the segment for the shape.
    radius: float


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


def create_particle(screen_pos: np.ndarray, radius: float, cfg: ParticleCfg, *, rng: np.random.Generator | None = None):
    n_verts = 10
    bevel_radius = 1e-5

    poly_verts = gen_random_poly(radius, n_verts, rng=rng)
    body = pymunk.Body()
    body.position = Vec2d(screen_pos[0], screen_pos[1])

    shape = pymunk.Poly(body, poly_verts.tolist(), radius=bevel_radius)
    shape.density = cfg.density
    shape.friction = cfg.friction
    shape.color = (255, 255, 255, 255)

    return Particle(body, shape)


def create_pusher(spos: np.ndarray, theta: float, cfg: PusherCfg) -> Pusher:
    assert spos.shape == (2,)
    GREEN = (0, 255, 0, 255)
    # TRANSPARENT = (0, 0, 0, 0)

    PUSHER_COLOR = GREEN

    body = pymunk.Body(mass=cfg.mass, moment=cfg.inertia)
    body.position = (spos[0], spos[1])
    # angle is always 0. We put the angle info in the shape.
    body.angle = 0

    # Pusher is perpendicular to push direction.
    theta = theta - np.pi / 2

    offset = np.array([cfg.bar_width / 2.0 * np.cos(theta), cfg.bar_width / 2.0 * np.sin(theta)])

    start = +offset
    end = -offset

    shape = pymunk.Segment(body, start.tolist(), end.tolist(), cfg.radius)
    shape.elasticity = cfg.elasticity
    shape.friction = cfg.friction
    shape.color = PUSHER_COLOR

    return Pusher(body, shape)


def to_screen_pos(pos: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert from normalized position (-0.5, 0.5)^2 to screen coordinates (0, width) x (0, height)
    :param pos: (..., 2)
    """
    assert pos.shape[-1] == 2
    # (0, 1)^2
    pos = pos + 0.5
    # (2, )
    scale_factor = np.array([width, height])
    return pos * scale_factor


def from_screen_pos(pos: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert from screen coordinates (0, width) x (0, height) to normalized position (-0.5, 0.5)^2.
    :param pos: (..., 2)
    """
    assert pos.shape[-1] == 2
    scale_factor = np.array([width, height])
    # (0, 1)^2
    pos = pos / scale_factor
    # (-0.5, 0.5)^2
    return pos - 0.5


def control_info(u: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """
    :param u: (n_arms, 2, n_coord=2)
    :param width:
    :param height:
    """
    spos = to_screen_pos(u, width, height)
    x0s, xfs = spos[:, 0, :], spos[:, 1, :]
    x_diff = xfs - x0s

    # (n_arms, )
    thetas = np.arctan2(x_diff[:, 1], x_diff[:, 0])
    # (n_arms, )
    target_push_lengths = np.linalg.norm(x_diff, axis=1)

    return thetas, target_push_lengths


def control_to_state(u: np.ndarray, width: int, height: int) -> np.ndarray:
    n_arms = u.shape[0]
    spos = to_screen_pos(u, width, height)
    # (n_arms, 2)
    x0s, xfs = spos[:, 0, :], spos[:, 1, :]
    # (n_arms, )
    thetas, target_push_lengths = control_info(u, width, height)

    # (n_arms, 3)
    start_state = np.concatenate([x0s, thetas[:, None]], axis=1)
    assert start_state.shape == (n_arms, 3)

    return start_state


class BiArmSim(pyglet.window.Window):
    """
    Coordinate system:

            ↑
            ╎
      ↑     ╎
      │     ╎
    height  ╎
      │     ╎
      ↓     ╎
            ╎
            ┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌→
                ←---   width   ---→
    """

    def __init__(self, n_arms: int = 2, do_render: bool = False, render_arm: bool = False):
        super().__init__(vsync=False)

        # Sim window parameters. These also define the resolution of the image
        # Ruixiao: I changeed the height, width and the following parameter to match up Terry's paper
        self.width = 512
        self.height = 512
        self.n_arms = n_arms
        self.set_caption("BiArmSim ({} arms)".format(n_arms))

        # Simulation parameters.
        self.bar_width = 80.0
        self.vel_mag = 50.0

        self.global_time = 0.0
        self.particles: list[Particle] = []
        self.particle_num = 150
        self.particle_size = 10

        self.pushers: list[Pusher] = []
        self.space = pymunk.Space(threaded=False)

        self.image = None
        self.graphics_batch = pyglet.graphics.Batch()
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES

        self.draw_options.shape_outline_color = (255, 255, 255, 0)

        self.rng = np.random.default_rng(seed=84123)
        self.do_render = do_render
        self.render_arm = render_arm

        self.create_world()

    def create_world(self):
        self.space.gravity = Vec2d(0.0, 0.0)
        self.space.damping = 0.0001  # quasi-static. low value is higher damping.
        # Inner iterations for contact solver. Default is 10.
        self.space.iterations = 10

        self.add_particles(self.particle_num, self.particle_size)
        self.advance_s(2.0)

        self.render_to_screen()

        # Somehow need to call this twice if we want it to show up on screen.
        if self.do_render:
            self.render_to_screen()

    def to_screen_pos(self, pos: np.ndarray) -> np.ndarray:
        return to_screen_pos(pos, self.width, self.height)

    def from_screen_pos(self, pos: np.ndarray) -> np.ndarray:
        return from_screen_pos(pos, self.width, self.height)

    def add_particle(self, radius: float):
        particle_cfg = ParticleCfg(density=0.01, friction=0.6)

        # pos = self.rng.uniform(100, 400, size=(2,))
        pos = self.rng.uniform(-0.3, 0.3, size=(2,))
        screen_pos = self.to_screen_pos(pos)
        particle = create_particle(screen_pos, radius, particle_cfg, rng=self.rng)

        self.space.add(particle.body, particle.shape)
        self.particles.append(particle)

    def add_particles(self, n_particles: int, radius: float):
        for i in range(n_particles):
            self.add_particle(radius)

        # log.info("particle mass: {}".format(self.particles[0].body.mass))

    def remove_particles(self):
        for particle in self.particles:
            self.space.remove(particle.shape, particle.body)
        self.particles = []

    def add_pusher(self, spos: np.ndarray, theta: np.ndarray):
        mass, inertia, elasticity, friction = 100.0, 1e12, 0.1, 0.6
        radius = 5
        bar_width = 80.0
        pusher_cfg = PusherCfg(mass, inertia, elasticity, friction, bar_width, radius)
        pusher = create_pusher(spos, theta, pusher_cfg)
        self.space.add(pusher.body, pusher.shape)
        self.pushers.append(pusher)
        #
        # RED = (255, 0, 0)
        # self.circle = pyglet.shapes.Circle(spos[0], spos[1], 50, color=RED, batch=self.graphics_batch)
        # log.info("pusher mass: {}, inertia: {}".format(self.pushers[0].body.mass, self.pushers[0].body.moment))

    def add_pushers(self, sposes: np.ndarray, thetas: np.ndarray):
        """
        :param sposes: (n_pushers, 2)
        :param thetas: (n_pushers, )
        """
        # log.info("thetas: {}".format(thetas))
        assert sposes.shape == (self.n_arms, 2)
        for ii in range(self.n_arms):
            self.add_pusher(sposes[ii], thetas[ii])

        # log.info("spos[0]: {}, pusher pos: {}".format(sposes[0], self.pushers[0].body.position))

    def remove_pushers(self) -> None:
        for pusher in self.pushers:
            self.space.remove(pusher.shape, pusher.body)
        self.pushers = []

    def apply_control(
        self, u: Float[np.ndarray, "8"], render_at_length: float | None = None, render_every: int | None = None
    ) -> tuple[list[Image], dict]:
        """
        Apply a control action, run the simulation forward then return.
        :param u: (8, ) within (-0.5, 0.5) OR (n_arms, 2, N_COORDS=2), where
            Initial (2, ) denotes which arm.
            Final (2, 2) is [ (x0, y0), (xf, yf) ].
        :param render_at_length: How often to render.
        """
        if u.shape == (self.n_arms * 2,):
            u = u.reshape((self.n_arms, 2, 2))

        assert u.shape == (self.n_arms, 2, 2)
        spos = self.to_screen_pos(u)

        # (2, 2)
        x0s, xfs = spos[:, 0, :], spos[:, 1, :]

        # (2, 2)
        x_diff = xfs - x0s

        thetas = np.arctan2(x_diff[:, 1], x_diff[:, 0])
        target_push_lengths = np.linalg.norm(x_diff, axis=1)
        assert thetas.shape == target_push_lengths.shape == (self.n_arms,)

        # (n_pushers, 2)
        force_vecs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
        # log.info("force_vecs: {} {}".format(force_vecs[0], force_vecs[1]))

        # Add the pushers.
        self.add_pushers(x0s, thetas)
        n_pushers = len(self.pushers)

        goal_atol = 3.0
        max_force_thresh = 100_000

        fps = 60.0
        step_dt = 1 / fps

        v_noms = np.full(n_pushers, 50.0)

        max_steps = 10_000
        v_err_hist = np.zeros((max_steps, n_pushers))

        images = []
        pusher_states = []

        save_at_length = 0.0

        steps = 0
        while True:
            t1 = time.time()
            # Check if we have reached the goal.
            reached_goal, push_lengths = self.has_reached_goal(x0s, target_push_lengths, atol=goal_atol)
            if np.any(reached_goal):
                # log.info("Reached goal!")
                break

            if render_at_length is not None:
                push_length = push_lengths[0]
                if push_length >= save_at_length:
                    self.clear_screen()
                    self.debug_draw()
                    images.append(self.get_image())

                    # (n_pushers, 2)
                    pusher_poss = np.stack([pusher.body.position for pusher in self.pushers], axis=0)
                    # (n_pushers, )
                    pusher_angles = thetas + np.stack([pusher.body.angle for pusher in self.pushers], axis=0)
                    pusher_state = np.concatenate([pusher_poss, pusher_angles[:, None]], axis=1)
                    pusher_states.append(pusher_state)

                    save_at_length += render_at_length
            elif render_every is not None and steps % render_every == 0:
                # Just for visualization.
                self.render_to_screen()

            vels = np.stack([np.linalg.norm(pusher.body.velocity) for pusher in self.pushers], axis=0)
            v_err_hist[steps] = vels - v_noms
            # (n_pushers, )
            forces = self.pusher_controller(vels, v_noms, v_err_hist[: steps + 1])

            # log.info("v: {}, v_nom: {}".format(vels, v_noms))

            # log.info("vels: {}, forces: {}".format(vels, forces))
            if not np.all(np.isfinite(forces)):
                ipdb.set_trace()

            max_force = np.max(np.abs(forces))
            if max_force > max_force_thresh:
                # log.info("Max force!")
                break

            # (n_pushers, 2)
            forces = forces[:, None] * force_vecs
            forces = [(force[0], force[1]) for force in forces]

            [pusher.body.apply_force_at_local_point(force) for force, pusher in zip(forces, self.pushers)]

            self.space.step(step_dt)
            self.global_time += step_dt
            steps += 1

            if steps >= max_steps:
                # log.info("Hit max steps!")
                break

            t2 = time.time()
            ms = (t2 - t1) * 1e3
            # log.info("Loop time: {:.2f} ms".format(ms, ))

        # Wait 1 second in sim time to slow down moving pieces, and render.
        self.advance_s(1.0)
        self.remove_pushers()
        # self.render_to_screen()

        if len(pusher_states) > 0:
            pusher_states = np.stack(pusher_states, axis=0)

        info = {"steps": steps, "pusher_state": pusher_states}

        return images, info

    def has_reached_goal(
        self, x0s: np.ndarray, target_push_length: np.ndarray, atol: float = 3.0
    ) -> tuple[np.ndarray, np.ndarray]:
        assert len(x0s) == len(target_push_length) == len(self.pushers)
        n_pushers = len(self.pushers)

        has_reached = np.zeros(n_pushers, dtype=bool)
        push_lengths = []
        for ii in range(n_pushers):
            spos = self.pushers[ii].body.position
            push_length = np.linalg.norm(spos - x0s[ii])
            push_lengths.append(push_length)

            has_reached[ii] = np.abs(target_push_length[ii] - push_length) < atol
        push_lengths = np.array(push_lengths)

        return has_reached, push_lengths

    def pusher_controller(self, vs: np.ndarray, v_noms: np.ndarray, v_err_hist: np.ndarray):
        """
        :param vs: (n_pushers, )
        :param v_noms: (n_pushers, )
        :param v_err_hist: (hist_length, n_pushers)
        :return:
        """
        k_p, k_i, k_d = 1_000.0, 100.0, 1.0
        # k_p, k_i, k_d = 0., 100., 0.

        v_err = vs - v_noms
        a_err = 0.0

        if len(v_err_hist) >= 2:
            a_nom = 0.0
            pred_a = v_err_hist[-1] - v_err_hist[-2]
            a_err = pred_a - a_nom

        v_err_int = np.sum(v_err_hist, axis=0)

        forces = -k_p * v_err - k_i * v_err_int - k_d * a_err
        assert forces.shape == vs.shape

        return forces

    def advance_s(self, duration_s: float) -> None:
        t = 0
        step_dt = 1 / 60.0
        while t < duration_s:
            self.space.step(step_dt)
            t += step_dt

    def on_draw(self):
        pass
        # self.graphics_batch.draw()

        # # This isn't called anywhere.
        # raise NotImplementedError("on_draw called!")
        # self.render()

    def clear_screen(self):
        self.clear()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()

    def debug_draw(self, draw_pusher: bool = False):
        # log.info("use_chipmunk_debug_draw: {}".format(self.draw_options._use_chipmunk_debug_draw))

        if not self.render_arm:
            # Remove pusher, draw, then add pusher back.
            for pusher in self.pushers:
                self.space.remove(pusher.shape, pusher.body)

        self.space.debug_draw(self.draw_options)

        if not self.render_arm:
            for pusher in self.pushers:
                self.space.add(pusher.shape, pusher.body)

        # # Only draw the particles.
        # for particle in self.particles:
        #     self.draw_options.draw_shape(particle.shape)

        # if draw_pusher:
        #     for pusher in self.pushers:
        #         self.draw_options.draw_shape(pusher.shape)

    def render_to_screen(self):
        pyglet.clock.tick()

        self.clear_screen()
        self.debug_draw()
        self.dispatch_events()  # necessary to refresh somehow....
        self.dispatch_event("on_draw")
        self.flip()
        # self.update_image()

    def get_image(self) -> Image:
        pitch = -(self.width * len("RGB"))
        img_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data("RGB", pitch=pitch)
        pil_im = Image.frombytes("RGB", (self.width, self.height), img_data)
        # return pil_im.resize((32, 32))
        return pil_im

    def get_image_grayscale_np(self) -> np.ndarray:
        pil_im = self.get_image().convert("L")
        cv_image = np.array(pil_im).copy()
        return cv_image

    def get_image_np(self) -> np.ndarray:
        pil_im = self.get_image()
        cv_image = np.array(pil_im)[:, :, ::-1].copy()
        return cv_image

    def update_image(self):
        self.image = self.get_image()

    def get_current_image(self):
        return self.image

    def refresh(self, particle_num: int | None = None):
        if particle_num is None:
            particle_num = self.particle_num

        self.remove_particles()
        self.add_particles(particle_num, self.particle_size)
        self.advance_s(2.0)  # Give some time for collision pieces to stabilize.
        self.render_to_screen()
