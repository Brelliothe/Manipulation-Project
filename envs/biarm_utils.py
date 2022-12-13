import numpy as np
import pymunk
from attr import define
from pymunk import Vec2d
from scipy.spatial import ConvexHull

from utils.angles import get_rotmat


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
    assert np.all(np.abs(pos) <= 0.5)
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


def biarm_state_to_centered(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert biarm state to tuple of (center_x, center_y, center_rot), (left_rot, right_rot), arm_sep"""
    assert state.shape == (2, 3)

    # The first arm should have a smaller x coordinate, i.e., left arm.
    assert state[0, 0] < state[1, 0]

    center_pos = np.mean(state[:, :2], axis=0)
    assert center_pos.shape == (2,)

    # (2,)
    diff_pos = state[1, :2] - state[0, :2]
    center_rot = np.arctan2(diff_pos[1], diff_pos[0])

    center_state = np.array([center_pos[0], center_pos[1], center_rot])
    assert center_state.shape == (3,)

    rots = state[:, 2] - center_rot
    assert rots.shape == (2,)

    arm_sep = np.linalg.norm(diff_pos)

    return center_state, rots, arm_sep


def centered_to_biarm_state(center_state: np.ndarray, rots: np.ndarray, arm_sep: float) -> np.ndarray:
    c_x, c_y, c_rot = center_state

    # [left, right]
    # (n_arm=2, 2)
    pos = np.array([[-arm_sep / 2, 0], [arm_sep / 2, 0]])
    pos = np.squeeze(get_rotmat(c_rot) @ pos[:, :, None], axis=2)
    assert pos.shape == (2, 2)
    pos = pos + np.array([c_x, c_y])

    state_l = np.array([pos[0, 0], pos[0, 1], c_rot + rots[0]])
    state_r = np.array([pos[1, 0], pos[1, 1], c_rot + rots[1]])

    state = np.stack([state_l, state_r], axis=0)
    assert state.shape == (2, 3)

    # x coordinate should be ascending.
    assert state[0, 0] < state[1, 0]

    return state


def state_to_goal(state: np.ndarray, push_length: float) -> np.ndarray:
    assert state.shape == (3,)
    px, py, theta = state
    dx, dy = np.cos(theta) * push_length, np.sin(theta) * push_length

    return np.array([px + dx, py + dy])


def state_to_control(state: np.ndarray, push_length: float, width: float) -> np.ndarray:
    assert state.shape == (2, 3)

    goal_states = np.stack([state_to_goal(s, push_length) for s in state], axis=0)
    assert goal_states.shape == (2, 2)

    # [0, width]
    u_screen = np.stack([state[:, :2], goal_states], axis=1)
    # [0, width] -> [-0.5, 0.5]
    u = u_screen / width - 0.5
    return u


def modify_pushlength(u: np.ndarray, A_length: float, n_pred_lengths: int) -> np.ndarray:
    # A_length = 32 * 32 * push_frames / 512
    # u: (n_arms, 2, nx=2)
    WIDTH, HEIGHT = 512, 512
    push_frames = A_length * 512 / (32 * 32)

    thetas, real_pushlengths = control_info(u, WIDTH, HEIGHT)
    desired_pushlengths = n_pred_lengths * push_frames * 32

    coeff = desired_pushlengths / real_pushlengths
    coeff = coeff[:, None]

    new_u = u.copy()
    new_u[:, 1, :] = u[:, 0, :] + coeff * (u[:, 1, :] - u[:, 0, :])

    assert np.all(np.abs(new_u) < 0.5)

    return new_u
