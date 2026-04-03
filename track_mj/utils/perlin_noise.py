import jax
import jax.numpy as jnp


@jax.jit
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@jax.jit
def _lerp(a, b, t):
    return a + t * (b - a)


def _get_gradient(h, x, y):
    vectors = jnp.array([[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]])
    g = vectors[h % 8]
    return g[0] * x + g[1] * y


@jax.jit
def _perlin_noise_point(coords, p):
    coords_floor = jnp.floor(coords).astype(jnp.int32)
    coords_fract = coords - coords_floor
    ix0, iy0 = coords_floor[0], coords_floor[1]
    fx0, fy0 = coords_fract[0], coords_fract[1]
    ix1, iy1 = ix0 + 1, iy0 + 1
    fx1, fy1 = fx0 - 1, fy0 - 1
    u = _fade(fx0)
    v = _fade(fy0)
    ix0_masked, iy0_masked = ix0 & 255, iy0 & 255
    ix1_masked, iy1_masked = ix1 & 255, iy1 & 255
    h00 = p[p[ix0_masked] + iy0_masked]
    h10 = p[p[ix1_masked] + iy0_masked]
    h01 = p[p[ix0_masked] + iy1_masked]
    h11 = p[p[ix1_masked] + iy1_masked]
    n00 = _get_gradient(h00, fx0, fy0)
    n10 = _get_gradient(h10, fx1, fy0)
    n01 = _get_gradient(h01, fx0, fy1)
    n11 = _get_gradient(h11, fx1, fy1)
    x1 = _lerp(n00, n10, u)
    x2 = _lerp(n01, n11, u)
    return _lerp(x1, x2, v)


@jax.jit
def _fbm_noise_point_fori(coords, octaves, persistence, lacunarity, p):
    def loop_body(_, carry):
        total, amplitude, frequency = carry

        noise_val = _perlin_noise_point(coords * frequency, p) * amplitude

        new_total = total + noise_val
        new_amplitude = amplitude * persistence
        new_frequency = frequency * lacunarity

        return (new_total, new_amplitude, new_frequency)

    initial_carry = (0.0, 1.0, 1.0)  # (initial_total, initial_amplitude, initial_frequency)

    final_carry = jax.lax.fori_loop(0, octaves, loop_body, initial_carry)
    final_total = final_carry[0]

    max_value = jnp.where(
        persistence == 1.0, octaves.astype(jnp.float32), (1.0 - persistence**octaves) / (1.0 - persistence)
    )

    return final_total / max_value


@jax.jit
def generate_terrain_jax(key, scale, octaves, persistence, lacunarity):
    p = jax.random.permutation(key, jnp.arange(256, dtype=jnp.int32))
    p_shuffled = jnp.concatenate([p, p])

    x = jnp.linspace(0, 255, 256)
    y = jnp.linspace(0, 255, 256)
    xx, yy = jnp.meshgrid(x, y)

    coords_x = (xx - 256 / 2) / scale
    coords_y = (yy - 256 / 2) / scale

    coords_grid = jnp.stack([coords_x, coords_y], axis=-1)

    vmap_fbm = jax.vmap(
        jax.vmap(_fbm_noise_point_fori, in_axes=(0, None, None, None, None)),
        in_axes=(0, None, None, None, None),
    )

    noise_map = vmap_fbm(coords_grid, octaves, persistence, lacunarity, p_shuffled)

    return (noise_map + 1) / 2
